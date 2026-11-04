"""Prepare an ACS/CCS all_cv union-cohort fold with naturally missing views.

Samples are kept when at least one view is observed. Transform fitting,
feature selection and neutral-reference estimates use training patients only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import KFold, StratifiedKFold


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, default=Path('data/Prepared_sc_Data.h5ad'))
    p.add_argument('--aux-store', type=Path, default=None,
                   help='Legacy HDF5 store: /ck/<sample_id>, /neutro/<sample_id>, etc.')
    p.add_argument('--aux-views', nargs='*', default=['ck', 'neutro', 'proteo', 'clinical'])
    p.add_argument('--sample-metadata-csv', type=Path, default=None,
                   help='Optional sample_id,patient_id[,group] table, needed for auxiliary-only samples with arbitrary IDs')
    p.add_argument('--output', type=Path, default=Path('results/acs_ccs/prepared'))
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--folds', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--hvg', type=int, default=500)
    p.add_argument('--hvg-method', choices=['seurat_v3', 'variance'], default='seurat_v3')
    p.add_argument('--sample-column', default='sample_id')
    p.add_argument('--library-column', default='library')
    p.add_argument('--patient-column', default='patient_id',
                   help='Preferred person identifier in .obs; if absent use --patient-rule')
    p.add_argument('--patient-rule', choices=['error', 'before_dot', 'sample'], default='before_dot')
    p.add_argument('--stratify-column', default=None,
                   help='Optional OBS diagnosis column for patient-stratified folds')
    p.add_argument('--min-cells-per-sample', type=int, default=32)
    p.add_argument('--max-mt', type=float, default=15.0)
    p.add_argument('--min-genes', type=int, default=500)
    p.add_argument('--max-genes', type=int, default=5000)
    p.add_argument('--max-counts', type=int, default=20000)
    return p.parse_args()


def verify_counts(x):
    values = x.data if sparse.issparse(x) else np.asarray(x).ravel()
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError('Raw expression has negative or nonfinite values')
    if not np.allclose(values, np.round(values), atol=1e-5, rtol=0):
        raise ValueError('.raw.X is not integer-like raw counts; do not use log-normalized data')


def select_features(x, genes, train_rows, libraries, n, method):
    can_use = ~(genes.str.startswith('MT-') | genes.str.startswith('RPS') |
                genes.str.startswith('RPL'))
    eligible = np.flatnonzero(np.asarray(can_use))
    # Original 3-cell detectability threshold, evaluated on TRAIN patients only.
    detectable = np.asarray((x[train_rows][:, eligible] > 0).sum(axis=0)).ravel() >= 3
    eligible = eligible[detectable]
    if not len(eligible):
        raise ValueError('No eligible nonmitochondrial/nonribosomal genes')
    x_train = x[train_rows][:, eligible]
    n = min(n, x_train.shape[1])
    if method == 'seurat_v3':
        import anndata as ad
        import scanpy as sc
        a = ad.AnnData(X=x_train, obs=pd.DataFrame({'library': libraries[train_rows].astype(str)}),
                       var=pd.DataFrame(index=genes[eligible].astype(str)))
        sc.pp.highly_variable_genes(a, n_top_genes=n, flavor='seurat_v3',
                                  batch_key='library', subset=False)
        selected = np.flatnonzero(a.var['highly_variable'].to_numpy())
        # Record order by deterministic original gene order, not HVG score order.
        return eligible[selected]
    # Explicit fallback for environments without scanpy[skmisc]. Train patients only.
    totals = np.asarray(x_train.sum(axis=1)).ravel().clip(min=1)
    normalized = x_train.multiply(1e4 / totals[:, None]) if sparse.issparse(x_train) else x_train * (1e4 / totals[:, None])
    lognorm = normalized.copy()
    if sparse.issparse(lognorm):
        lognorm.data = np.log1p(lognorm.data)
        variance = np.asarray(lognorm.power(2).mean(0)).ravel() - np.asarray(lognorm.mean(0)).ravel() ** 2
    else:
        variance = np.log1p(lognorm).var(axis=0)
    chosen = np.argsort(-variance, kind='stable')[:n]
    return eligible[np.sort(chosen)]


def split_patients(samples, patient_ids, group_labels, folds, seed, fold):
    table = pd.DataFrame({'sample_id': samples, 'patient_id': patient_ids, 'group': group_labels})
    # Group by person; all samples and their cross-library replicates move together.
    unique = table.groupby('patient_id', sort=True)['group'].first().reset_index()
    if len(unique) < folds:
        raise ValueError(f'Only {len(unique)} patients available for {folds} folds')
    if unique['group'].notna().all() and unique['group'].value_counts().min() >= folds and unique['group'].nunique() > 1:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        split = cv.split(unique['patient_id'], unique['group'])
        stratified = True
    else:
        cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
        split = cv.split(unique['patient_id'])
        stratified = False
    assignment = np.empty(len(unique), dtype=int)
    for f, (_, testing) in enumerate(split):
        assignment[testing] = f
    unique['fold'] = assignment
    table = table.merge(unique[['patient_id', 'fold']], on='patient_id', how='left')
    table['role'] = np.where(table.fold == fold, 'test',
                             np.where(table.fold == (fold + 1) % folds, 'val', 'train'))
    return table, stratified


def main(args):
    import anndata as ad
    args.output.mkdir(parents=True, exist_ok=True)
    if not (0 <= args.fold < args.folds) or args.folds < 3:
        raise ValueError('Require 0 <= fold < folds and folds >= 3')
    print('Reading prepared .h5ad; `.raw.X` must contain original UMIs...')
    prepared = ad.read_h5ad(args.prepared)
    if prepared.raw is None:
        raise ValueError('Prepared object has no `.raw` count matrix')
    x = prepared.raw.X
    verify_counts(x)
    x = sparse.csr_matrix(x) if not sparse.issparse(x) else x.tocsr()
    genes = pd.Index(prepared.raw.var_names.astype(str))
    if not genes.is_unique:
        raise ValueError('Gene names in .raw.var_names are not unique')
    obs = prepared.obs.copy()
    required = [args.sample_column, args.library_column]
    missing = [col for col in required if col not in obs]
    if missing:
        raise KeyError(f'Missing obs columns: {missing}')
    obs['sample_id'] = obs[args.sample_column].astype('string')
    obs['library'] = obs[args.library_column].astype('string')
    if obs[['sample_id','library']].isna().any().any():
        raise ValueError('Missing sample or library identities')
    if args.patient_column in obs:
        obs['patient_id'] = obs[args.patient_column].astype('string')
        rule = f'obs[{args.patient_column}]'
    elif args.patient_rule == 'before_dot':
        obs['patient_id'] = obs.sample_id.str.split('.', n=1, regex=False).str[0]
        rule = 'sample_id prefix before dot (explicitly verify person IDs!)'
    elif args.patient_rule == 'sample':
        obs['patient_id'] = obs.sample_id
        rule = 'sample IDs are patients (cannot protect longitudinal leakage)'
    else:
        raise ValueError('No patient_id column; provide it or --patient-rule')
    if obs['patient_id'].isna().any():
        raise ValueError('Missing patient IDs')
    obs['cell_id'] = obs.index.astype(str)
    if not obs['cell_id'].is_unique:
        raise ValueError('Cell identifiers are not unique')
    if args.stratify_column:
        if args.stratify_column not in obs:
            raise KeyError(f'Cannot stratify on missing obs[{args.stratify_column!r}]')
        obs['group'] = obs[args.stratify_column].astype('string')
    else:
        obs['group'] = pd.NA
    # QC is annotation-blind; it precedes train-fold feature selection.
    total_umi = np.asarray(x.sum(axis=1)).ravel()
    n_genes = np.diff(x.indptr)
    mt_ids = np.flatnonzero(genes.str.startswith('MT-'))
    mt_umi = np.asarray(x[:, mt_ids].sum(axis=1)).ravel() if len(mt_ids) else np.zeros(x.shape[0])
    mt_pct = 100 * mt_umi / np.maximum(total_umi, 1)
    keep = ((n_genes > args.min_genes) & (n_genes < args.max_genes) &
            (total_umi < args.max_counts) & (mt_pct < args.max_mt) & (total_umi > 0))
    qc = pd.DataFrame({'library': obs.library, 'keep': keep}).groupby('library')['keep'].agg(['size','sum'])
    qc.to_csv(args.output / 'qc_by_library.csv')
    obs = obs.iloc[np.flatnonzero(keep)].copy().reset_index(drop=True)
    x = x[keep]
    obs['full_umi'] = total_umi[keep].astype(np.float32)
    # One sample must represent one physical sampling timepoint, one patient.
    if obs.groupby('sample_id')['patient_id'].nunique().max() != 1:
        raise ValueError('Conflicting person IDs within a sample')
    # RNA eligibility is decided independently of auxiliary-view availability.
    # Samples with too few QC-passed cells may remain if any auxiliary view exists.
    min_cells = max(2, args.min_cells_per_sample)
    cell_counts = obs.groupby('sample_id').size()
    usable_rna = set(cell_counts[cell_counts >= min_cells].index.astype(str))
    row_keep = obs.sample_id.isin(usable_rna).to_numpy()
    x = x[row_keep]
    obs = obs.loc[row_keep].reset_index(drop=True)

    aux_raw = {}
    aux_dims = {}
    missing_aux_diagnostics = []
    if args.aux_store is not None:
        if not args.aux_store.is_file():
            raise FileNotFoundError(args.aux_store)
        with h5py.File(args.aux_store, 'r') as store:
            for view in args.aux_views:
                if view not in store:
                    raise KeyError(f'Legacy aux store lacks group {view!r}')
                vectors = {}
                for sid in store[view].keys():
                    sid = str(sid)
                    try:
                        vector = np.asarray(store[view][sid], dtype=np.float32).reshape(-1)
                    except (ValueError, TypeError) as exc:
                        missing_aux_diagnostics.append((sid, view, str(exc)))
                        continue
                    if not len(vector) or not np.isfinite(vector).all():
                        missing_aux_diagnostics.append((sid, view, 'empty or nonfinite'))
                        continue
                    vectors[sid] = vector
                sizes = {len(v) for v in vectors.values()}
                if len(sizes) > 1:
                    raise ValueError(f'Inconsistent dimensions for auxiliary view {view!r}: {sizes}')
                if not sizes:
                    raise ValueError(f'No usable samples for requested auxiliary view {view!r}')
                aux_raw[view] = vectors
                aux_dims[view] = next(iter(sizes))
    elif args.aux_views:
        print('WARNING: no --aux-store; running RNA-only.')

    # Read authoritative IDs where supplied. For auxiliary-only samples, never
    # silently borrow another sample's patient ID or diagnosis.
    external = None
    if args.sample_metadata_csv is not None:
        external = pd.read_csv(args.sample_metadata_csv, dtype='string')
        if 'sample_id' not in external or 'patient_id' not in external:
            raise ValueError('Metadata CSV must contain sample_id and patient_id')
        if external.sample_id.isna().any() or external.patient_id.isna().any():
            raise ValueError('Metadata CSV contains missing IDs')
        if external.sample_id.duplicated().any():
            raise ValueError('Metadata CSV contains duplicate sample IDs')
        external = external.set_index('sample_id')

    rna_meta = obs.groupby('sample_id', sort=True).agg(
        patient_id=('patient_id', 'first'), group=('group', 'first'),
        n_cells=('sample_id', 'size'))
    union = set(rna_meta.index.astype(str))
    for vectors in aux_raw.values():
        union.update(vectors)
    records = []
    for sid in sorted(union):
        has_rna = sid in rna_meta.index
        metadata_row = external.loc[sid] if external is not None and sid in external.index else None
        if has_rna:
            pid = str(rna_meta.loc[sid, 'patient_id'])
            label = rna_meta.loc[sid, 'group']
            ncells = int(rna_meta.loc[sid, 'n_cells'])
            if metadata_row is not None and str(metadata_row.patient_id) != pid:
                raise ValueError(f'Patient ID disagreement for sample {sid!r}')
            if pd.isna(label) and metadata_row is not None and 'group' in external:
                label = metadata_row['group']
        else:
            ncells = 0
            if metadata_row is not None:
                pid = str(metadata_row.patient_id)
            elif args.patient_column in prepared.obs and args.patient_rule != 'sample':
                raise ValueError(f'No verified patient ID for auxiliary-only sample {sid!r}; '
                                 'provide --sample-metadata-csv')
            elif args.patient_rule == 'before_dot':
                pid = sid.split('.')[0]
            elif args.patient_rule == 'sample':
                pid = sid
            else:
                raise ValueError(f'Cannot infer patient ID for auxiliary-only sample {sid!r}')
            label = (metadata_row['group'] if metadata_row is not None and 'group' in external
                     else pd.NA)
        record = {'sample_id':sid, 'patient_id':pid, 'group':label,
                  'n_cells':ncells, 'observed_RNA':bool(has_rna)}
        for view, vectors in aux_raw.items():
            record[f'observed_{view}'] = sid in vectors
        if any(record[f'observed_{v}'] for v in ['RNA', *aux_raw]):
            records.append(record)

    sample_info = pd.DataFrame(records)
    if sample_info.empty:
        raise ValueError('No samples have any observed view after QC')
    # Different samples from the same patient must not carry conflicting labels.
    groups = sample_info.dropna(subset=['group']).groupby('patient_id')['group'].nunique()
    if (groups > 1).any():
        raise ValueError('Conflicting histology/clinical groups within a patient')
    partitions, stratified = split_patients(
        sample_info.sample_id.to_numpy(), sample_info.patient_id.to_numpy(),
        sample_info.group.to_numpy(), args.folds, args.seed, args.fold)
    if set(partitions.role) != {'train', 'val', 'test'}:
        raise ValueError('Train/val/test partition empty')
    # split_patients returns the split columns, so rejoin the union-view masks.
    partitions = partitions.merge(
        sample_info[['sample_id', 'n_cells', *[f'observed_{v}' for v in ['RNA', *aux_raw]]]],
        on='sample_id', how='left', validate='one_to_one')
    partitions.to_csv(args.output / 'sample_manifest.csv', index=False)
    mask_names = ['observed_RNA', *[f'observed_{v}' for v in aux_raw]]
    partitions[['sample_id', *mask_names]].to_csv(args.output / 'view_masks.csv', index=False)
    pd.DataFrame(missing_aux_diagnostics, columns=['sample_id','view','reason']).to_csv(
        args.output / 'invalid_auxiliary_values.csv', index=False)
    obs = obs.merge(partitions[['sample_id','role','fold']], on='sample_id', how='left',
                    validate='many_to_one')
    obs.to_csv(args.output / 'cells.csv', index=False)
    composition = pd.crosstab(obs.sample_id, obs.library).reindex(
        partitions.sample_id, fill_value=0)
    composition.to_csv(args.output / 'sample_library_composition.csv')

    train_rows = np.flatnonzero((obs.role == 'train').to_numpy())
    if len(train_rows) == 0:
        raise ValueError('Training fold has no observed RNA; cannot select genes')
    print(f'{len(partitions)} union samples; {len(train_rows):,} RNA training cells; '
          f'selecting {args.hvg} genes from train only...')
    selected = select_features(x, genes, train_rows, obs.library.to_numpy(),
                               args.hvg, args.hvg_method)
    subset = x[:, selected].astype(np.float32).tocsr()
    sparse.save_npz(args.output / 'counts_selected.npz', subset, compressed=True)
    pd.DataFrame({'gene': genes[selected]}).to_csv(args.output / 'selected_genes.csv', index=False)
    library_names = sorted(obs.library.unique().astype(str).tolist())
    sample_ids = partitions.sample_id.astype(str).to_list()
    aux_out, aux_stats = {}, {}
    for view, vectors in aux_raw.items():
        available = partitions[f'observed_{view}'].astype(bool).to_numpy()
        train_observed = available & (partitions.role == 'train').to_numpy()
        if not np.any(train_observed):
            raise ValueError(f'View {view!r} has no observed training samples in fold '
                             f'{args.fold}; cannot fit train-only scaling')
        raw = np.zeros((len(sample_ids), aux_dims[view]), dtype=np.float32)
        for i, sid in enumerate(sample_ids):
            if available[i]:
                raw[i] = vectors[sid]
        mean = raw[train_observed].mean(axis=0)
        sd = raw[train_observed].std(axis=0)
        sd = np.where(sd > 1e-5, sd, 1.)
        # Zeros in missing rows are numerical placeholders, *not* observations.
        # The corresponding observed_<view> mask excludes them throughout training.
        standardized = np.zeros_like(raw)
        standardized[available] = np.clip((raw[available] - mean) / sd, -10, 10)
        aux_out[view] = standardized
        aux_stats[view] = {'mean':mean.tolist(), 'sd':sd.tolist(),
                           'train_observed_samples':int(train_observed.sum())}
    np.savez_compressed(args.output / 'auxiliary.npz', **aux_out)
    (args.output / 'aux_train_scalers.json').write_text(json.dumps(aux_stats, indent=2))
    ref_depth = float(np.median(obs.loc[obs.role == 'train', 'full_umi']))
    if not np.isfinite(ref_depth) or ref_depth <= 0:
        raise ValueError('Invalid training-only reference depth')
    mask_coverage = {}
    for role, group in partitions.groupby('role'):
        mask_coverage[str(role)] = {k:int(group[k].astype(bool).sum()) for k in mask_names}
    pd.DataFrame([{'role':role,'view':view,'observed':count}
                  for role, vals in mask_coverage.items()
                  for view,count in vals.items()]).to_csv(
        args.output / 'view_coverage.csv', index=False)
    config = {'input':str(args.prepared), 'count_source':'.raw.X',
              'aux_store':str(args.aux_store) if args.aux_store else None,
              'aux_views':list(aux_out), 'patient_id_rule':rule,
              'fold':args.fold, 'folds':args.folds, 'seed':args.seed,
              'hvg_method':args.hvg_method, 'n_genes':len(selected),
              'n_libraries':len(library_names), 'library_names':library_names,
              'ref_depth_train_median_umi':ref_depth,
              'cohort_policy':'eligible union of all observed views',
              'view_mask_columns':mask_names,
              'view_coverage':mask_coverage,
              'stratified':stratified,
              'caveat':'Clinical biomarkers may overlap ACS/CCS outcome.'}
    (args.output / 'preparation.json').write_text(json.dumps(config, indent=2))
    print('Prepared all_cv union:', len(partitions), 'samples;',
          sum(~partitions.observed_RNA.astype(bool)), 'without RNA')


if __name__ == '__main__':
    main(arguments())
