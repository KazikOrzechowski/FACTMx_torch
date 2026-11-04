"""Train ACS/CCS FACTMx with one of three structured raw-count RNA heads.

RNA head options: 'nb', 'zinb' (gene-specific after NB warm start), 'topic'.
Four optional simple heads: ck, neutro, proteo, clinical.
All models use the exact same fold preparation and cell drawing sequence.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.nn.utils import clip_grad_norm_

from factmx_torch import MissingViewFACTMx


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-fold', type=Path, required=True,
                   help='Output directory of prepare_acs_ccs.py')
    p.add_argument('--model', choices=['nb', 'zinb', 'topic', 'all'], required=True)
    p.add_argument('--output', type=Path, default=Path('results/acs_ccs/models'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto')
    p.add_argument('--latent-dim', type=int, default=20)
    p.add_argument('--components', type=int, default=15)
    p.add_argument('--cells-per-sample', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=8,
                   help='Samples (not cells) per gradient update')
    p.add_argument('--epochs-nb', type=int, default=40)
    p.add_argument('--epochs-zinb', type=int, default=20)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--zinb-lr', type=float, default=2e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--kl-beta', type=float, default=0.02)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--view-dropout', type=float, default=0.15,
                   help='Artificially hide observed inputs during training only; retain their loss targets')
    p.add_argument('--center-samples', type=int, default=256,
                   help='Maximum training samples for initial observed-only neutral centers')
    p.add_argument('--checkpoint-metric', choices=['rna', 'scaled_total'], default='scaled_total',
                   help='scaled_total = held-out view-wise predictive NLL; RNA option requires val RNA')
    p.add_argument('--eval-cell-repeats', type=int, default=1,
                   help='Independent fixed context/query draws per held-out sample')
    return p.parse_args()


class FoldData:
    """Sparse cells, sample-level views and one manifest, prepared train-only."""
    def __init__(self, folder):
        self.folder = Path(folder)
        self.counts = sparse.load_npz(self.folder / 'counts_selected.npz').tocsr()
        self.cells = pd.read_csv(self.folder / 'cells.csv', dtype={'sample_id': str, 'patient_id': str,
                                                                   'library': str, 'cell_id': str})
        self.samples = pd.read_csv(self.folder / 'sample_manifest.csv', dtype={'sample_id': str, 'patient_id': str})
        self.config = json.loads((self.folder / 'preparation.json').read_text())
        self.genes = pd.read_csv(self.folder / 'selected_genes.csv')['gene'].astype(str).tolist()
        self.aux_data = dict(np.load(self.folder / 'auxiliary.npz'))
        self.aux_views = self.config['aux_views']
        if list(self.aux_data) != list(self.aux_views):
            raise ValueError('Auxiliary view names/order inconsistent')
        if len(self.cells) != self.counts.shape[0] or self.counts.shape[1] != len(self.genes):
            raise ValueError('Cell/gene order mismatch with sparse count cache')
        if self.cells['cell_id'].duplicated().any():
            raise ValueError('Duplicate cell IDs')
        if self.samples['sample_id'].duplicated().any():
            raise ValueError('Duplicate sample IDs')
        self.view_names = ['RNA', *self.aux_views]
        columns = [f'observed_{name}' for name in self.view_names]
        if any(name not in self.samples for name in columns):
            raise ValueError('Expected all_cv sample_manifest.csv with observed_<view> columns')
        self.observed = self.samples[columns].astype(bool).to_numpy(dtype=bool)
        if not np.all(self.observed.any(axis=1)):
            raise ValueError('At least one sample has no observed view')
        self.sample_ids = self.samples.sample_id.tolist()
        self.row_groups = self.cells.groupby('sample_id', sort=False).indices
        if set(self.row_groups) != set(self.samples.loc[self.observed[:, 0], 'sample_id']):
            raise ValueError('RNA-observed sample IDs differ from cells.csv')
        for name, matrix in self.aux_data.items():
            if len(matrix) != len(self.samples):
                raise ValueError(f'Auxiliary {name} length does not match manifest')
        self.index_by_role = {
            role: np.flatnonzero((self.samples.role == role).to_numpy()).astype(int).tolist()
            for role in ['train', 'val', 'test']
        }
        if not all(len(v) for v in self.index_by_role.values()):
            raise ValueError('At least one train/val/test role is empty')
        libs = self.config['library_names']
        library_ids = self.cells.library.map({b: i for i, b in enumerate(libs)})
        if library_ids.isna().any():
            raise ValueError('Library absent from saved library names')
        self.library_ids = library_ids.to_numpy(dtype=np.int64)
        self.full_umi = self.cells.full_umi.to_numpy(dtype=np.float32)
        self.ref_depth = self.config['ref_depth_train_median_umi']

    def _choose_rows(self, sample_idx, count, rng, pool='all'):
        sid = self.sample_ids[sample_idx]
        possible = np.asarray(self.row_groups[sid], dtype=int)
        if pool != 'all':
            # Deterministic, disjoint context/query pools for predictive scoring.
            # Stable within a run and *does not* depend on the model being evaluated.
            # Split shuffled indices deterministically per sample, with separate local RNG.
            import hashlib
            digest = hashlib.sha256(sid.encode('utf8')).digest()
            local_seed = (int.from_bytes(digest[:8], 'little') + 314159) % (2**32)
            shuffled = np.random.default_rng(local_seed).permutation(possible)
            half = max(1, len(shuffled) // 2)
            possible = shuffled[:half] if pool == 'context' else shuffled[half:]
            if len(possible) == 0:
                raise ValueError(f'Sample {sid} has fewer than 2 cells')
        return rng.choice(possible, size=count, replace=len(possible) < count)

    def make_batch(self, indices, *, cells_per_sample, seed, role, pool='all', device='cpu'):
        rng = np.random.default_rng(seed)
        b = len(indices)
        c = cells_per_sample
        # Placeholders are computationally valid but NEVER treated as observed.
        x = np.zeros((b, c, len(self.genes)), dtype=np.float32)
        sf = np.ones((b, c), dtype=np.float32)
        lib = np.zeros((b, c), dtype=np.int64)
        for i, index in enumerate(indices):
            if not self.observed[index, 0]:
                continue
            rows = self._choose_rows(index, c, rng, pool)
            x[i] = self.counts[rows].toarray()
            sf[i] = self.full_umi[rows] / self.ref_depth
            lib[i] = self.library_ids[rows]
        counts = torch.as_tensor(x, device=device)
        factors = torch.as_tensor(sf, dtype=torch.float32, device=device)
        lib_ids = torch.as_tensor(lib, dtype=torch.long, device=device)
        aux = [torch.as_tensor(self.aux_data[name][indices], device=device, dtype=torch.float32)
               for name in self.aux_views]
        observed = torch.as_tensor(self.observed[indices], device=device, dtype=torch.bool)
        return counts, factors, lib_ids, aux, observed


def make_model(kind, data, latent_dim, components, beta):
    head = {
        'head_type': 'TopicModel' if kind == 'topic' else 'ZINBMixture',
        'head_name': 'RNA', 'dim': components,
        'objective': 'marginal', 'assignment_estimator': 'mean',
        'contextual_gate': False,
    }
    if kind == 'topic':
        head.update({'dim_words': len(data.genes), 'normalize_profiles': True})
    else:
        head.update({'dim_genes': len(data.genes), 'n_libraries': data.config['n_libraries'],
                     'inflation': 'gene' if kind == 'zinb' else 'none',
                     'initial_inflation': 'none',
                     'normalize_by_genes': False,
                     'batch_effect_sd': 0.25, 'batch_effect_penalty': 1.0})
    heads = [head] + [
        {'head_type': 'MultiNormal', 'head_name': name, 'dim': data.aux_data[name].shape[1]}
        for name in data.aux_views
    ]
    model = MissingViewFACTMx(
        dim_latent=latent_dim, heads_config=heads,
        encoder_config={'encoder_type': 'Linear', 'scale_parameterization': 'softplus'},
        beta=beta, neutral_centering=True, availability_to_scale=False,
    )
    # Auxiliary views are standardized on train; scale Normal emissions accordingly.
    with torch.no_grad():
        for item in model.heads[1:]:
            item.log_scale.zero_()
    return model


def build_view(kind, batch):
    counts, factors, lib, auxiliary, observed = batch
    return [counts if kind == 'topic' else (counts, factors, lib), *auxiliary]


def initial_rna_components(model, kind, data, seed, cells_per_sample, device):
    # Training-only, cap initialization to a representative sample of patients.
    train = [i for i in data.index_by_role['train'] if data.observed[i, 0]]
    if not train:
        raise ValueError('No observed training RNA samples for component initialization')
    rng = np.random.default_rng(seed + 911)
    subset = rng.choice(train, size=min(len(train), 32), replace=False).tolist()
    batch = data.make_batch(subset, cells_per_sample=cells_per_sample,
                            seed=seed + 991, role='train', device=device)
    head = model.heads[0]
    if kind != 'topic':
        summary = head.initialize_from_counts((batch[0], batch[1], batch[2]), seed=seed,
                                              max_init_cells=min(5000, batch[0].shape[0]*batch[0].shape[1]))
        return summary
    # Topic initialization on normalized counts is training-only; unlike the
    # original Topic head, we explicitly normalize emission log profiles.
    from sklearn.cluster import MiniBatchKMeans
    x = batch[0].reshape(-1, batch[0].shape[-1]).cpu().numpy()
    sf = batch[1].reshape(-1, 1).cpu().numpy()
    features = np.log1p(x / np.clip(sf, 1e-5, None))
    km = MiniBatchKMeans(n_clusters=head.dim, random_state=seed, n_init=3,
                         batch_size=min(256, len(features)))
    labels = km.fit_predict(features)
    rates = np.stack([x[labels == k].sum(axis=0) + 1.0 for k in range(head.dim)])
    proportions = rates / rates.sum(axis=1, keepdims=True)
    freq = (np.bincount(labels, minlength=head.dim) + 1.) / (len(labels) + head.dim)
    with torch.no_grad():
        head.topic_profiles_trainable.copy_(torch.tensor(np.log(proportions.T),
                                                        dtype=torch.float32, device=device))
        head.mixture_logits.weight.zero_()
        head.mixture_logits.bias.copy_(torch.tensor(np.log(freq), dtype=torch.float32, device=device))
        # Raw-UMI encoder inputs can be large; initialize a small discriminant
        # from normalized centroid profiles rather than random large logits.
        centers = km.cluster_centers_.astype(np.float32)
        centered = centers - centers.mean(axis=0)
        norms = np.linalg.norm(centered, axis=1, keepdims=True).clip(min=1e-6)
        head.encoder_classifier.weight.copy_(torch.tensor(centered / norms * 0.001,
                                                          dtype=torch.float32, device=device))
        head.encoder_classifier.bias.copy_(torch.tensor(np.log(freq), dtype=torch.float32, device=device))
    return {'method': 'train-only MiniBatchKMeans', 'cells_used': len(x),
            'component_probs': freq.tolist()}


def _subsample_head_data(value, mask):
    """Index a Tensor or the (counts, depth, library) head input by sample."""
    if isinstance(value, torch.Tensor):
        return value[mask]
    return tuple(item[mask] for item in value)


def artificial_encoder_mask(observed, rate):
    """Artificially drop observed inputs, never hide *all* inputs of a sample."""
    if rate <= 0:
        return observed
    keep = observed & (torch.rand(observed.shape, device=observed.device) >= rate)
    empty = (~keep.any(dim=1)).nonzero(as_tuple=True)[0]
    for idx in empty.tolist():
        options = torch.where(observed[idx])[0]
        keep[idx, options[torch.randint(len(options), (1,), device=observed.device)]] = True
    return keep


def masked_components(model, kind, batch, *, dropout=0., update_centers=False,
                      deterministic=False):
    """Averages each observed view's likelihood over only its observed patients.

    Natural missing views never contribute loss. Artificially masked observed
    views ARE reconstruction/prediction targets, as in all_cv.
    """
    mask = batch[4]
    inputs = build_view(kind, batch)
    encoder_mask = artificial_encoder_mask(mask, dropout)
    latent, loc, scale, encoded = model.infer_masked(
        inputs, encoder_mask=encoder_mask, deterministic=deterministic,
        update_centers=update_centers, center_update_mask=mask)
    losses = {'latent_kl':model._latent_kl(loc, scale) * model.beta}
    for j, head in enumerate(model.heads):
        present = mask[:, j]
        if bool(present.any()):
            head_input = _subsample_head_data(inputs[j], present)
            local_encoded = {key:val[present] for key,val in encoded[j].items()}
            losses[head.head_name] = head.loss(head_input, latent[present], local_encoded)
        else:
            losses[head.head_name] = latent.sum() * 0.
    return losses


def masked_loss(model, kind, batch, *, dropout=0., update_centers=False,
                deterministic=False):
    losses = masked_components(model, kind, batch, dropout=dropout,
                               update_centers=update_centers,
                               deterministic=deterministic)
    return (torch.stack(list(losses.values())) * model.loss_scales).mean()


@torch.no_grad()
def initialize_neutral_centers(model, kind, data, *, seed, cells_per_sample, device,
                               maximum=256, batch_size=16):
    """Compute encoder reference means from *observed train views* only."""
    model.eval()
    train = data.index_by_role['train']
    rng = np.random.default_rng(seed + 3301)
    # Ensure that even rare observed views contribute to neutral centers.
    required = {int(rng.choice([i for i in train if data.observed[i, j]]))
                for j in range(len(model.heads))}
    random_ids = rng.choice(train, size=min(len(train), max(maximum, len(required))),
                            replace=False).tolist()
    sample_ids = sorted(required | set(map(int, random_ids)))
    sums = [torch.zeros(dim, device=device) for dim in model.head_dims]
    counts = [0] * len(model.heads)
    for start in range(0, len(sample_ids), batch_size):
        ids = sample_ids[start:start+batch_size]
        batch = data.make_batch(ids, cells_per_sample=cells_per_sample,
                                seed=seed + start + 731, role='train', device=device)
        mask = batch[4]
        encodings = [head.encode(inp, deterministic=True)
                     for head, inp in zip(model.heads, build_view(kind, batch))]
        for j, encoded in enumerate(encodings):
            observed = mask[:, j]
            if bool(observed.any()):
                sums[j] += encoded['encoder_input'][observed].sum(dim=0)
                counts[j] += int(observed.sum())
    if any(n == 0 for n in counts):
        raise ValueError(f'Cannot estimate training-only neutral centers; counts={counts}')
    centers = torch.cat([total / n for total, n in zip(sums, counts)])
    model.set_encoder_centers(centers)
    return dict(zip(data.view_names, counts))


@torch.no_grad()
def set_training_only_loss_weights(model, kind, batch):
    model.eval()
    values = masked_components(model, kind, batch, deterministic=True)
    weights = [1.0]
    for j, head in enumerate(model.heads):
        if not bool(batch[4][:, j].any()):
            raise ValueError(f'Calibration minibatch lacks observed {head.head_name}')
        raw = float(values[head.head_name].detach())
        if not np.isfinite(raw):
            raise ValueError(f'Nonfinite initial {head.head_name} loss')
        weights.append(1.0 / max(abs(raw), 1.0))
    model.loss_scales.copy_(torch.tensor(weights, device=model.loss_scales.device))
    return dict(zip(['latent_kl'] + [h.head_name for h in model.heads], weights))


@torch.no_grad()
def score_split(model, kind, data, role, seed, cells_per_sample, batch_size,
                device, repeats=1):
    """Score ONLY observed targets, using the same held-out patient folds.

    RNA: disjoint query cells given context RNA+observed auxiliary views.
    Auxiliary: leave-one-view-out prediction (target view excluded from encoder).
    No holdout target contributes its own encoder evidence.
    """
    model.eval()
    totals = {name:0. for name in data.view_names}
    counts = {name:0 for name in data.view_names}
    indices = data.index_by_role[role]
    for repeat in range(repeats):
        for start in range(0, len(indices), batch_size):
            ids = indices[start:start+batch_size]
            context = data.make_batch(ids, cells_per_sample=cells_per_sample,
                                      seed=seed + repeat*2039 + start, role=role,
                                      pool='context', device=device)
            query = data.make_batch(ids, cells_per_sample=cells_per_sample,
                                    seed=seed + repeat*2039 + start + 900001, role=role,
                                    pool='query', device=device)
            mask = context[4]
            inputs = build_view(kind, context)
            for j, head in enumerate(model.heads):
                present = mask[:, j]
                if not bool(present.any()):
                    continue
                if j == 0:
                    # RNA has separate context and query cells; it may participate
                    # in encoder inference without contaminating query targets.
                    effective_mask = mask
                else:
                    effective_mask = mask.clone()
                    effective_mask[:, j] = False
                _, latent, _, _ = model.infer_masked(
                    inputs, effective_mask, deterministic=True, update_centers=False)
                if j == 0:
                    score = head.predictive_loss(
                        _subsample_head_data(build_view(kind, query)[0], present),
                        latent[present], _subsample_head_data(inputs[0], present))
                else:
                    score = head.loss(inputs[j][present], latent[present], {})
                if not torch.isfinite(score):
                    raise FloatingPointError(f'Invalid predictive {head.head_name} loss')
                n = int(present.sum())
                totals[head.head_name] += n * float(score)
                counts[head.head_name] += n
    result = {name:(totals[name] / counts[name] if counts[name] else None)
              for name in data.view_names}
    result.update({f'n_{name}':int(counts[name]) for name in data.view_names})
    return result


def balanced_score(model, scores):
    valid = [(scores[h.head_name], float(model.loss_scales[j+1]))
             for j,h in enumerate(model.heads) if scores[h.head_name] is not None]
    if not valid:
        raise ValueError('Validation role contains no observed targets')
    return sum(v * weight for v,weight in valid) / sum(weight for _,weight in valid)

def train_stage(model, kind, data, cfg, device, stage, epochs, lr, seed, out):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    best = math.inf
    bad = 0
    history = []
    best_path = out / f'best_{stage}.pt'
    for epoch in range(epochs):
        model.train()
        # Stable per-model random state: all three models see the same patient and
        # cell sampling sequence for a given seed, stage and epoch.
        rng = np.random.default_rng(seed + epoch * 1723)
        order = rng.permutation(data.index_by_role['train']).tolist()
        total = 0.
        batches = 0
        for start in range(0,len(order),cfg.batch_size):
            ids = order[start:start+cfg.batch_size]
            batch = data.make_batch(ids, cells_per_sample=cfg.cells_per_sample,
                                    seed=seed + epoch * 20003 + start, role='train', device=device)
            inputs = build_view(kind, batch)
            opt.zero_grad(set_to_none=True)
            loss = masked_loss(model, kind, batch, dropout=cfg.view_dropout,
                               update_centers=True)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite {kind} {stage} training loss')
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            total += float(loss.detach())
            batches += 1
        scores = score_split(model, kind, data, 'val', cfg.seed + 1818,
                             cfg.cells_per_sample, cfg.batch_size, device,
                             repeats=cfg.eval_cell_repeats)
        if cfg.checkpoint_metric == 'rna':
            if scores['RNA'] is None:
                raise ValueError('No observed validation RNA for RNA-only checkpoint metric')
            criterion = scores['RNA']
        else:
            criterion = balanced_score(model, scores)
        row = {'stage': stage, 'epoch': epoch+1, 'train_loss': total/max(batches,1),
               'val_checkpoint_metric': criterion, 'checkpoint_metric': cfg.checkpoint_metric,
               'val_scaled_total': balanced_score(model, scores),
               **{f'val_{k}': v for k,v in scores.items()}}
        history.append(row)
        print(f'{kind} {stage} epoch {epoch+1:03d}: '
              f'train={row["train_loss"]:.4f} val={criterion:.4f}', flush=True)
        if criterion < best - 1e-5:
            best = criterion
            bad = 0
            torch.save({'state_dict': copy.deepcopy(model.state_dict()),
                        'model': kind, 'stage': stage, 'epoch': epoch+1}, best_path)
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    return history, best_path


@torch.no_grad()
def export_predictions(model, kind, data, device, cfg, out):
    model.eval()
    records = []
    gates = []
    for role in ['train', 'val', 'test']:
        for idx in data.index_by_role[role]:
            # Sample cells *deterministically*, independent of the training loss.
            # No held-out query counts are used for the latent inference.
            batch = data.make_batch([idx], cells_per_sample=cfg.cells_per_sample,
                                    seed=cfg.seed + 4321 + idx,
                                    role=role, pool='context', device=device)
            _, latent, _, _ = model.infer_masked(build_view(kind, batch),
                                                 batch[4], deterministic=True)
            g = model.heads[0].prior_gate_probs(latent)[0].cpu().numpy()
            row = data.samples.iloc[idx]
            records.append({'sample_id': str(row.sample_id), 'patient_id': str(row.patient_id),
                            'role': role, 'fold': int(row.fold),
                            **{f'observed_{name}':bool(data.observed[idx,j])
                               for j,name in enumerate(data.view_names)},
                            **{f'z_{j:02d}': float(x) for j,x in enumerate(latent[0].cpu().numpy())}})
            gates.append({'sample_id': str(row.sample_id), 'role': role,
                          **{f'component_{j:02d}': float(v) for j,v in enumerate(g)}})
    latent_table = pd.DataFrame(records)
    latent_table.to_csv(out / 'sample_latent.csv', index=False)
    for role, filename in [('train', 'train_latent.csv'), ('val', 'val_latent.csv'),
                           ('test', 'patient_latent.csv')]:
        latent_table.loc[latent_table.role == role].to_csv(out / filename, index=False)
    pd.DataFrame(gates).to_csv(out / 'sample_mixture_gates.csv', index=False)



@torch.no_grad()
def export_rna_parameters(model, kind, data, out):
    """Export biological mixture parameters and identifiable technical offsets."""
    head = model.heads[0]
    genes = data.genes
    if kind == 'topic':
        probs = head.topic_profiles().detach().cpu().numpy().T
        pd.DataFrame(probs, columns=genes).assign(component=np.arange(head.dim)).to_csv(
            out / 'topic_component_gene_probabilities.csv', index=False)
        return
    rates = head.log_component_mean.exp().detach().cpu().numpy()
    pd.DataFrame(rates, columns=genes).assign(component=np.arange(head.dim)).to_csv(
        out / 'nb_component_gene_rates.csv', index=False)
    dispersion = head.dispersion().detach().cpu().numpy()
    per_gene = pd.DataFrame({'gene':genes, 'nb_inverse_dispersion':dispersion})
    logits = head.inflation_logits()
    if logits is not None:
        per_gene['zinb_gene_inflation_pi'] = torch.sigmoid(logits).mean(0).detach().cpu().numpy()
    per_gene.to_csv(out / 'nb_gene_dispersion_and_inflation.csv', index=False)
    effects = head.batch_effects().detach().cpu().numpy()
    pd.DataFrame(effects, columns=genes).assign(library=data.config['library_names']).to_csv(
        out / 'gene_library_log_offsets.csv', index=False)


def run_kind(kind, data, cfg, device):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    folder = cfg.output / f'fold_{data.config["fold"]:02d}' / kind
    folder.mkdir(parents=True, exist_ok=True)
    model = make_model(kind, data, cfg.latent_dim, cfg.components, cfg.kl_beta).to(device)
    init = initial_rna_components(model, kind, data, cfg.seed, cfg.cells_per_sample, device)
    # Unlike complete-case covariance initialization, neutral encoder references
    # are computed independently for every view using observed training samples.
    # They are further updated by train-only EMA (never val/test, never dropout).
    center_counts = initialize_neutral_centers(
        model, kind, data, seed=cfg.seed, cells_per_sample=cfg.cells_per_sample,
        device=device, maximum=cfg.center_samples)
    train_idx = data.index_by_role['train']
    rng = np.random.default_rng(cfg.seed + 6111)
    seed_indices = set()
    for j in range(len(model.heads)):
        choices = [i for i in train_idx if data.observed[i,j]]
        seed_indices.add(int(rng.choice(choices)))
    random_sample = rng.choice(train_idx, size=min(32,len(train_idx)), replace=False)
    subset = list(sorted(seed_indices | set(map(int,random_sample))))
    init_batch = data.make_batch(subset, cells_per_sample=cfg.cells_per_sample,
                                 seed=cfg.seed+6111, role='train', device=device)
    weights = set_training_only_loss_weights(model, kind, init_batch)
    (folder/'init_and_loss_scales.json').write_text(json.dumps({
        'initialization':init, 'neutral_center_training_counts':center_counts,
        'loss_scales':weights}, indent=2))
    epochs = cfg.epochs_nb
    history, best = train_stage(model, kind, data, cfg, device,
                                'nb_warm' if kind=='zinb' else 'main',
                                epochs, cfg.lr, cfg.seed, folder)
    checkpoint = torch.load(best, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    selected = best
    if kind == 'zinb' and cfg.epochs_zinb:
        model.heads[0].set_inflation_mode('gene')
        history2, best2 = train_stage(model, kind, data, cfg, device,
                                      'gene_zinb', cfg.epochs_zinb, cfg.zinb_lr,
                                      cfg.seed+100000, folder)
        history.extend(history2)
        checkpoint = torch.load(best2, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        selected = best2
    pd.DataFrame(history).to_csv(folder / 'training_history.csv', index=False)
    (folder / 'config.json').write_text(json.dumps({
        'model': kind, 'fold': data.config['fold'], 'seed': cfg.seed,
        'components': cfg.components, 'latent_dim': cfg.latent_dim,
        'cells_per_sample': cfg.cells_per_sample, 'batch_size': cfg.batch_size,
        'checkpoint_metric': cfg.checkpoint_metric,
        'genes': len(data.genes), 'aux_views': data.aux_views,
        'loss_scales': weights, 'selected_checkpoint': str(selected),
        'nb_epochs': cfg.epochs_nb, 'zinb_epochs': cfg.epochs_zinb if kind=='zinb' else 0,
        'patient_rule': data.config['patient_id_rule'],
        'view_dropout':cfg.view_dropout,
        'cohort_policy':'all_cv union with natural missing-view masks',
    }, indent=2))
    # Note: test is evaluated *once* only after validation-based checkpoint selection.
    val = score_split(model, kind, data, 'val', cfg.seed+1818,
                      cfg.cells_per_sample, cfg.batch_size, device, repeats=cfg.eval_cell_repeats)
    test = score_split(model, kind, data, 'test', cfg.seed+2727,
                       cfg.cells_per_sample, cfg.batch_size, device, repeats=cfg.eval_cell_repeats)
    (folder / 'heldout_scores.json').write_text(json.dumps({
        'validation': val, 'test': test,
        'note': 'Scores cover observed targets only. RNA query-cell NLL uses disjoint '
                'context/query cells; auxiliary view scores are leave-one-view-out. '
                'Topic and NB raw likelihood scales are not comparable.',
    }, indent=2))
    export_predictions(model, kind, data, device, cfg, folder)
    export_rna_parameters(model, kind, data, folder)
    torch.save({'state_dict': model.state_dict(), 'heads_config': model.head_configs,
                'encoder_config': model.encoder_config, 'dim_latent': model.dim_latent,
                'beta': model.beta, 'loss_scales': model.loss_scales.detach().cpu(),
                'model': kind}, folder/'model_final.pt')
    print('Saved', folder)
    return {'model': kind, 'test_RNA': test['RNA'], 'test_view_scores': test}


def main(cfg):
    if not 0 <= cfg.view_dropout < 1:
        raise ValueError('--view-dropout must be between 0 and 1')
    if cfg.cells_per_sample < 1 or cfg.components < 2:
        raise ValueError('Need at least 1 cell per sample, 2 components')
    device = 'cuda' if cfg.device == 'auto' and torch.cuda.is_available() else (
        'cpu' if cfg.device == 'auto' else cfg.device)
    data = FoldData(cfg.prepared_fold)
    train_libraries = set(data.cells.loc[data.cells.role == 'train', 'library'])
    unavailable = set(data.config['library_names']) - train_libraries
    if unavailable:
        print('WARNING: libraries absent from training:', sorted(unavailable),
              '; their batch corrections cannot be estimated; inspect results.')
    cfg.output.mkdir(parents=True,exist_ok=True)
    models = ['nb','zinb','topic'] if cfg.model=='all' else [cfg.model]
    summary = []
    for kind in models:
        summary.append(run_kind(kind, data, cfg, device))
    path = cfg.output / f'fold_{data.config["fold"]:02d}' / 'score_summary.json'
    path.write_text(json.dumps(summary, indent=2))
    print('Note: conditional Topic and NB/ZINB likelihoods are different probability families;')
    print('do NOT rank models by raw test NLL across these three heads.')


if __name__ == '__main__':
    main(parse_args())
