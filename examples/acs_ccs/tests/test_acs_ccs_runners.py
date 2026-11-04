"""Union-cohort and missing-view regression tests (no real patient data)."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from acs_ccs_runner import (FoldData, make_model, build_view,
                            initial_rna_components, masked_components,
                            masked_loss, score_split, run_kind, artificial_encoder_mask,
                            initialize_neutral_centers)
from prepare_acs_ccs import split_patients, verify_counts


def test_count_verification():
    verify_counts(sparse.csr_matrix([[0.,2.],[1.,3.]]))
    with pytest.raises(ValueError, match='integer'):
        verify_counts(sparse.csr_matrix([[0.,1.23]]))
    with pytest.raises(ValueError, match='negative'):
        verify_counts(sparse.csr_matrix([[0.,-1.]]))


def test_splits_keep_patient_timepoints_together():
    samples = [f'{i}.{j}' for i in range(30) for j in (1,2)]
    patients = [str(i) for i in range(30) for _ in (1,2)]
    labels = ['ACS' if i % 2 else 'CCS' for i in range(30) for _ in (1,2)]
    df, stratified = split_patients(samples,patients,labels,5,42,0)
    assert stratified
    assert df.groupby('patient_id').role.nunique().max() == 1
    assert set(df.role) == {'train','val','test'}


def make_fold(tmp_path):
    rng = np.random.default_rng(33)
    n, c, g = 14, 10, 6
    # Samples 10 and 11 are auxiliary-only; 12 and 13 are RNA+aux.
    rna_ids = [i for i in range(n) if i not in (10,11)]
    dense = rng.poisson(1.0, (len(rna_ids)*c,g)).astype(np.float32)
    sparse.save_npz(tmp_path/'counts_selected.npz', sparse.csr_matrix(dense))
    sample_ids = [str(i)+'.1' for i in range(n)]
    roles = ['test']*2 + ['val']*2 + ['train']*(n-4)
    cells = pd.DataFrame([{'cell_id':f'cell_{i}_{j}', 'sample_id':sample_ids[i],
                           'patient_id':str(i), 'library':'L1' if j%2 else 'L2',
                           'full_umi':50., 'fold':0, 'role':roles[i]}
                          for i in rna_ids for j in range(c)])
    cells.to_csv(tmp_path/'cells.csv', index=False)
    # Sample 0 has missing CK, sample 3 missing proteomics; any other sample observed.
    mask = np.array([[i not in (10,11), i != 0, i != 3] for i in range(n)])
    manifest = pd.DataFrame([{'sample_id':sample_ids[i], 'patient_id':str(i),
                              'role':roles[i], 'fold':0, 'n_cells':c if mask[i,0] else 0,
                              'observed_RNA':bool(mask[i,0]),
                              'observed_ck':bool(mask[i,1]),
                              'observed_proteo':bool(mask[i,2])} for i in range(n)])
    manifest.to_csv(tmp_path/'sample_manifest.csv', index=False)
    pd.DataFrame({'gene':[f'G{j}' for j in range(g)]}).to_csv(tmp_path/'selected_genes.csv',index=False)
    aux_ck = rng.normal(0,1,(n,2)).astype('float32')
    aux_proteo = rng.normal(0,1,(n,3)).astype('float32')
    aux_ck[~mask[:,1]] = 0
    aux_proteo[~mask[:,2]] = 0
    np.savez_compressed(tmp_path/'auxiliary.npz',ck=aux_ck,proteo=aux_proteo)
    (tmp_path/'preparation.json').write_text(json.dumps({
        'aux_views':['ck','proteo'], 'library_names':['L1','L2'], 'n_libraries':2,
        'ref_depth_train_median_umi':50,'fold':0,'patient_id_rule':'test'}))
    return FoldData(tmp_path)


def test_union_and_disjoint_context_query(tmp_path):
    d = make_fold(tmp_path)
    assert len(d.samples) == 14
    assert d.observed[10].tolist() == [False, True, True]
    assert set(d.row_groups) == set(d.samples.loc[d.observed[:,0],'sample_id'])
    rng = np.random.default_rng(19)
    a = d._choose_rows(0,8,rng,'context')
    b = d._choose_rows(0,8,rng,'query')
    assert not set(a).intersection(b)
    batch = d.make_batch([0,10],cells_per_sample=4,seed=4,role='test')
    assert batch[4].tolist() == [[True,False,True],[False,True,True]]
    assert torch.equal(batch[0][1],torch.zeros_like(batch[0][1]))
    assert torch.all(batch[1][1] == 1)


def test_missing_data_not_loss_target(tmp_path):
    d = make_fold(tmp_path)
    batch = d.make_batch([4,10],cells_per_sample=5,seed=11,role='train')
    for kind in ('nb','zinb','topic'):
        torch.manual_seed(14)
        m = make_model(kind,d,latent_dim=3,components=3,beta=0.02)
        initial_rna_components(m,kind,d,seed=12,cells_per_sample=5,device='cpu')
        initialize_neutral_centers(m,kind,d,seed=12,cells_per_sample=5,
                                   device='cpu',maximum=14)
        terms = masked_components(m,kind,batch,deterministic=True)
        assert all(torch.isfinite(value) for value in terms.values())
        # The missing RNA placeholder must not change either encoder evidence or loss.
        hacked = (batch[0].clone(),batch[1].clone(),batch[2].clone(),batch[3],batch[4])
        hacked[0][1].fill_(125.)
        hacked[1][1].fill_(100.)
        # Gene-expression likelihood and raw local encodings may depend on the
        # placeholder, but the masked posterior and loss must not.
        val1 = masked_loss(m,kind,batch,deterministic=True)
        val2 = masked_loss(m,kind,hacked,deterministic=True)
        assert torch.allclose(val1,val2,atol=1e-4,rtol=1e-5)
        masked = artificial_encoder_mask(batch[4],0.95)
        assert torch.all(masked.any(1))
        assert torch.all((masked & ~batch[4]) == 0)
        val1.backward()
        assert any(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_heldout_score_counts_exclude_missing(tmp_path):
    d = make_fold(tmp_path)
    m = make_model('nb',d,latent_dim=3,components=3,beta=0.02)
    initialize_neutral_centers(m,'nb',d,seed=7,cells_per_sample=3,device='cpu')
    result = score_split(m,'nb',d,'test',seed=4,cells_per_sample=3,
                         batch_size=2,device='cpu')
    assert result['n_RNA'] == 2
    assert result['n_ck'] == 1
    assert result['n_proteo'] == 2
    assert all(result[k] is not None for k in ('RNA','ck','proteo'))


def test_three_models_end_to_end_with_missing_samples(tmp_path):
    d = make_fold(tmp_path/'input' if False else tmp_path)
    cfg = SimpleNamespace(seed=42,output=tmp_path/'results',latent_dim=3,
                          components=3,cells_per_sample=3,batch_size=3,
                          kl_beta=0.02,center_samples=14,weight_decay=1e-4,
                          epochs_nb=1,epochs_zinb=1,lr=3e-4,zinb_lr=2e-4,
                          view_dropout=0.15,eval_cell_repeats=1,
                          checkpoint_metric='scaled_total',patience=2)
    for kind in ('nb','zinb','topic'):
        result = run_kind(kind,d,cfg,device='cpu')
        assert result['test_view_scores']['n_RNA']==2
        table = pd.read_csv(tmp_path/'results'/'fold_00'/kind/'patient_latent.csv')
        assert 'observed_RNA' in table
        assert (tmp_path/'results'/'fold_00'/kind/'model_final.pt').exists()


def test_prepare_union_preserves_aux_only_samples(tmp_path, monkeypatch):
    """Test full HDF5 joins without requiring AnnData/Scanpy at test runtime."""
    import h5py
    import types
    from prepare_acs_ccs import main
    rng = np.random.default_rng(707)
    n = 24
    metadata = pd.DataFrame({'sample_id':[f'S_{i}' for i in range(n)],
                             'patient_id':[f'P{i}' for i in range(n)],
                             'group':['ACS' if i%2 else 'CCS' for i in range(n)]})
    metadata.to_csv(tmp_path/'metadata.csv',index=False)
    rows = [{'sample_id':f'S_{i}', 'patient_id':f'P{i}',
             'library':'L1' if j%2 else 'L2', 'cell_id':f'C{i}_{j}'}
            for i in range(n-2) for j in range(6)]
    obs = pd.DataFrame(rows).set_index('cell_id')
    x = sparse.csr_matrix(rng.poisson(4.0,(len(obs),24)).astype(np.float32))
    raw = SimpleNamespace(X=x,var_names=pd.Index([f'G{i}' for i in range(24)]))
    prepared = SimpleNamespace(obs=obs,raw=raw)
    monkeypatch.setitem(sys.modules,'anndata',types.SimpleNamespace(read_h5ad=lambda _:prepared))
    store = tmp_path/'auxiliary.h5'
    with h5py.File(store,'w') as h:
        for view in ('ck','proteo'):
            grp = h.create_group(view)
            for i in range(n):
                if (view == 'ck' and i == 2) or (view == 'proteo' and i == 3):
                    continue
                grp.create_dataset(f'S_{i}',data=rng.normal(size=5).astype('float32'))
    args = SimpleNamespace(prepared=tmp_path/'Prepared_sc_Data.h5ad',aux_store=store,
                           aux_views=['ck','proteo'],sample_metadata_csv=tmp_path/'metadata.csv',
                           output=tmp_path/'prepared',fold=0,folds=4,seed=14,hvg=8,
                           hvg_method='variance',sample_column='sample_id',
                           library_column='library',patient_column='patient_id',
                           patient_rule='before_dot',stratify_column=None,
                           min_cells_per_sample=3,max_mt=15,min_genes=1,
                           max_genes=100,max_counts=20000)
    main(args)
    d = FoldData(tmp_path/'prepared')
    assert len(d.samples)==n
    assert sum(~d.observed[:,0])==2
    assert sum(~d.observed[:,1])==1
    assert sum(~d.observed[:,2])==1
    assert d.samples.groupby('patient_id').role.nunique().max() == 1
    assert pd.read_csv(tmp_path/'prepared'/'view_coverage.csv').observed.sum() == d.observed.sum()
    assert np.all(np.isfinite(d.aux_data['ck']))
    missing_ck = d.samples.index[d.samples.sample_id == 'S_2'][0]
    assert np.all(d.aux_data['ck'][missing_ck] == 0)
