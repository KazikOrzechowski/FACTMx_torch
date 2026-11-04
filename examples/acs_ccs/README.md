# ACS/CCS: union-cohort `all_cv` runners for FACTMx

This is the **naturally missing-view** revision of the ACS/CCS three-runner example.
It uses the union of samples with **at least one observed view**, rather than the
intersection of RNA + all auxiliary views. The input `.raw.X` is verified as
original integer UMI counts. Three RNA alternatives share one preparation:

| Runner | Structured RNA head | Other view heads |
|---|---|---|
| `nb` | Negative-binomial mixture; gene dispersions and technical library offsets | Gaussian CK / neutrophils / proteomics / clinical |
| `zinb` | Same mixture initialized as NB, then gene-specific zero inflation | Identical Gaussian heads |
| `topic` | Multinomial Topic mixture, no explicit technical offset | Identical Gaussian heads |

The model is now `MissingViewFACTMx`: neutral encoder summaries are estimated
from **observed training data only** and masked when a view is unavailable.
View availability is *not* inserted into the deterministic latent-mean pathway.

## Install

From the repository root:

```bash
python -m pip install -e .
python -m pip install 'scanpy[skmisc]>=1.10' anndata scipy scikit-learn pandas h5py
```

## 1. Prepare each outer fold (union cohort)

```bash
python examples/acs_ccs/prepare_acs_ccs.py \
  --prepared data/Prepared_sc_Data.h5ad \
  --aux-store ../MOFAplus/processed_data/general_data_store \
  --aux-views ck neutro proteo clinical \
  --patient-rule before_dot \
  --fold 0 --folds 10 --hvg 500 \
  --output results/acs_ccs/prepared/fold_00
```

If auxiliary-only sample names cannot be **safely and exactly** converted to
patient IDs with `--patient-rule before_dot`, supply a CSV with unique
`sample_id,patient_id` columns (optional `group`) via:

```bash
  --sample-metadata-csv data/acs_ccs_sample_patient_map.csv
```

**Do not use `--patient-rule sample` for repeated timepoints.** When an
explicit `.obs['patient_id']` is available, an auxiliary-only sample without
an authoritative mapping requires the metadata CSV. In all cases verify that
cross-library replicates and repeat timepoints map to the same patient.

- A sample with sufficient QC-passed RNA cells has `observed_RNA=True`.
- Samples with no usable RNA are still included if `ck`, `neutro`, `proteo`,
  or `clinical` is observed.
- Samples missing a subset of auxiliary views remain eligible.
- All-observed-false samples are excluded (no evidence from any modality).
- Empty/nonfinite auxiliary arrays are treated as **missing entire views**,
  never as zero measurements. Partially observed vectors require an explicit
  per-feature masking model and are therefore treated as missing views here.
- An auxiliary view with **no training examples in a fold** raises a clear
  error; it cannot be standardized or learned using held-out patients.
- RNA gene selection uses only RNA-observed training patients; 500 batch-aware
  HVGs with `seurat_v3` by default, or the explicit `variance` fallback.
- Auxiliary z-scoring uses only **observed training vectors per view**.
- Train, validation, test are patient-grouped 8:1:1 with test fold `f`,
  validation fold `(f+1) mod 10`, remaining 8 folds for training.

Outputs:

- `sample_manifest.csv`: sample ID, patient ID, role, `n_cells` and separate
  `observed_RNA`, `observed_ck`, etc. boolean columns.
- `view_masks.csv` and `view_coverage.csv`: mask audit by role and view.
- `cells.csv`, `counts_selected.npz`, `selected_genes.csv`: available RNA
  cells only; RNA-missing samples have no fabricated cell rows.
- `auxiliary.npz`: observed standardized vectors; **zeros in missing rows are
  computationally neutral placeholders identified by explicit masks**.
- `aux_train_scalers.json`, `sample_library_composition.csv`,
  `invalid_auxiliary_values.csv`, `preparation.json`.

The runner rejects older complete-case prepared directories without view masks.
**Re-run the preparer once per fold.**

## 2. Train NB / ZINB / Topic on the same prepared fold

```bash
python examples/acs_ccs/acs_ccs_runner.py \
  --prepared-fold results/acs_ccs/prepared/fold_00 \
  --model all --output results/acs_ccs/models \
  --latent-dim 20 --components 15 \
  --cells-per-sample 64 --batch-size 8 \
  --epochs-nb 40 --epochs-zinb 20 \
  --view-dropout 0.15 --checkpoint-metric scaled_total \
  --device auto
```

- Both natural view masks and artificial dropout affect **encoder evidence**.
  Artificially hidden *observed* views remain supervised reconstruction targets,
  whereas naturally missing views never contribute any observed-data head loss.
- At least one actually observed view is retained in each training sample's
  encoder after artificial dropout.
- Neutral view-local encoder references are initialized from observed training
  samples and updated using train-only EMA with the **natural masks**, never
  with held-out subjects or artificial view-dropout patterns.
- Missing RNA is represented with a nonnegative zero placeholder and neutral
  depth/library tensors, but both its encoded evidence and its RNA loss are
  masked. This is **not** an RNA measurement of zero counts.
- Validation selection defaults to `scaled_total`, a weighted aggregate of
  scores for **observed targets only**. RNA uses disjoint context/query cells;
  an auxiliary view's score uses leave-one-view-out latent inference so the
  target does not inform its own prediction. View-specific counts are exported.
  The optional `--checkpoint-metric rna` requires observed validation RNA.
- The Topic and NB/ZINB likelihood families are not directly comparable in
  absolute magnitude. Use common downstream criteria for three-way ranking.
- No entire held-out dataset is ever used to fit heads, HVGs, neutral centers,
  scalers or select checkpoints. Validation targets select checkpoints only.
- `sample_latent.csv`, `train_latent.csv`, `val_latent.csv`, and
  `patient_latent.csv` include observed-view masks and sample/patient IDs.

Exports also include training history, view-specific held-out scores and
observation counts, mixture gates, gene dispersion/inflation (NB/ZINB),
technical gene-library offsets (NB/ZINB), and topic profiles (Topic).

## 3. Full 10-fold `all_cv`

```bash
for f in $(seq 0 9); do
    printf -v ff '%02d' "$f"
    python examples/acs_ccs/prepare_acs_ccs.py \
        --prepared data/Prepared_sc_Data.h5ad \
        --aux-store ../MOFAplus/processed_data/general_data_store \
        --patient-rule before_dot \
        --fold "$f" --folds 10 \
        --output "results/acs_ccs/prepared/fold_${ff}"
    python examples/acs_ccs/acs_ccs_runner.py \
        --prepared-fold "results/acs_ccs/prepared/fold_${ff}" \
        --model all --output results/acs_ccs/models \
        --checkpoint-metric scaled_total
 done
```

Include `--sample-metadata-csv` in the preparation command when necessary.
Every eligible patient appears in a held-out test role exactly once per 10-fold
run, and all their timepoints stay together. Each fold independently fits all
transformations and model parameters.

**Do not concatenate foldwise latent coordinates and treat them as aligned.**
For clinical comparisons, fit clinical probes or Cox directions on each fold's
training representation and concatenate the held-out *scalar predictions*.

## Limitations

1. RNA and auxiliary views are **missing at whole-sample/whole-view level**.
   Partial cytokine panel missingness is treated as missing the whole panel.
2. The Topic head does not have gene-by-library technical offsets, whereas the
   NB/ZINB heads do. Interpret batch comparisons accordingly.
3. Library IDs that appear only in validation/test cannot have train-derived
   library offsets; the runner warns, and affected folds are not a fair
   technical-correction test.
4. The model is unsupervised; ACS/CCS clinical biomarkers used as inputs could
   make subsequent diagnosis association analyses circular.
5. A residualized/held-out patient-level clinical validation still requires
   downstream probes. Real patient convergence and batch-effect reduction have
   **not** been checked in this delivery.
