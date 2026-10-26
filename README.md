# FACTMx PyTorch

FACTMx is a multimodal latent-variable model for learning a shared patient/sample representation while allowing individual views to use observation models suited to their data structure. This repository contains the generic PyTorch implementation only: cohort-specific analyses, comparator runners, and manuscript pipelines are intentionally kept outside the package.

The implementation supports simple Bernoulli, multinomial, and multivariate-Normal views; structured Gaussian-mixture, topic-model, and regularized raw-count ZINB-mixture views; contextual structured gates; optional Deep Sets context encoders; structured-first encoder-gradient-balanced training; and neutral missing-view inference.

> **Research software:** this package is intended for methodological research. Validate modeling assumptions, preprocessing, and evaluation procedures for each application.


## Raw-count NB / ZINB structured mixture (new in 0.1.2)

`ZINBMixtureHead` is an optional structured view for **per-cell raw UMI counts**.
It uses a patient-latent mixture gate and component-by-gene NB means, with
**gene-shared NB dispersion**, per-cell library-size offsets, and centered,
shrinkage-regularized gene-by-sequencing-library emission offsets. Optional
zero-inflation has nested levels: `"none"`, `"gene"`, or
`"gene_component"` (shrunken component deviations). Exact `logsumexp`
marginalization is used for the cell-component mixture.

The structured input is one tuple per view:

```python
# Each tensor must use the same patient/cell order.
# counts:       [patients, cells, selected_genes] raw integer UMI counts
# size_factors: [patients, cells] measured from FULL raw gene counts,
#               divided by a training-chosen reference depth (around 1)
# library_ids:  [patients, cells] zero-based technical sequencing library IDs
rna_view = (counts.float(), size_factors.float(), library_ids.long())

model = FACTMxModel(
    dim_latent=8,
    heads_config=[{
        "head_type": "ZINBMixture", "head_name": "rna_cells",
        "dim": 15, "dim_genes": counts.shape[-1],
        "n_libraries": 14,
        "inflation": "gene_component",  # maximum enabled complexity
        "initial_inflation": "none",   # train NB first
        "assignment_estimator": "mean",
        "objective": "marginal",
    }],
)
head = model.heads[0]
head.initialize_from_counts(rna_view, seed=42)  # training patients only
model.fit([rna_view], epochs=10, batch_size=8)
head.set_inflation_mode("gene")
model.fit([rna_view], epochs=5, batch_size=8, lr=3e-4)
# Only if held-out predictions justify more complex inflation:
# head.set_inflation_mode("gene_component")
```

Use raw counts, **never** `exp(log-normalized-X) - 1`. The library covariate
is an emission adjustment, not a patient gate input; the local encoder also
uses the estimated emission offsets to correct its count summaries. The model expects a
fixed-size sampled cell set per patient, does not implement padded-cell masks,
and intentionally does not infer size factors from the HVG subset.
The count initializer and its gate proportions must use the **training fold only**.
Full details, identifiability warnings, input preparation, and examples are in
[`docs/zinb_mixture.md`](docs/zinb_mixture.md) and
[`examples/zinb_mixture_training.py`](examples/zinb_mixture_training.py).

## Default global-covariance initialization

`FACTMxModel.fit_optimized()` now applies a train-only block-balanced global-covariance warm start by default. It initializes the shared linear latent map from whitened covariance directions, ridge-initializes compatible continuous simple-view decoders, and starts structured latent-to-gate maps at zero slope with population-logit biases. Pass explicit `population_gate_probs` when independent GMM/topic weights are available; otherwise population proportions are estimated from deterministic local assignments. Use `initialization="legacy"` (or `None`) to preserve the historical parameter initialization. See `docs/global_covariance_initialization.md`.


## Installation

FACTMx supports Python 3.10–3.12 and current PyTorch 2.x releases.

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## Minimal example

```python
import torch
from factmx_torch import FACTMxModel

# One continuous view and one structured Gaussian-mixture view.
model = FACTMxModel(
    dim_latent=2,
    heads_config=[
        {"head_type": "MultiNormal", "head_name": "continuous", "dim": 3},
        {
            "head_type": "GMM",
            "head_name": "structured",
            "dim": 3,                 # mixture components
            "dim_normal": 2,          # features per subobservation
            "assignment_estimator": "mean",
            "objective": "marginal",
            "covariance_parameterization": "softplus",
            "contextual_gate": True,
            "contextual_prior_strength": 4.0,
            "contextual_steps": 2,
        },
    ],
    encoder_config={
        "encoder_type": "Linear",
        "scale_parameterization": "softplus",
    },
    refinement_steps=1,
)

n = 24
continuous = torch.randn(n, 3)
structured = torch.randn(n, 5, 2)  # [patients, subobservations, features]
data = [continuous, structured]

history = model.fit(data, epochs=2, batch_size=8, lr=1e-3)
latent = model.get_latent_representation(data)
print(latent.shape)  # torch.Size([24, 2])
```

A runnable version is provided in [`examples/minimal_training.py`](examples/minimal_training.py).

## Recommended optimized training

`FACTMxModel.fit()` is retained for compatibility with the original PyTorch port. `FACTMxModel.fit_optimized()` adds the development-validated structured-first schedule and observed-data encoder-gradient balancing. These optimization defaults are empirical defaults rather than universal hyperparameters; applications should freeze tuning choices before final evaluation.

See [`docs/optimized_training.md`](docs/optimized_training.md) and [`docs/contextual_gate.md`](docs/contextual_gate.md).

## Missing views

`MissingViewFACTMx` keeps view availability out of the posterior-mean pathway. Per-view encoder summaries are centered using training-derived neutral centers; a missing view contributes a zero centered summary. Optionally, missingness can increase posterior uncertainty through a scale-only path.

```python
from factmx_torch import MissingViewFACTMx

# Construct as for FACTMxModel, then set training-derived centers.
model.set_encoder_centers(initial_center, refined_center)
latent, loc, scale, encoded = model.infer_masked(data, encoder_mask)
```

See [`docs/missing_view.md`](docs/missing_view.md) for semantics and checkpoint notes.

## Structured heads and held-out prediction

For structured mixture views, FACTMx separates a shared-latent prior gate from an optional view-local contextual posterior gate. Context observations may update the local gate; held-out query observations are used only for the final predictive likelihood/responsibilities. Optional Deep Sets context encoders add a learned permutation-invariant residual before analytic contextual refinement.

See [`docs/contextual_gate.md`](docs/contextual_gate.md) and [`docs/deepsets_context.md`](docs/deepsets_context.md).

## Compatibility

The canonical import is:

```python
from factmx_torch import FACTMxModel
```

Historical research code using `methods.factmx_torch` continues to work through thin compatibility wrappers:

```python
from methods.factmx_torch import FACTMxModel
```

The original uploaded source layout is retained under `legacy/original_methods_layout/` for provenance, but it is excluded from package discovery and the normal import path.

## Checkpoints

```python
model.save_checkpoint("model.pt")
restored = FACTMxModel.load_checkpoint("model.pt")
```

Package-native checkpoints store the constructor configuration required to rebuild the model and then load the `state_dict`. `MissingViewFACTMx` additionally stores its missing-view constructor settings.

## Development

Run:

```bash
pytest
python -m compileall -q src
ruff check .
```

CI runs tests and linting on Python 3.10, 3.11, and 3.12.

## Scope

This repository intentionally does **not** include cohort-specific preprocessing, outcome-analysis notebooks, comparator implementations, or application-specific plotting. Keeping those layers separate prevents cohort-specific assumptions from becoming core FACTMx dependencies.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Update the authors, repository URL, DOI, and release metadata when the public repository/publication identifiers are finalized.

## License

MIT. See [`LICENSE`](LICENSE).
