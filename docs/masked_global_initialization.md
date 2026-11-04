# Mask-aware global initialization (FACTMx v0.1.3)

This initialization reproduces the original **train-only**, missing-view-aware
TCGA/IMMUCAN adapter, now implemented inside the core `factmx_torch` package.
Only the model-initialization mathematics is in the package: neither fold
orchestration nor independent structured emission fitting is moved here.

## Installation

```bash
python -m pip install ./factmx_torch-0.1.3-py3-none-any.whl
python -c "import factmx_torch; print(factmx_torch.__version__)"
```

## Public API

```python
from factmx_torch import MissingViewFACTMx, initialize_masked_global_covariance

# model is a MissingViewFACTMx whose independent Topic/GMM components and
# small random gate weights have ALREADY been initialized from training data.
# model.set_encoder_centers(initial=train_neutral_centers) was called earlier.
# summaries: list of n_heads arrays [N, encoder_dim], aligned on training IDs.
# observed_mask: boolean array [N, n_heads] aligned with summaries and targets.
# simple_targets: dict head_name -> [N, Gaussian_feature_dim]
# population_gate_probs: dict structured_head_name -> [n_components]

info = model.initialize_masked_global_covariance(
    summaries=train_summaries,
    availability=observed_mask,
    simple_targets=train_gaussian_targets,
    population_gate_probs=train_population_mixture_weights,
    ridge_alpha=1e-4,
    min_relative_singular=1e-5,
    max_whitening_gain=100.0,
    gate_weight_mode="preserve_random",
    gate_bias_mode="zero",
)
# Equivalent:
# info = initialize_masked_global_covariance(model, train_summaries, observed_mask, ...)
```

The package initializer accepts NumPy arrays and PyTorch tensors without
requiring NumPy as a core package dependency.

## Gate policies

| Experiment | gate_weight_mode | gate_bias_mode |
| --- | --- | --- |
| Full global | `zero` | `population` |
| Global + random gate weights | `preserve_random` | `population` |
| Global + complete legacy gate initialization | `preserve_random` | `zero` |
| Zero weight + zero bias (control) | `zero` | `zero` |

`preserve_random` retains the actual weights that are already in
`head.mixture_logits.weight`; it **does not** draw fresh random weights and
therefore does not advance the random-number generator. The calling runner
must seed before constructing the model and must preserve its historical
`Normal(0, 0.02)` gate-weight initialization if that is the intended control.

`population` applies centered log independent mixture weights. For exact
masked adapter parity, provide `population_gate_probs` for every structured
head **even with** `gate_bias_mode='zero'`. The masked initializer requires
these training-derived proportions and records gate provenance consistently.
The complete-data initializer allows omitted population weights in zero-bias
mode because no population estimate is mathematically necessary there.

## Numerical behavior

- Observed training rows determine the mean per view. Missing summaries are
  imputed with that training mean only to estimate global covariance.
- Active view blocks are balanced to unit total variance, SVD is computed in
  float64 on CPU, and leading latent scores are whitened.
- Weak singular directions below `min_relative_singular` are dropped;
  `max_whitening_gain` limits amplification in the remaining directions.
- The encoder bias accounts for the neutral center already subtracted by
  `MissingViewFACTMx`.
- Ridge fitting uses **only patients observed for each Gaussian head**.
- This function changes the shared encoder mean, compatible simple decoder
  means, and structured gate weights/biases. It does not change emissions,
  posterior scales, optimizer, checkpoint naming, or objective.
- The model-level initializer returns an information dictionary with rank,
  scales, gate policies, sources, and post-initialization diagnostics. Existing
  keys (including `method='global_covariance_mask_aware_v0.1.1'`) are kept for
  compatibility with earlier analysis and provenance tools.

## Integrating the existing cohort runners

Replace the external import:

```python
from real_cohort.masked_global_initialization import initialize_masked_global_covariance
```

with:

```python
from factmx_torch import initialize_masked_global_covariance
```

Retain the runner's own `seed_before_model` helper (it seeds Python, NumPy,
PyTorch and CUDA). No changes to the downstream call arguments are needed.
The representative updated TCGA and NSCLC runners in the companion integration
bundle use the package API directly.

Before running a cohort, verify:

```bash
python -c "import factmx_torch; print(factmx_torch.__version__, factmx_torch.__file__)"
```

It must print `0.1.3` from the intended environment. To reproduce prior
experiments, do not change H&E or Topic independent-fit restart counts, seeds,
model configuration, splits, or preprocessing while comparing gate policies.

## Compatibility and tests

The complete-data covariance initializer keeps its 0.1.2 default gate policies.
A frozen reference copy of the external real-cohort adapter is kept under
`tests/reference_masked_initializer_v012.py` so the regression tests can compare
all pretraining `state_dict` tensors **exactly** for all four gate-policy
combinations. The package does not depend on this test-only reference module.
