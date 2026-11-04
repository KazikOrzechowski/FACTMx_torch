# FACTMx PyTorch v0.1.3 — 2026-11-04

- Added `initialize_masked_global_covariance` as a public core initializer
  plus `MissingViewFACTMx.initialize_masked_global_covariance`.
- Ported existing TCGA/IMMUCAN train-only mask-aware balanced SVD, bounded
  whitening, center-corrected encoder bias, observed-only Gaussian decoder ridge,
  and structured gate policies. Removed NumPy runtime dependency from this
  initializer (PyTorch-only core dependency preserved).
- Added independent `gate_weight_mode` and `gate_bias_mode` keyword options to
  the complete-data initializer and `FACTMxModel.initialize_global_covariance`
  and `fit_optimized`. Backward-compatible defaults: `zero` / `population`.
- Kept checkpoint parameter names and existing probabilistic model unchanged.
- Included frozen external adapter reference tests for strict numerical parity
  across the four gate mode combinations.
- Preserved historical masked initialization metadata method string for
  downstream provenance compatibility.
