# Structured-first optimized training

`FACTMxModel.fit()` preserves the historical training loop. `FACTMxModel.fit_optimized()` provides the newer structured-first, encoder-gradient-balanced path using only observed-data losses.

The default schedule consists of:

- an initial structured-only hold period;
- a linear ramp that reintroduces simple-view reconstruction;
- linear KL warm-up;
- periodic balancing of scaled simple-view and structured-view encoder-gradient norms;
- an adaptive simple-view multiplier constrained to a configured range.

The balancing rule uses no labels, simulator truth, component truth, or post-fit recovery metric. Gradient norms are evaluated infrequently (once per epoch by default), then the resulting multiplier is reused for the remaining minibatches in that epoch. The EMA decay is adjusted for the number of batches between updates.

The main controls are arguments to `fit_optimized()` and correspond to `StructuredFirstBalanceConfig`:

- `structured_hold_epochs`
- `simple_ramp_epochs`
- `balance_encoder_gradients`
- `balance_update_every_epochs`
- `balance_ema_beta`
- `balance_min_simple_multiplier`
- `balance_max_simple_multiplier`
- `balance_target_gradient_ratio`
- `kl_final_beta`
- `kl_warmup_epochs`

These defaults are research defaults, not universal optimum values. Tune them only on development data and freeze them before final evaluation.
