# Missing-view FACTMx

`MissingViewFACTMx` extends `FACTMxModel` with neutral per-view encoder centers and optional scale-only missingness handling.

## Model behavior

- `neutral_centering=True` by default.
- `center_ema=0.99` by default.
- `availability_to_scale=False` by default.
- Missingness is not concatenated into the posterior-mean pathway.
- Training-derived encoder centers are supplied with `set_encoder_centers()`.
- `infer_masked()` accepts a Boolean `[batch, n_heads]` encoder mask.
- `posterior_from_encoded()` supports full-vs-masked comparisons without a second head-encoding pass.

When `availability_to_scale=True`, a missing view can increase posterior uncertainty through a learned non-negative scale contribution without directly shifting the posterior mean.

Latent-consistency regularization is intentionally outside the model class; it is a training-policy term rather than part of the probabilistic model API.

## Checkpoint compatibility

Stable missing-view state names include:

- `encoder_center_initial`
- `encoder_center_refined`
- `encoder_centers_initialized`
- `missing_scale_raw` when `availability_to_scale=True`

Package-native checkpoints also store `neutral_centering`, `center_ema`, `availability_to_scale`, and `missing_scale_init` so `MissingViewFACTMx.load_checkpoint()` can rebuild a matching instance before loading its state.
