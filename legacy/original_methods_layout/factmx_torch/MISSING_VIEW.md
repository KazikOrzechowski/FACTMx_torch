# Missing-view FACTMx

`MissingViewFACTMx` is the package-level missing-view extension of `FACTMxModel`.
It is exported directly from `methods.factmx_torch`.

## Model behavior

- `neutral_centering=True` by default.
- `center_ema=0.99` by default.
- `availability_to_scale=False` by default.
- `missing_scale_init=-2.5` is used only when the scale path is enabled.
- Missingness is not concatenated into the posterior-mean pathway.
- Training-derived encoder centers are supplied with `set_encoder_centers()`.
- `infer_masked()` accepts a Boolean `[batch, n_heads]` encoder mask.
- `posterior_from_encoded()` supports full-vs-masked comparisons without a second head-encoding pass.

Latent-consistency regularization is intentionally not part of the model class.
It is a training-policy term and should default to weight `0` unless explicitly
enabled by a runner or experiment.

## Checkpoint compatibility

The names and shapes of the missing-view buffers/parameters are preserved from
the NSCLC embedded implementation:

- `encoder_center_initial`
- `encoder_center_refined`
- `encoder_centers_initialized`
- `missing_scale_raw` when `availability_to_scale=True`

Consequently, an existing NSCLC `state_dict` can be loaded with `strict=True`
when the packaged model is instantiated with the configuration stored alongside
the checkpoint, including the checkpoint's `availability_to_scale` setting.
