# Experimental CNA multinomial encoder ablations

Two new **experimental** heads are available in `factmx_torch >= 0.1.4.dev1`:

- `MultinomialCenteredOneHot`: `(x - p_train)`
- `MultinomialFrequencyStandardized`: `(x - p_train) / sqrt(p_train*(1-p_train) + variance_floor)`, with an optional maximum inverse standard deviation.

`x` is each CNA locus's categorical one-hot (or normalized multinomial count) vector, and `p_train` is the observed category frequency at that locus estimated **only** from training patients with a measured CNA view. The default variance floor is `0.01` and maximum inverse standard deviation is `10.0`. They leave the standard `MultinomialHead` decoder, 5-state likelihood, smoothing epsilon and loss **unchanged**.

## Train-only fit is mandatory

```python
from factmx_torch import build_head
import torch

config = dict(head_type="MultinomialCenteredOneHot", head_name="cna_top1000",
              dim=5000, dim_pos=1000, dim_cat=5, eps=0.01)
head = build_head(config, dim_latent=10)
observations = torch.tensor(train_cna).reshape(-1,1000,5)
counts = observations.sum(dim=-1)
info = head.fit_category_frequencies((observations, counts))  # TRAIN ONLY
encoder_summary = head.encode((observations, counts))["encoder_input"]
```

For a missing-view cohort, pass `patient_mask=train_view_availability` when training arrays contain unavailable patients. Do not fit frequencies on validation/test patients; the `.encode` method fails clearly if the training frequencies were never fitted. Fitted frequencies, `category_frequencies_fitted` and `empty_training_positions` are checkpoint buffers and load with the matching experimental head type.

The alternative config is:

```python
config = dict(head_type="MultinomialFrequencyStandardized", head_name="cna_top1000",
              dim=5000, dim_pos=1000, dim_cat=5, eps=0.01,
              variance_floor=0.01, max_inverse_std=10.0)
```

**Compatibility:** The baseline `Multinomial` head and checkpoints are unchanged. Experimental checkpoints add frequency buffers, so instantiate the exact corresponding head type before loading their `state_dict`. The runner saves `cna_encoder` and `cna_encoder_config` metadata for provenance.

## Methodological caveats

- These are encoder-only changes, not new likelihood families. The decoder still emits categorical logits and the likelihood is identical.
- For one-hot data and constant `eps`, the centered baseline log-one-hot representation is an affine rescaling of centered one-hot by `log((1+eps)/eps)`. At the TCGA default `eps=0.01`, this factor is ~4.615. With balanced-covariance initialization, global block scaling can partly cancel simple feature rescalings at initialization. The experiment tests the *training-time* behavior as well, not merely input scale.
- Per-position zero-trial inputs yield a neutral-zero encoder vector; loci with no observed training trials use uniform reference frequencies and are recorded in fitting metadata.
- Per-fold frequencies must be fitted after train/validation/test assignments are known and before calculating training-only neutral centers and the global initialization.
- Better reconstruction or gradient balance does not guarantee higher clinically residualized survival C-index. Compare the same split, loss scales, initialization, seed, and checkpoint selection.
