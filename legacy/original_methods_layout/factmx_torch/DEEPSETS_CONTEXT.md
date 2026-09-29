# Deep Sets + analytic contextual refinement

Structured mixture heads can now use a learned permutation-invariant residual on the latent-decoded mixture gate before the existing analytic responsibility refinement.

For patient i and structured view v:

1. latent prior logits: `a_iv = W_v z_i`
2. Deep Sets context embedding: `h_iv = rho(mean_j phi(x_ijv))`
3. residual logits: `delta_iv = B_v h_iv` (centered across K components)
4. learned local prior: `pi_tilde = softmax(a_iv + scale * delta_iv)`
5. analytic refinement: responsibilities are computed under `pi_tilde` and the emission likelihood, then soft counts update the gate with the configured `contextual_prior_strength` and `contextual_steps`.

Topic heads feed normalized word frequencies to Deep Sets while keeping raw counts for the multinomial likelihood. GMM heads feed the continuous observations directly.

The output residual map is initialized with a small non-zero standard deviation (default 0.01). Exact zero initialization is deliberately avoided because it blocks first-step gradients into the set encoder.

Configuration:

```yaml
training:
  contextual_gate: true
  contextual_encoder: deepsets
  contextual_hidden_dim: 64
  contextual_residual_scale: 1.0
  contextual_residual_init_std: 0.01
  contextual_prior_strength: 4.0
  contextual_steps: 2
```

For `encoded_simple`, the adapter disables all structured contextual inference, preserving the decomposition semantics. Query observations never enter the set encoder or contextual gate during held-out evaluation.
