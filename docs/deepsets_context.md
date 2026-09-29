# Deep Sets contextual residuals

A structured mixture head can optionally add a learned permutation-invariant residual to the latent-decoded mixture gate before analytic contextual refinement.

For sample `i` and structured view `v`:

1. latent prior logits: `a_iv = W_v z_i`
2. context embedding: `h_iv = rho(mean_j phi(x_ijv))`
3. residual logits: `delta_iv = B_v h_iv`, centered across components
4. learned local prior: `pi_tilde = softmax(a_iv + scale * delta_iv)`
5. analytic refinement: responsibilities under `pi_tilde` and the emission likelihood produce soft counts used by the configured contextual update

Topic heads feed normalized word frequencies to the neural context encoder while retaining raw counts for the multinomial likelihood. GMM heads feed continuous observations directly.

Configuration keys are passed through the structured head configuration:

```python
{
    "contextual_gate": True,
    "contextual_encoder": "deepsets",
    "contextual_hidden_dim": 64,
    "contextual_residual_scale": 1.0,
    "contextual_residual_init_std": 0.01,
    "contextual_prior_strength": 4.0,
    "contextual_steps": 2,
}
```

The output residual map uses a small non-zero initialization so gradients reach the set encoder from the first optimization step. During held-out evaluation, query observations should remain excluded from the set encoder and contextual gate.
