# View-local contextual posterior for structured mixture heads

Structured FACTMx heads can distinguish two patient/sample-level mixture distributions:

1. a **latent prior gate**, `pi0 = softmax(W z)`, decoded from the shared latent representation; and
2. a **view-local contextual posterior gate**, inferred from context observations in the target structured view.

For context observations `x_ij`, the analytic update is:

```text
pi^(0) = pi0
r_ijk  = softmax_k(log pi^(t)_ik + log p(x_ij | k))
pi^(t+1)_ik = (kappa * pi0_ik + sum_j r_ijk) / (kappa + n_context)
```

`kappa` is the configurable `contextual_prior_strength`; larger values keep the contextual gate closer to the latent prior. `contextual_steps` controls the number of unrolled updates.

## Held-out semantics

For held-out prediction, context and query observations must remain separated:

```text
(other observed views + structured context) -> q(z_i)
z_i -> latent prior gate
(structured context, latent prior gate) -> contextual gate
(query observation, contextual gate) -> predictive likelihood/responsibility
```

Query observations should not enter the latent encoder or contextual gate before their predictive likelihood is evaluated.

## Main structured-head API

`StructuredMixtureHead` exposes:

- `prior_gate_probs(latent)`
- `contextual_gate_probs(latent, context_data, prior_strength=None, steps=None)`
- `predictive_gate_probs(latent, context_data)`
- `posterior_probs(latent, query_data, gate_probs=...)`
- `predictive_loss(query_data, latent, context_data, encoded_query=None)`

When contextual gating is enabled, `refined_encoder_input()` uses the contextual gate so iterative structured-summary/latent refinement is internally consistent.
