# View-local contextual posterior for structured mixture heads

Structured FACTMx heads now distinguish two patient-level mixture distributions:

1. **latent prior gate**
   `pi0 = softmax(W z)`, decoded from the shared patient representation;
2. **view-local contextual posterior gate**, inferred from the target structured
   view's *context* observations only.

For context observations `x_ij` in structured view `v`, the posterior update is

```text
pi^(0) = pi0
r_ijk  = softmax_k(log pi^(t)_ik + log p(x_ij | k))
pi^(t+1)_ik = (kappa * pi0_ik + sum_j r_ijk) / (kappa + n_context)
```

The update is differentiable and unrolled for a small number of iterations.
`kappa` is a prior pseudo-count: larger values keep the gate closer to the
latent-decoded prior; smaller values allow within-view patient context to have
more influence.

## Held-out semantics

For validation/test patients:

```text
(simple views + structured context) -> q(z_i)
                               z_i -> pi0_i,v
(context observations in view v, pi0_i,v) -> contextual gate pi_i,v
(query observation x_iq, pi_i,v) -> p(component | x_iq, patient context)
```

Query observations never enter `q(z_i)` or the contextual gate.  They only
enter the final query likelihood/responsibility calculation.

## Benchmark defaults

The development configuration currently uses:

```yaml
contextual_gate: true
contextual_prior_strength: 4.0
contextual_steps: 2
```

These are exposed configuration values, not simulator-truth-derived constants.
They should be frozen from development before final benchmark seeds are run.

The contextual gate is enabled for `structured_context`, `frozen_factmx`, and
`full_factmx`.  It is deliberately disabled for `encoded_simple`, otherwise the
encoded-simple decomposition step would receive structured patient context and
would no longer isolate simple-view conditioning.

## Main API

Every `StructuredMixtureHead` now provides:

- `prior_gate_probs(latent)`
- `contextual_gate_probs(latent, context_data, prior_strength=None, steps=None)`
- `predictive_gate_probs(latent, context_data)`
- `posterior_probs(latent, query_data, gate_probs=...)`
- `predictive_loss(query_data, latent, context_data, encoded_query=None)`

`refined_encoder_input()` also uses the contextual gate when enabled, making the
structured-summary/latent refinement loop consistent with the new local
posterior.

## Training

The predictive benchmark objective now scores each structured query set under
its context-only local gate.  Consequently gradients flow through both the
latent-decoded prior gate and the contextual update.  Existing structured-first
warm-up and once-per-epoch encoder-gradient balancing are unchanged.
