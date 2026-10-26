# FACTMx Torch change log — 12 October 2026

- Added public `initialize_global_covariance(...)`.
- Added `FACTMxModel.initialize_global_covariance(...)`.
- `FACTMxModel.fit_optimized()` now uses the validated global-covariance warm start by default.
- Explicit `population_gate_probs` are preferred when supplied; otherwise structured population gates are estimated from deterministic local assignments.
- Compatible continuous simple-view decoders are initialized by ridge regression.
- Structured latent-to-gate slopes start at zero with population-logit biases.
- Added `initialization="legacy"` / `None` opt-out for backward compatibility.
- Checkpoints preserve initialization metadata.
- Added initialization unit tests and documentation.
- Version bumped to 0.1.1.
