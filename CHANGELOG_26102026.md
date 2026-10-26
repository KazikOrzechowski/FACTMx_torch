# FACTMx-torch v0.1.2 — 26 October 2026

Additive source release derived from the user's 0.1.1 zip. The original GMM,
Topic, and simple heads remain supported; no cohort-specific analysis code
is bundled and no real-cohort training was performed for this release.

- Added `ZINBMixtureHead` to the canonical package, `build_head()`, and public
  imports. It integrates with standard and missing-view FACTMx inference,
  contextual posterior gates, checkpointing, and the optimized training path.
- Added raw-count NB emission with gene-level inverse dispersion, patient
  mixture gates, explicit per-cell size factors, centered/shrunk library-by-gene
  technical log-rate offsets, and exact whole-cell `logsumexp` marginalization.
- Added optional gene-specific and hierarchical gene-component ZINB inflation,
  low-inflation initialization, Beta shrinkage of gene inflation, dispersion
  shrinkage, component-deviation regularization, and a saveable switch between
  NB and ZINB submodels.
- Added train-only data-driven component emission and gate initialization,
  input checks for integer UMI counts, batch-adjusted local neural summaries,
  and raw count sampling.
- Corrected optimized-training gradient balancing when an objective group
  has no active heads (avoids autograd on an unconnected zero loss).
- Added an executable synthetic example, documentation, and focused tests for
  likelihood math, gradients, technical offsets, checkpoints, initialization,
  contextual prediction, and missing-view integration.
- Bumped package metadata and citation version from 0.1.1 to 0.1.2.

**Scope:** This is an implementation-level release with synthetic smoke tests.
No biological claims, ACS/CCS batch-invariance guarantees, fitted model
parameters, or validated real-data performance are implied.
