# ACS/CCS all_cv missing-view update — 26 October 2026

- Replaced complete-case auxiliary intersection with union-of-observed-samples
  preparation, including auxiliary-only samples and RNA-missing patients.
- Added per-view `observed_<view>` masks in the sample manifest, coverage CSV,
  invalid-value audit, and train-only observed-vector standardization.
- Replaced the three runner model instances with `MissingViewFACTMx` and
  train-only neutral encoder centers, updated by train-only EMA.
- Added 0.15 training-only artificial view dropout (not dropping targets),
  observed-only reconstruction losses, and mask-aware inference/export.
- Added observed-target-only, RNA context/query and leave-one-auxiliary-view-out
  held-out likelihoods with counts per view and a default scaled-total
  checkpoint criterion.
- Added tests for natural missingness, masked placeholder invariance,
  patient-disjoint splits, auxiliary-only preparation, and synthetic training
  of all three head types.
- Existing core FACTMx model/head implementation and backward compatibility
  are unchanged; only ACS/CCS examples are updated.
