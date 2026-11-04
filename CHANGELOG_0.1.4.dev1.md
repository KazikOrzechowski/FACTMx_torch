# 0.1.4.dev1 experimental CNA encoder ablations

- Add `CenteredOneHotMultinomialHead` and `FrequencyStandardizedMultinomialHead` with explicit training-frequency fitting, missing-trial handling and safe rare-category floor.
- Register both head types with `build_head` and public package exports.
- Preserve baseline `MultinomialHead` and its loss/decoder behavior unchanged.
- Provide TCGA convergence-runner integration with `--cna-encoder` selector.
- Add explicit unit/integration tests for leakage prevention, decoder equivalence, checkpoint persistence and missing-view integration.

This is a pre-release experimental build based on FACTMx v0.1.3.
