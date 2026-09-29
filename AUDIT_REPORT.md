# FACTMx PyTorch GitHub package audit

**Package:** `factmx-torch` 0.1.0  
**Prepared:** 29 September 2026  
**Authoritative source:** uploaded `factmx_torch(4).zip`  
**Target scope:** generic/core FACTMx only

## Executive summary

The uploaded FACTMx PyTorch implementation has been converted into a conventional installable GitHub repository using a `src/` layout. The statistical implementation was preserved. The only executable behavior change made to the core source is package-level checkpoint metadata support for `MissingViewFACTMx`, allowing its missing-view constructor settings to round-trip through `save_checkpoint()` / `load_checkpoint()`; no inference equation, likelihood, posterior update, masking rule, loss, or training schedule was changed.

The repository contains the generic model, heads, encoders, missing-view extension, and optimized-training utilities. Cohort-specific analysis code, survival code, comparator runners, preprocessing pipelines, and manuscript workflows are not included in the installable package.

## Repository structure

- `src/factmx_torch/` — canonical implementation
- `src/methods/factmx_torch/` — thin historical import-compatibility wrappers
- `tests/` — unit and smoke tests
- `examples/minimal_training.py` — synthetic end-to-end example
- `docs/` — generic technical documentation
- `legacy/original_methods_layout/` — preserved source snapshot outside package discovery
- `.github/workflows/ci.yml` — Python 3.10/3.11/3.12 test and lint matrix
- `.github/ISSUE_TEMPLATE/` and `.github/PULL_REQUEST_TEMPLATE.md` — contribution templates
- `pyproject.toml` — build metadata, dependencies, pytest and ruff configuration
- `LICENSE` — MIT
- `CITATION.cff`, `CONTRIBUTING.md`, `.gitignore`

## Changes made

### Packaging and API

1. Moved the canonical implementation to `src/factmx_torch/`.
2. Added PEP 621/setuptools packaging in `pyproject.toml`.
3. Declared Python support as `>=3.10,<3.13` and the mandatory runtime dependency as `torch>=2.0`.
4. Added package version `0.1.0` and `py.typed` marker.
5. Expanded top-level exports to include the model classes, head classes, encoder classes, factories, and training utilities already present in the source.
6. Added historical `methods.factmx_torch` wrappers, including submodule wrappers, so existing research imports resolve to the canonical implementation rather than a duplicated implementation.

### Compatibility and persistence

1. Preserved the historical `GMM_prop` configuration alias.
2. Preserved existing constructor configuration names and the established head/encoder behavior.
3. Preserved missing-view state names (`encoder_center_initial`, `encoder_center_refined`, `encoder_centers_initialized`, and optional `missing_scale_raw`).
4. Refactored checkpoint metadata collection into `_checkpoint_config()` without changing the base model state dictionary.
5. Added missing-view constructor metadata (`neutral_centering`, `center_ema`, `availability_to_scale`, `missing_scale_init`) to package-native missing-view checkpoints so strict reload reconstructs the correct model shape/configuration.
6. Added `name` to new base-model checkpoint metadata; older checkpoint payloads remain loadable because `name` is optional in the constructor.

### Documentation and repository hygiene

1. Replaced the internal/port-oriented README with an external-facing README.
2. Kept the technical descriptions of contextual gates, Deep Sets context, optimized training, and missing-view behavior while removing application/benchmark-specific references from the public docs.
3. Preserved the uploaded source and original documentation under `legacy/` for provenance; this directory is excluded from package discovery and ruff scanning.
4. Added MIT license, contribution guidance, citation metadata, issue templates, PR template, and `.gitignore`.

### Tests added

The suite covers:

- canonical and historical imports;
- model construction;
- forward/inference shapes;
- simple head decoding/loss;
- structured GMM construction and the `GMM_prop` alias;
- contextual Deep Sets gate probabilities;
- normalized topic profiles;
- legacy `fit()` smoke training;
- structured-first `fit_optimized()` smoke training;
- training schedule/EMA utilities;
- missing-view masks and scale-only missingness behavior;
- base checkpoint save/load and deterministic inference equality;
- missing-view checkpoint save/load and state equality.

## Validation performed

Validation was run in the available execution environment using Python 3.13.5 and PyTorch 2.10.0+cpu. Python 3.13 is intentionally outside the declared support range, but it was useful as an additional forward-compatibility smoke environment.

Results:

- source-tree pytest: **10 passed**;
- `compileall` over source, tests, and example: **passed**;
- minimal synthetic training example: **passed**;
- offline wheel build using locally installed setuptools/wheel: **passed**;
- install of built wheel into an isolated target directory: **passed**;
- pytest against the installed wheel: **10 passed**;
- canonical `factmx_torch` import: **passed**;
- historical `methods.factmx_torch` import and submodule imports: **passed** and resolve to identical canonical classes.

The produced wheel was `factmx_torch-0.1.0-py3-none-any.whl`.

## Scientific behavior preserved

A diff against the authoritative uploaded implementation shows no executable changes in `encoders.py` or `training.py`. `heads.py` received documentation-only cleanup. Changes in `model.py` and `missing_view.py` are limited to checkpoint configuration metadata described above. The likelihoods, structured marginal objective, contextual posterior update, Deep Sets residual path, latent refinement, missing-view posterior computation, structured-first schedule, and encoder-gradient balancing implementation were not altered.

## Legacy code isolation

The exact uploaded Python source and technical markdown files are retained under:

`legacy/original_methods_layout/factmx_torch/`

They are not included in setuptools package discovery. The installed compatibility namespace contains wrappers only; it does not contain a second implementation. This avoids divergence while preserving historical import paths.

## Unresolved technical debt / release tasks

1. **Python support matrix:** the local environment did not provide Python 3.10–3.12 interpreters. GitHub Actions is configured to test all three declared versions after the repository is pushed.
2. **Ruff execution:** ruff was not installed in the offline execution environment and could not be downloaded. Ruff configuration and a CI lint step are included; the local validation used Python compilation plus the full pytest suite.
3. **GPU CI:** no CUDA/GPU job is configured. Current tests validate CPU behavior only.
4. **Historical external checkpoints:** package-native base and missing-view checkpoint round-trips are tested, and stable state names are preserved, but no independent historical checkpoint file was supplied as a regression fixture for this package build.
5. **Legacy Keras/TensorFlow serialization:** arbitrary Keras `Sequential` configuration deserialization remains unsupported, matching the inherited PyTorch port limitation.
6. **Missing-view center estimation:** estimating neutral encoder centers remains a caller/training-runner responsibility by design; it is not silently inferred from evaluation data.
7. **Citation/repository metadata:** `CITATION.cff` currently uses `FACTMx contributors` and has no DOI or repository URL. Replace these fields when the public authorship, repository URL, release tag, and publication identifiers are finalized.
8. **Distribution naming:** `factmx-torch` is used as the distribution name and `factmx_torch` as the import package. Availability of the distribution name on the intended public package index should be checked before publication.
9. **Security/trust of checkpoint files:** checkpoint loading uses PyTorch serialization and should be restricted to trusted checkpoint files.

## Recommended release sequence

Create the GitHub repository from the supplied ZIP, push the initial commit, let the Python 3.10–3.12 CI matrix run, resolve any environment-specific lint/test issue, fill in final authorship/repository/DOI metadata, and tag the first public release. No application-specific analysis code needs to be moved into this repository for that release.
