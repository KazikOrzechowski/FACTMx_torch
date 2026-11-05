# FACTMx v0.1.4.dev1 + molecular GradNorm (experimental integration)

This package derives **exactly from the user-supplied `FACTMx_torch-0.1.4.dev1_source.zip`** and its matching `tcga_runner_cna_encoder_ablations.py`. The local-version identifier `0.1.4.dev1+gradnorm.1` is a research patch, **not** an upstream FACTMx release.

## Contents

- `factmx_torch-0.1.4.dev1+gradnorm.1-py3-none-any.whl`: build from patched sources, Python **3.10–3.12** as declared in the upstream metadata.
- `FACTMx_torch-0.1.4.dev1_gradnorm_source.zip`: complete patched Python package source, including existing and new tests.
- `tcga_runner_cna_gradnorm.py`: patched actual TCGA runner preserving the three v0.1.4.dev1 CNA variants and supporting an **opt-in** molecular GradNorm controller.
- `runner_tests/`: 4 original CNA runner contract tests and 4 additional synthetic integration tests.
- `CHANGELOG_GRADNORM.md`: concise change details.

The **TCGA runner is not standalone**: like the uploaded original, it imports `real_cohort.common` (or `common`) from the user's existing repository. It also needs the prepared patient cohort, fold manifests, H&E data, and trained preprocessing bundles. No real TCGA patient run was performed here.

## Installation

On Python 3.10, 3.11 or 3.12, preferably in the same environment used by your TCGA runner:

```bash
python -m pip install --force-reinstall --no-deps './factmx_torch-0.1.4.dev1+gradnorm.1-py3-none-any.whl'
python -c 'import factmx_torch; print(factmx_torch.__version__)'
```

Copy `tcga_runner_cna_gradnorm.py` into your TCGA runner directory so that its `real_cohort.common` import resolves. The new runner's original balancing is the default (`--simple-balancing original`), making direct paired comparisons possible.

## First paired experiment

Use your real `--project-dir`, existing preparation and your validated initialization/gate policy; hold other runner parameters fixed:

```bash
# Reference: identical v0.1.4.dev1 training objective
python tcga_runner_cna_gradnorm.py \
  --project-dir results/tcga_luad --mode all_cv --fold 0 \
  --cna-encoder centered_onehot --initialization global_legacy_gates \
  --simple-balancing original --results-tag original_centered

# Chen et al. molecular GradNorm, with the same original outer simple/H&E balancing
python tcga_runner_cna_gradnorm.py \
  --project-dir results/tcga_luad --mode all_cv --fold 0 \
  --cna-encoder centered_onehot --initialization global_legacy_gates \
  --simple-balancing gradnorm --gradnorm-alpha 1.5 \
  --gradnorm-lr 0.025 --gradnorm-progress-mode native_nll \
  --results-tag gradnorm_a15_centered

# Alpha=0 equal-gradient control: also worth running
# use --gradnorm-alpha 0 --results-tag gradnorm_a0_centered
```

Once the one-fold smoke tests pass on your machine, extend to ten folds for both LUAD and LUSC (and then joint TCGA). Separate result tags are mandatory because each run overwrites its own namespace.

## Mathematical objective and provenance

The package's existing `loss_scales` (KL followed by heads) are retained. The simple loss is constructed before the original outer balancing:

\[
L_{\mathrm{simple}} = \sum_{v\in\{\mathrm{RNA,CNA,METH}\}} w_v\,s_v\,L_v,
\quad w_v>0, \quad \sum_v w_v=3.
\]

For GradNorm, the gradient of each **already-scaled** native simple-view loss is measured at `model.encoder.loc` (the shared latent-mean encoder). The task target is the detached canonical Chen et al. gradient-rate target:

\[
G_v=\|\nabla_{\theta_{\mathrm{shared}}}(w_v s_v L_v)\|_2,\quad
r_v=\frac{\left(L_v(t)/L_v(0)\right)}{\text{mean}_j\left(L_j(t)/L_j(0)\right)},\quad
G_v^*=\overline G\,r_v^\alpha.
\]

GradNorm uses its own Adam optimizer to update the `w_v` using `sum_v |G_v - stop_gradient(G_v^*)|`, then clips to a positive floor and renormalizes. Model training backpropagates detached task weights through the unchanged original network. The outer adaptive multiplier still compares the resulting **weighted simple sum** to H&E. The structured-first hold, simple ramp, KL warmup, latent consistency, optimizer and clipping are preserved.

**This explicitly changes the effective training objective; it does not preserve MLE.** Checkpoint selection remains the **original unweighted, scaled per-view predictive NLL** on the validation fold, without KL or task weights, as in the supplied runner. It must not be presented as a Fisher-based or likelihood-principled normalization.

## Positive progress rates

Canonical `--gradnorm-progress-mode native_nll` uses the model's scaled RNA/CNA/methylation NLLs. It requires strictly **positive** task progress measures. Gaussian continuous-density NLL may legitimately become zero or negative due to learned variance: rather than silently shift it, this mode **fails with a clear error**.

Use `--gradnorm-progress-mode reconstruction` as an explicitly **noncanonical positive-proxy adaptation** when necessary. It compares mean-decoder MSE for RNA and methylation and categorical NLL for CNA to their respective first eligible minibatch values, while the network objective still uses exact native scaled NLLs. It is a separate experimental condition; do not pool results with canonical native-NLL rates. Since first-minibatch anchors depend on minibatch sampling, fixed seeds and repeated runs matter.

## Missing views, logging and checkpoints

- Only observed targets contribute to molecular training loss. An H&E-only minibatch contributes an exact zero simple term; the GradNorm update is skipped. Artificial encoder view dropout does not erase an actually observed training target.
- GradNorm weight updates occur only if **each** molecular view has at least `--gradnorm-min-observed-patients` observed targets (default 2). Other training batches still update the model with all observed targets.
- By default GradNorm runs on each eligible minibatch after the structured hold. `--gradnorm-update-every-batches` reduces overhead, at the expense of changing the controller dynamics.
- `training_history.csv` adds `gradnorm_weight__<head>`, `gradnorm_g0__<head>`, `gradnorm_gweighted__<head>`, `gradnorm_relative_rate__<head>`, update counts, and status. Fixed checkpoint-gradient diagnostics now include the *actual* learned head weight in `grad_encoder_effective__*` and `grad_decoder_effective__*`.
- `checkpoint.pt` includes the controller **best-checkpoint** training state (learned weights, first progress baselines, optimizer). `run_metadata.json` records the configuration, best and final weights, and count of updates. Existing model-only loading uses the unchanged `state_dict`. The runner does not implement automatic mid-run resume.
- All per-fold feature preprocessing, per-locus CNA training frequencies, missing-view masks, H&E initialization and 8:1:1 fold roles remain in their original code paths.

## Verification performed

- **69 tests passed:** package regression tests (including the original CNA encoding head tests), GradNorm unit/actual-head tests, and patched-runner selector/short synthetic trainer tests.
- Direct package wheel build completed; the wheel includes `factmx_torch/gradnorm.py` and the local-version label.
- Real TCGA training, end-to-end missing-view data paths (`real_cohort.common`) and GPU/CUDA have **not** been verified in this container.
- The current test interpreter is Python 3.13, outside the declared upstream supported range. Production deployment should use Python 3.10–3.12 and rerun tests there.

Reference: Chen et al., *GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks*, ICML 2018, https://proceedings.mlr.press/v80/chen18a.html.
