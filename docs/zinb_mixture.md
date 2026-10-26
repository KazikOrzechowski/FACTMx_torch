# NB / ZINB mixture structured head

`ZINBMixtureHead` is a **new** structured observation head in FACTMx-torch
0.1.2. It models a patient's set of single-cell **raw UMI counts** as a
mixture of biological components, with a mixing distribution decoded from the
FACTMx patient latent `z`.

## Probabilistic specification

For patient `i`, cell `j`, gene `g`, component `k`, technical library `b`:

- Component gate: `w_ik = softmax(W z_i + a)_k`.
- Biological component expression rate: `exp(alpha_kg)`.
- Library-specific offset: `delta_bg`, centered across libraries and penalized
  towards zero; **never** supplied to the patient gate.
- Fixed per-cell size factor: `s_ij` **from full raw counts**, normalized so
  that a typical cell is near 1 (e.g., total UMIs / training median UMIs).
- NB mean: `mu_ijgk = s_ij * exp(alpha_kg + delta_bg)`.
- NB inverse dispersion: `theta_g = clamp(softplus(d_g), min_theta, max_theta)`,
  **shared across biological components and libraries**.
- ZINB structural-zero probability: either `0`, `sigmoid(a_g)`, or
  `sigmoid(a_g + u_kg - mean_k(u_kg))`.

One **whole cell** selects a component, not each gene independently:

```
log p(x_ij | z_i) = logsumexp_k [ log w_ik + sum_g log p(x_ijg | k,b,s) ]
```

With `p0_NB = (theta / (theta+mu))**theta`, the zero probability is
`pi + (1-pi)*p0_NB`; for `x>0` it is `(1-pi)*p_NB(x)`.
All terms are accumulated in log space. The local assignment estimator
defaults to `"mean"` (deterministic soft probabilities, rather than
near-hard Gumbel sampling) for stable patient encoder summaries. `normalize_by_genes=True` optionally
normalizes each whole-cell *component* log-likelihood by gene count. Since
this changes the component posterior temperature, the default `False` is
recommended when faithful mixture probabilities matter; instead adjust
FACTMx view loss scales when matching losses of other modalities.

## Why inflation is restricted

ZINB can poorly identify `mu`, `theta`, and `pi`: a low biological expression
mean, small NB theta (overdispersion), or a large inflation probability can
all explain zeros. A biological *mixture* adds another explanation for zeros.

The recommended nested comparison, using identical patient folds and gene sets:

| Configuration | `inflation` (maximum) | `set_inflation_mode(...)` |
|---|---|---|
| M0 NB | `none` or higher | `none` |
| M1 gene-ZINB | `gene` or higher | `gene` |
| M2 hierarchical ZINB | `gene_component` | `gene_component` |

`gene_component` deviations are initialized **exactly at zero** and strongly
shrunk with `component_inflation_sd=0.35`,
`component_inflation_penalty=1.0` (per-parameter mean-square penalty). The
inflation logits start at `logit(0.01)`. A weak Beta(1,20) penalty on
gene-specific inflation (`gene_inflation_prior_strength=0.05`) discourages
high pi when unnecessary. Log-dispersion is weakly shrunk toward the initial
`theta=10` (`dispersion_prior_strength=0.01`) to reduce ZINB/NB
non-identifiability. Both priors can be disabled explicitly for sensitivity
analysis. Only enable richer inflation after
NB- and gene-ZINB-level *held-out* evidence supports it. Switching modes
changes an internal registered buffer, preserved in a saved checkpoint.

Technical gene-by-library offsets are mean-centered to remove their additive
redundancy and regularized (`batch_effect_sd=0.25`, penalty 1 by default).
This is **not a guarantee of batch invariance**. When a patient/phenotype
appears in only one library, biological/technical effects may be non-identifiable.
Use technical replicate cells from the *same sample in different libraries*
for evaluation. Prefer one shared gene order and cross-library HVG selection.

## Input contract

One view is a tuple `(counts, size_factors, library_ids)`:

- `counts`: dense PyTorch tensor `[N, C, G]` of nonnegative raw integer UMIs;
  floating dtype storing integers is allowed. **Not** normalized expression.
- `size_factors`: `[N, C]` or `[N, C, 1]` finite positive depth multipliers
  from **all genes**, not just the selected genes. Supply externally.
- `library_ids`: `[N, C]` with `0 ... B-1`; may be omitted when `B=1`.
- Each patient has exactly `C` real cells in this implementation. When patient
  sizes vary, sample cells (with a documented fixed random seed), or extend
  the head with an explicit padded-cell mask before using padding.
- Preserve the original cell-to-library identifiers, not a patient-level
  library label; a patient may contribute cells from multiple libraries.
- Build `counts`, `size_factors`, and `library_ids` with identical cell order.
- Training/predictive contexts must contain only observations available at
  inference time; never include held-out queries when updating a contextual
  gate. `predictive_loss(query, latent, context)` supports this split.

A typical application can use `adata.raw.X` *only after* verifying that it
still contains raw integer counts. `adata.raw` is **not guaranteed** to contain
counts in every AnnData project. `np.expm1(adata.X)` is not a count recovery
method once library-size normalization has occurred.

## Initialization and training recipe

1. Construct train-only input tensors and clinically grouped folds.
2. Build head with `inflation="gene_component"` and
   `initial_inflation="none"`, keeping future extensions available.
3. `head.initialize_from_counts(train_view, seed=..., max_init_cells=5000)`.
   This performs a small deterministic (seeded) torch k-means on sampled
   size-factor-normalized `log1p` counts, or uses supplied **train-only** soft
   `assignments` with shape `[N, C, K]`. It sets component rate profiles by
   exposure-weighted count averages, initializes a weak local classifier, and
   initializes gate intercepts using smoothed component proportions. Latent
   gate slopes start at zero. NB dispersion remains at the conservative
   stable initial 10 rather than an unreliable zero-based estimate.
4. Train NB using a validation checkpoint based on **total predictive NLL**.
   The package's `.fit()` is an in-memory training convenience; application
   runners must provide the patient-grouped validation/checkpoint loop.
5. Enable `head.set_inflation_mode("gene")`, lower the learning rate, and
   check *held-out* zero-rate calibration **and positive-count tails**.
6. Only if warranted, enable `gene_component` and compare held-out NLL,
   dispersion/inflation extremes, component occupancy, library association,
   seed stability, and replicate-sample concordance.

The default `.fit_optimized()` applies *global covariance initialization*
to the patient encoder, and may reinitialize the structured gate intercepts.
If preserving the gate initialization from `initialize_from_counts` is
important, pass `initialization="legacy"` or call the count initializer
**after** covariance initialization. Both initialization steps must be
performed on **training patients only**.

The regularization terms are included in training `.loss()`. In contrast,
`.predictive_loss()` returns predictive NLL (plus optional distillation),
**without parameter prior/regularization**: use the latter for comparing
predictive validation performance across configurations. The regularization
strengths and gene normalization should be reported explicitly.

## Core API

```python
head = model.heads[index]
head.initialize_from_counts(train_view, seed=42)
head.set_inflation_mode("none")
head.set_inflation_mode("gene")
head.set_inflation_mode("gene_component")   # when configured

ll = head.component_log_likelihood(view)     # [patients,cells,components]
responsibilities = head.posterior_probs(latent, view)
gates = head.prior_gate_probs(latent)
synthetic = head.sample_counts(latent, size_factors, library_ids)
```

The head implements the same `encode`, `loss`, `predictive_loss`, and
`refined_encoder_input` protocol as existing structured heads. Existing
Gaussian/Topic heads remain unchanged. `methods.factmx_torch` compatibility
imports continue to reference the canonical implementation.

## Model checking

Evaluate held-out predictive NLL (same gene set, same folds), empirical vs
predicted zeros by gene/library/depth, positive-count tail calibration, and
component stability. Inspect inflation saturation near 1, inverse-dispersion
at bounds, component occupancy collapse, and per-library expression offsets.
Very flexible mixture emissions can absorb the actual biological signal and
reduce downstream patient prognosis even if likelihood improves. No real
ACS/CCS cohort training has been performed for this software release.
