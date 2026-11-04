# Global-covariance initialization

The recommended/default optimized FACTMx initialization is now a train-only,
block-balanced global-covariance warm start.

For a fitted set of view-local encoders/components, the initializer:

1. computes deterministic per-patient encoder summaries for every active view;
2. centers each active view block and scales it to unit total variance;
3. computes the leading global covariance directions and whitens their scores;
4. writes the resulting affine transform into the shared linear latent-mean encoder;
5. initializes linear Gaussian/simple decoders by train-only ridge regression from
   the initialized latent scores;
6. initializes structured gate slopes and biases using independently configurable
   policies (by default, zero slopes and population mixture-logit biases).

No labels, simulator truth, validation observations, or test observations enter
this initializer.  A zero latent-to-gate slope does **not** freeze the gate map:
the gate weights receive non-zero gradients immediately because the initialized
latent scores vary across patients.  The only transient effect is that the first
structured-loss backward pass cannot yet propagate through a zero gate slope into
the shared encoder; after the first optimizer update that route is active.

## API

```python
info = model.initialize_global_covariance(
    training_data,
    population_gate_probs={
        "gmm": gmm_weights,
        "topic1": topic1_weights,
        "topic2": topic2_weights,
    },
)
```

`FACTMxModel.fit_optimized()` uses this initializer by default.  To preserve an
already prepared model or reproduce the historical random shared-encoder start,
pass `initialization="legacy"` (or `None`).

When explicit population gate probabilities are omitted, structured population
proportions are estimated from the current deterministic local assignments on the
supplied training observations.  In the simulation benchmark, the independently
fitted GMM/Topic mixture weights are passed explicitly because that is the
validated warm-start configuration.

## Independent gate initialization policies (v0.1.3)

```python
info = model.initialize_global_covariance(
    training_data,
    population_gate_probs={"gmm": gmm_weights},
    gate_weight_mode="preserve_random",  # or "zero"
    gate_bias_mode="zero",              # or "population"
)
```

The same two parameters can be passed to `fit_optimized()` when it runs the
complete-data covariance initializer. `preserve_random` does *not* sample new
weights; the caller should initialize the gate weights to the desired small
random distribution before invoking the initializer. For masked data, use
`initialize_masked_global_covariance`; see `masked_global_initialization.md`.
