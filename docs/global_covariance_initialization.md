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
6. initializes each structured latent-to-gate slope to zero and its bias to the
   population mixture proportions.

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
