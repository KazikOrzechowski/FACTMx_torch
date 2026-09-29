# Optimized FACTMx training

This package contains the current PyTorch FACTMx model plus the structured-first,
encoder-gradient-balanced training path validated on the development simulation.
The historical `FACTMxModel.fit()` method is retained unchanged.  The new
same-data API is `FACTMxModel.fit_optimized()`, while the simulation benchmark's
context/query objective uses the same reusable balancing implementation through
`simulation_benchmark/src/factmx_adapter.py`.

## Training schedule

The validated development configuration is:

- 45 structured-first epochs: simple-view reconstruction multiplier = 0;
- 45-epoch linear reintroduction of the simple-view objective;
- KL beta linearly warmed to its configured final value;
- once-per-epoch balancing of scaled simple vs structured encoder-gradient norms;
- per-batch reference EMA beta 0.9, converted to the equivalent update-frequency
  decay as `0.9 ** (# batches between balance updates)`;
- adaptive simple multiplier clipped to [0.02, 1.0];
- target weighted simple/structured encoder-gradient norm ratio = 1;
- validation checkpointing begins only after the hold+ramp schedule is complete.

The balance update uses only the two observed-data loss groups.  It does not use
simulation labels, true latent factors, true assignments, true component
parameters, or any post-fit recovery metric.

## Low-overhead implementation

Gradient norms are evaluated only on the first minibatch once per epoch by
default.  The simple and structured gradients are each computed once and reused
for the norm calculation; gradient-cosine diagnostics are not computed in the
production path.  The resulting multiplier is reused for the remaining
minibatches in the epoch.

For the development split (180 training patients, batch size 64), once-per-epoch
updates give an EMA decay of `0.9 ** 3 = 0.729` and 115 balance updates over the
post-hold portion of a 160-epoch run.

## Regression checks

`tests/test_training.py` checks the ramp schedule, matched EMA decay, non-mutating
gradient-norm calculation, and a known 4:1 gradient-ratio case.

The integrated benchmark adapter was also rerun on the exact stored development
datasets.  It reproduced the previously validated once-per-epoch matched-EMA
results for seed 1101:

- complementary_context mean structured-view ARI: 0.7349559288254356,
  best epoch 148, mean effective simple multiplier 0.6735912277793276;
- joint_critical mean structured-view ARI: 0.5441607982758775,
  best epoch 151, mean effective simple multiplier 0.6267679024435074.

These checks validate implementation agreement with the tested optimization
prototype; they do not imply that the 45/45 schedule is universally optimal for
all datasets or cohort sizes.

## View-local contextual structured gates

The optimized benchmark additionally uses a differentiable view-local posterior
for structured mixture proportions.  The shared latent decoder supplies a prior
gate, while context observations in each structured view contribute soft
component counts.  Held-out query observations never enter this local update.
See `CONTEXTUAL_GATE.md` for the equations and configuration.
