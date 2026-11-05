# Research patch: 0.1.4.dev1+gradnorm.1

- Added package-level `MolecularGradNorm` with Chen-style gradient matching and inverse relative training rates; keeps network and weight optimizers separate.
- Added `--simple-balancing original|gradnorm` to the supplied v0.1.4.dev1 TCGA CNA selector runner; optional `alpha`, weight LR, update interval, positive rate mode, minimum observed count.
- Outer simple/H&E controller, structured-first curriculum, masked heads, original multinomial decoder, training-only categorical frequencies and validation-NLL checkpoint criterion are unchanged.
- Weighted simple-group gradients and learned weights are now logged; gradient convergence metrics correctly apply both the outer multiplier and GradNorm task weight.
- Best-checkpoint GradNorm state and optimizer stored alongside existing checkpoint data, without changing `model.state_dict()` key structure.
- Added H&E-only-batch safety, nonpositive Gaussian NLL guard and optional noncanonical positive reconstruction progress proxy.
- Added targeted integrated real-head tests for baseline log-onehot, centered-onehot and frequency-standardized CNA, plus original selector tests and synthetic trainer tests.
