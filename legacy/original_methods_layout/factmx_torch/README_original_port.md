# FACTMx PyTorch port and clustering diagnostic

This package is a runnable PyTorch translation of the supplied TensorFlow/
TensorFlow Probability FACTMx archive. It contains linear, mean, and attention
latent encoders plus Bernoulli, multinomial, multivariate-Normal, Gaussian-
mixture, and topic-model heads.

The defaults of the structured heads reproduce the archived one-pass
variational objective. Two opt-in changes support the proposed repair:

- `normalize_profiles=True` constrains every topic profile to a probability
  simplex.
- `objective="marginal"` analytically sums over the categorical assignment
  instead of sampling a low-temperature Concrete/Gumbel variable.
- `refinement_steps=1` recomputes structured proportions from
  `p(component | subobservation, patient latent)` and re-encodes the patient.
- `assignment_estimator="mean"` uses expected assignments in encoder summaries,
  removing unnecessary Monte Carlo noise.

The notebook's `GMM_prop` configuration name is accepted as an alias for the
archive's `GMM` head. This compatibility alias is necessary because `GMM_prop`
does not exist in the supplied source archive.

## Run

```bash
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD"
python examples/clustering_improvement.py \
  --seeds 3 --epochs 400 --conditional-steps 300 \
  --output-dir diagnostics
python -m unittest discover -s tests -v
```

The experiment reproduces notebook cells 141--165 using PyTorch, evaluates
assignment accuracy and adjusted Rand index, and writes per-seed and summary
tables plus a diagnostic plot. See `DIAGNOSIS.md` for interpretation.

## Minimal API

```python
from factmx_torch import FACTMxModel

model = FACTMxModel(
    dim_latent=1,
    heads_config=[
        {"head_type": "MultiNormal", "head_name": "rna", "dim": 1},
        {
            "head_type": "GMM",
            "head_name": "gmm",
            "dim": 2,
            "dim_normal": 1,
            "assignment_estimator": "mean",
            "objective": "marginal",
            "covariance_parameterization": "softplus",
        },
    ],
    encoder_config={
        "encoder_type": "Linear",
        "scale_parameterization": "softplus",
    },
    refinement_steps=1,
)

history = model.fit(data, epochs=400, batch_size=200, lr=1e-3)
latent = model.get_latent_representation(data)
```

The port does not deserialize arbitrary Keras `Sequential` configurations;
the default linear maps used by the supplied notebook are translated directly.
