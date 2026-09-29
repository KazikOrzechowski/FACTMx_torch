# Contributing

Contributions that improve correctness, documentation, testing, interoperability, or generic FACTMx functionality are welcome.

## Development setup

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

The supported Python range is 3.10–3.12. Keep the mandatory dependency set small; analysis, plotting, cohort-processing, and benchmark dependencies should normally remain outside the core package.

## Scientific-change policy

Packaging/refactoring changes should preserve established model behavior. Changes to objectives, posterior semantics, defaults, masking rules, or training schedules are scientific changes and should be isolated, documented, and covered by targeted regression tests rather than folded into unrelated refactors.

For structured held-out prediction, preserve the context/query separation: query observations must not enter the latent encoder or contextual gate before their predictive likelihood is evaluated.

## Pull requests

Please include:

- a concise description of the change;
- tests for new behavior or bug fixes;
- documentation for public API changes;
- a note identifying any checkpoint/API compatibility impact;
- a note identifying whether the change is purely software-level or changes the statistical method.
