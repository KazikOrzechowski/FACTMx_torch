from __future__ import annotations

import torch

from factmx_torch import FACTMxModel


def _model() -> FACTMxModel:
    return FACTMxModel(
        dim_latent=2,
        heads_config=[
            {"head_type": "MultiNormal", "head_name": "x", "dim": 3},
            {"head_type": "Bernoulli", "head_name": "b", "dim": 2},
        ],
        encoder_config={"encoder_type": "Linear", "scale_parameterization": "softplus"},
    )


def _data(n: int = 8) -> list[torch.Tensor]:
    torch.manual_seed(1)
    return [torch.randn(n, 3), torch.randint(0, 2, (n, 2)).float()]


def test_forward_shapes_and_loss() -> None:
    model = _model()
    data = _data()
    latent, loc, scale, encoded = model.infer(data, deterministic=True)
    assert latent.shape == loc.shape == scale.shape == (8, 2)
    assert len(encoded) == 2
    assert torch.isfinite(model.loss(data, deterministic_latent=True))
    decoded = model.full_pass(data, deterministic=True)
    assert decoded[0].shape == (8, 3)
    assert decoded[1].shape == (8, 2)


def test_fit_smoke() -> None:
    model = _model()
    history = model.fit(_data(), epochs=1, batch_size=4, lr=1e-3)
    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(history)))
