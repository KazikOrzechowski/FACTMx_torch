from __future__ import annotations

from pathlib import Path

import torch

from factmx_torch import MissingViewFACTMx


def _model(*, availability_to_scale: bool = True) -> MissingViewFACTMx:
    return MissingViewFACTMx(
        dim_latent=2,
        heads_config=[
            {"head_type": "MultiNormal", "head_name": "a", "dim": 2},
            {"head_type": "MultiNormal", "head_name": "b", "dim": 3},
        ],
        encoder_config={"encoder_type": "Linear", "scale_parameterization": "softplus"},
        availability_to_scale=availability_to_scale,
    )


def test_masking_and_scale_only_missingness() -> None:
    torch.manual_seed(4)
    model = _model()
    data = [torch.randn(5, 2), torch.randn(5, 3)]
    centers = torch.cat([data[0].mean(0), data[1].mean(0)])
    model.set_encoder_centers(centers)

    full_mask = torch.ones(5, 2, dtype=torch.bool)
    missing_b = full_mask.clone()
    missing_b[:, 1] = False

    _, full_loc, full_scale, encoded = model.infer_masked(data, full_mask, deterministic=True)
    _, missing_loc, missing_scale, _ = model.infer_masked(data, missing_b, deterministic=True)

    assert full_loc.shape == missing_loc.shape == (5, 2)
    assert full_scale.shape == missing_scale.shape == (5, 2)
    assert torch.all(missing_scale >= full_scale)

    loc2, scale2 = model.posterior_from_encoded(encoded, missing_b)
    torch.testing.assert_close(loc2, missing_loc)
    torch.testing.assert_close(scale2, missing_scale)


def test_missing_view_checkpoint_round_trip(tmp_path: Path) -> None:
    model = _model(availability_to_scale=True)
    model.set_encoder_centers(torch.arange(5, dtype=torch.float32))
    path = tmp_path / "missing.pt"
    model.save_checkpoint(str(path))
    restored = MissingViewFACTMx.load_checkpoint(str(path))

    assert restored.availability_to_scale is True
    assert restored.neutral_centering is True
    torch.testing.assert_close(restored.encoder_center_initial, model.encoder_center_initial)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)
