from __future__ import annotations

from pathlib import Path

import torch

from factmx_torch import FACTMxModel


def test_checkpoint_round_trip_and_inference(tmp_path: Path) -> None:
    torch.manual_seed(5)
    model = FACTMxModel(
        dim_latent=2,
        heads_config=[{"head_type": "MultiNormal", "head_name": "x", "dim": 3}],
        encoder_config={"encoder_type": "Linear", "scale_parameterization": "softplus"},
        name="roundtrip",
    )
    data = [torch.randn(6, 3)]
    expected = model.get_latent_representation(data)
    path = tmp_path / "model.pt"
    model.save_checkpoint(str(path))
    restored = FACTMxModel.load_checkpoint(str(path))
    actual = restored.get_latent_representation(data)
    torch.testing.assert_close(actual, expected)
    assert restored.name == "roundtrip"
