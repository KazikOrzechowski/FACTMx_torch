from __future__ import annotations

import torch

from factmx_torch import FACTMxModel, initialize_global_covariance


def _model() -> FACTMxModel:
    return FACTMxModel(
        dim_latent=2,
        heads_config=[
            {"head_type": "MultiNormal", "head_name": "simple1", "dim": 3},
            {"head_type": "MultiNormal", "head_name": "simple2", "dim": 2},
            {
                "head_type": "GMM",
                "head_name": "structured",
                "dim": 2,
                "dim_normal": 2,
                "assignment_estimator": "mean",
                "objective": "marginal",
                "covariance_parameterization": "softplus",
            },
        ],
        encoder_config={"encoder_type": "Linear", "scale_parameterization": "softplus"},
    )


def _data(n: int = 24) -> list[torch.Tensor]:
    torch.manual_seed(17)
    z = torch.randn(n, 2)
    simple1 = torch.cat([z, z[:, :1] - 0.4 * z[:, 1:2]], dim=1) + 0.03 * torch.randn(n, 3)
    simple2 = torch.stack([z[:, 0] + z[:, 1], z[:, 0] - z[:, 1]], dim=1)
    structured = torch.stack(
        [
            torch.stack([z[:, 0], z[:, 1]], dim=-1),
            torch.stack([z[:, 0] + 0.2, z[:, 1] - 0.1], dim=-1),
            torch.stack([z[:, 0] - 0.1, z[:, 1] + 0.2], dim=-1),
        ],
        dim=1,
    )
    return [simple1, simple2, structured]


def test_global_covariance_initializer_uses_supplied_gate_probs() -> None:
    model = _model()
    data = _data()
    info = initialize_global_covariance(
        model,
        data,
        population_gate_probs={"structured": [0.75, 0.25]},
    )

    assert info["method"] == "global_covariance"
    assert info["gate_sources"]["structured"] == "supplied"
    gate = model.heads[2].mixture_logits
    assert torch.count_nonzero(gate.weight) == 0

    expected = torch.log(torch.tensor([0.75, 0.25]))
    expected = expected - expected.mean()
    assert torch.allclose(gate.bias.detach(), expected, atol=1e-6)

    with torch.no_grad():
        latent = model.get_latent_representation(data)
    assert torch.allclose(latent.mean(dim=0), torch.zeros(2), atol=2e-5)
    assert torch.allclose(
        latent.std(dim=0, unbiased=False), torch.ones(2), atol=2e-5
    )


def test_global_covariance_initializer_falls_back_to_local_assignments() -> None:
    model = _model()
    info = model.initialize_global_covariance(_data())
    assert info["gate_sources"]["structured"] == "local_assignment_mean"
    assert torch.count_nonzero(model.heads[2].mixture_logits.weight) == 0


def test_fit_optimized_uses_global_covariance_by_default_and_legacy_opt_out() -> None:
    data = _data()

    default_model = _model()
    result = default_model.fit_optimized(
        data,
        epochs=1,
        batch_size=12,
        lr=1e-3,
        structured_hold_epochs=0,
        simple_ramp_epochs=0,
        kl_warmup_epochs=1,
        population_gate_probs={"structured": [0.6, 0.4]},
    )
    assert result["initialization"]["method"] == "global_covariance"
    assert result["initialization"]["gate_sources"]["structured"] == "supplied"

    legacy_model = _model()
    result_legacy = legacy_model.fit_optimized(
        data,
        epochs=1,
        batch_size=12,
        lr=1e-3,
        structured_hold_epochs=0,
        simple_ramp_epochs=0,
        kl_warmup_epochs=1,
        initialization="legacy",
    )
    assert result_legacy["initialization"]["method"] == "legacy"


def test_checkpoint_roundtrip_preserves_global_initialization_state(tmp_path) -> None:
    model = _model()
    model.initialize_global_covariance(
        _data(), population_gate_probs={"structured": [0.55, 0.45]}
    )
    path = tmp_path / "model.pt"
    model.save_checkpoint(str(path))
    restored = FACTMxModel.load_checkpoint(str(path))
    assert restored._global_covariance_initialized is True
    assert restored._initialization_info["method"] == "global_covariance"
    assert restored._initialization_info["gate_sources"]["structured"] == "supplied"
