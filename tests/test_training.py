from __future__ import annotations

import math

import torch

from factmx_torch import (
    FACTMxModel,
    EncoderGradientBalancer,
    StructuredFirstBalanceConfig,
    kl_beta_for_epoch,
    simple_ramp_multiplier,
)


def test_training_schedules_and_matched_ema() -> None:
    assert simple_ramp_multiplier(0, 2, 2) == 0.0
    assert simple_ramp_multiplier(2, 2, 2) == 0.5
    assert simple_ramp_multiplier(3, 2, 2) == 1.0
    assert math.isclose(kl_beta_for_epoch(0, 0.3, 3), 0.1)

    cfg = StructuredFirstBalanceConfig(ema_beta_per_batch=0.9)
    balancer = EncoderGradientBalancer(cfg, n_batches_per_epoch=3)
    assert math.isclose(balancer.ema_decay, 0.9**3)


def test_optimized_training_smoke() -> None:
    torch.manual_seed(8)
    n = 10
    data = [torch.randn(n, 2), torch.randn(n, 4, 2)]
    model = FACTMxModel(
        dim_latent=2,
        heads_config=[
            {"head_type": "MultiNormal", "head_name": "simple", "dim": 2},
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
    result = model.fit_optimized(
        data,
        epochs=2,
        batch_size=5,
        lr=1e-3,
        structured_hold_epochs=0,
        simple_ramp_epochs=1,
        kl_warmup_epochs=1,
        max_grad_norm=5.0,
    )
    assert len(result["history"]) == 2
    assert result["balance_updates"] == 2
    assert all(math.isfinite(row["train_objective"]) for row in result["history"])
