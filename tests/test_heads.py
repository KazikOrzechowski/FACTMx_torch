from __future__ import annotations

import torch

from factmx_torch import GMMHead, TopicModelHead, build_head


def test_gmm_prop_historical_alias() -> None:
    head = build_head(
        {
            "head_type": "GMM_prop",
            "head_name": "gmm",
            "dim": 3,
            "dim_normal": 2,
            "assignment_estimator": "mean",
            "objective": "marginal",
            "covariance_parameterization": "softplus",
        },
        dim_latent=2,
    )
    assert isinstance(head, GMMHead)
    x = torch.randn(4, 5, 2)
    encoded = head.encode(x, deterministic=True)
    assert encoded["assignment"].shape == (4, 5, 3)
    assert encoded["encoder_input"].shape == (4, 3)


def test_contextual_gate_and_topic_profiles() -> None:
    head = GMMHead(
        dim=3,
        dim_latent=2,
        dim_normal=2,
        head_name="gmm",
        objective="marginal",
        assignment_estimator="mean",
        covariance_parameterization="softplus",
        contextual_gate=True,
        contextual_encoder="deepsets",
    )
    z = torch.randn(4, 2)
    context = torch.randn(4, 6, 2)
    probs = head.contextual_gate_probs(z, context)
    assert probs.shape == (4, 3)
    torch.testing.assert_close(probs.sum(-1), torch.ones(4), atol=1e-5, rtol=1e-5)

    topic = TopicModelHead(
        dim=2,
        dim_latent=2,
        dim_words=4,
        head_name="topic",
        normalize_profiles=True,
        objective="marginal",
        assignment_estimator="mean",
    )
    profiles = topic.topic_profiles()
    torch.testing.assert_close(profiles.sum(0), torch.ones(2), atol=1e-5, rtol=1e-5)
