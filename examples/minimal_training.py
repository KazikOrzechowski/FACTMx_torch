"""Minimal synthetic end-to-end FACTMx training example."""

from __future__ import annotations

import torch

from factmx_torch import FACTMxModel


def main() -> None:
    torch.manual_seed(7)
    n = 24
    data = [
        torch.randn(n, 3),
        torch.randn(n, 5, 2),
    ]

    model = FACTMxModel(
        dim_latent=2,
        heads_config=[
            {"head_type": "MultiNormal", "head_name": "continuous", "dim": 3},
            {
                "head_type": "GMM",
                "head_name": "structured",
                "dim": 3,
                "dim_normal": 2,
                "assignment_estimator": "mean",
                "objective": "marginal",
                "covariance_parameterization": "softplus",
                "contextual_gate": True,
                "contextual_prior_strength": 4.0,
                "contextual_steps": 2,
            },
        ],
        encoder_config={
            "encoder_type": "Linear",
            "scale_parameterization": "softplus",
        },
        refinement_steps=1,
    )

    history = model.fit(data, epochs=2, batch_size=8, lr=1e-3)
    latent = model.get_latent_representation(data)
    print(f"optimization steps: {len(history)}")
    print(f"latent shape: {tuple(latent.shape)}")


if __name__ == "__main__":
    main()
