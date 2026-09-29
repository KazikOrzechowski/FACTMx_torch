"""Shared latent encoders for the PyTorch FACTMx port."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _positive_scale(raw: Tensor, eps: float, parameterization: str) -> Tensor:
    if parameterization == "faithful_relu":
        # Mirrors ConstantResponse(activation="relu") + eps in the archive.
        return F.relu(raw) + eps
    if parameterization == "softplus":
        return F.softplus(raw) + eps
    if parameterization == "exp":
        return torch.exp(raw) + eps
    raise ValueError(f"Unknown scale parameterization: {parameterization}")


class LinearEncoder(nn.Module):
    """Linear diagonal-Normal encoder over concatenated head summaries."""

    encoder_type = "Linear"

    def __init__(
        self,
        dim_latent: int,
        head_dims: Sequence[int],
        eps: float = 1e-5,
        scale_parameterization: str = "faithful_relu",
    ) -> None:
        super().__init__()
        self.dim_latent = dim_latent
        self.head_dims = tuple(head_dims)
        self.eps = eps
        self.scale_parameterization = scale_parameterization
        self.loc = nn.Linear(sum(head_dims), dim_latent)
        nn.init.orthogonal_(self.loc.weight)
        nn.init.zeros_(self.loc.bias)

        if scale_parameterization == "softplus":
            initial = torch.log(torch.expm1(torch.tensor(0.1)))
        elif scale_parameterization == "exp":
            initial = torch.log(torch.tensor(eps))
        else:
            initial = torch.tensor(eps)
        self.raw_scale = nn.Parameter(initial.repeat(dim_latent))

    def forward(self, data: Tensor) -> tuple[Tensor, Tensor]:
        loc = self.loc(data)
        scale = _positive_scale(
            self.raw_scale, self.eps, self.scale_parameterization
        ).expand_as(loc)
        return loc, scale


class MeanEncoder(nn.Module):
    """Mean-pooling encoder for equal-sized head summaries."""

    encoder_type = "Mean"

    def __init__(self, dim_latent: int, head_dims: Sequence[int], eps: float = 1e-5):
        super().__init__()
        if not head_dims or any(d != dim_latent for d in head_dims):
            raise ValueError("MeanEncoder requires every head dimension to equal dim_latent")
        self.dim_latent = dim_latent
        self.head_dims = tuple(head_dims)
        self.eps = eps
        self.log_scale = nn.Parameter(torch.zeros(dim_latent))

    def forward(self, data: Tensor) -> tuple[Tensor, Tensor]:
        broad = data.reshape(data.shape[0], len(self.head_dims), self.dim_latent)
        loc = broad.mean(dim=1)
        scale = (torch.exp(self.log_scale) + self.eps).expand_as(loc)
        return loc, scale


class AttentionEncoder(nn.Module):
    """Attention aggregation for equal-sized head summaries."""

    encoder_type = "Attention"

    def __init__(self, dim_latent: int, head_dims: Sequence[int], eps: float = 1e-5):
        super().__init__()
        if not head_dims or any(d != dim_latent for d in head_dims):
            raise ValueError(
                "AttentionEncoder requires every head dimension to equal dim_latent"
            )
        self.dim_latent = dim_latent
        self.head_dims = tuple(head_dims)
        self.eps = eps
        self.query = nn.Linear(dim_latent, dim_latent, bias=False)

    def forward(self, data: Tensor) -> tuple[Tensor, Tensor]:
        values = data.reshape(data.shape[0], len(self.head_dims), self.dim_latent)
        queries = self.query(values)
        scores = torch.matmul(queries, values.transpose(-1, -2))
        scores = scores / self.dim_latent**0.5
        attended = torch.matmul(torch.softmax(scores, dim=-1), values)
        loc = attended.mean(dim=1)
        scale = attended.var(dim=1, unbiased=False).clamp_min(self.eps)
        return loc, scale


def build_encoder(
    encoder_type: str,
    dim_latent: int,
    head_dims: Sequence[int],
    **kwargs,
) -> nn.Module:
    mapping = {
        "Linear": LinearEncoder,
        "Attention": AttentionEncoder,
        "Mean": MeanEncoder,
    }
    try:
        cls = mapping[encoder_type]
    except KeyError as exc:
        raise ValueError(f"Unknown encoder type: {encoder_type}") from exc
    return cls(dim_latent=dim_latent, head_dims=head_dims, **kwargs)
