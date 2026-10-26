"""Observed-data initializers for FACTMx.

The default optimized-training initializer is a block-balanced global-covariance
warm start.  It uses only observed training data and current view-local encoders;
no labels, simulator truth, validation data, or test data are required.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn


def _as_probability_vector(value: Any, *, device: torch.device) -> Tensor:
    probs = torch.as_tensor(value, dtype=torch.float64, device=device).reshape(-1)
    if probs.numel() == 0 or not torch.isfinite(probs).all() or (probs < 0).any():
        raise ValueError("Population gate probabilities must be finite and non-negative")
    total = probs.sum()
    if not torch.isfinite(total) or float(total) <= 0:
        raise ValueError("Population gate probabilities must have positive sum")
    return probs / total


def _infer_population_gate(encoded: Mapping[str, Tensor], dim: int) -> Tensor | None:
    assignment = encoded.get("assignment")
    if assignment is not None:
        probs = assignment.detach().to(dtype=torch.float64)
        if probs.ndim < 2 or probs.shape[-1] != dim:
            return None
        reduce_dims = tuple(range(probs.ndim - 1))
        probs = probs.mean(dim=reduce_dims)
        return probs / probs.sum().clamp_min(1e-12)

    encoder_input = encoded.get("encoder_input")
    if encoder_input is not None and encoder_input.ndim == 2 and encoder_input.shape[-1] == dim:
        # Structured FACTMx heads encode log-proportions. Softmax is robust to
        # their common additive epsilon/normalization convention.
        probs = torch.softmax(encoder_input.detach().to(dtype=torch.float64), dim=-1).mean(dim=0)
        return probs / probs.sum().clamp_min(1e-12)
    return None


def _ridge_affine(z: Tensor, y: Tensor, alpha: float) -> tuple[Tensor, Tensor]:
    """Return y ~= z @ coef + intercept using ridge on centered variables."""
    if z.ndim != 2 or y.ndim != 2 or z.shape[0] != y.shape[0]:
        raise ValueError("Ridge initialization expects z[N,D] and y[N,P]")
    z = z.to(dtype=torch.float64)
    y = y.to(dtype=torch.float64)
    z_mean = z.mean(dim=0)
    y_mean = y.mean(dim=0)
    zc = z - z_mean
    yc = y - y_mean
    gram = zc.T @ zc
    eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    coef = torch.linalg.solve(gram + float(alpha) * eye, zc.T @ yc)
    intercept = y_mean - z_mean @ coef
    return coef, intercept


def initialize_global_covariance(
    model: nn.Module,
    data: Sequence[Any],
    *,
    population_gate_probs: Mapping[str, Any] | None = None,
    simple_head_names: Sequence[str] | None = None,
    active_head_names: Sequence[str] | None = None,
    ridge_alpha: float = 1e-4,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Initialize FACTMx from train-only global covariance structure.

    The procedure mirrors the initialization validated in the simulation benchmark:

    1. deterministically encode every training view into its head summary;
    2. center each active head block and scale it to unit *total* variance;
    3. take the leading global covariance/PCA directions and whiten their scores;
    4. write that affine transform into the shared linear latent-mean encoder;
    5. moment-match Gaussian/simple linear decoders by ridge regression;
    6. initialize structured latent-to-gate slopes to zero and gate biases to
       population mixture proportions.

    ``population_gate_probs`` is optional.  When omitted for a structured head,
    the population gate is estimated from that head's deterministic local
    assignments on the supplied training data.  For the benchmark, passing the
    independent GMM/Topic mixture weights reproduces the validated initializer.

    Only :class:`~factmx_torch.encoders.LinearEncoder`-style encoders exposing
    ``encoder.loc`` are supported because the covariance map is affine by design.
    """
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha must be non-negative")
    if eps <= 0:
        raise ValueError("eps must be positive")

    heads = list(getattr(model, "heads"))
    if len(data) != len(heads):
        raise ValueError("One training data item is required for each FACTMx head")
    encoder = getattr(model, "encoder")
    if not hasattr(encoder, "loc") or not isinstance(encoder.loc, nn.Linear):
        raise TypeError("Global covariance initialization requires a linear encoder with encoder.loc")

    model.eval()
    encoded_items: list[Mapping[str, Tensor]] = []
    blocks: list[Tensor] = []
    with torch.no_grad():
        for head, data_i in zip(heads, data):
            encoded = head.encode(data_i, deterministic=True)
            block = encoded["encoder_input"].detach()
            if block.ndim != 2:
                raise ValueError(
                    f"Head {head.head_name!r} encoder summary must be [N,D], got {tuple(block.shape)}"
                )
            encoded_items.append(encoded)
            blocks.append(block.to(dtype=torch.float64))

    n_values = {int(block.shape[0]) for block in blocks}
    if len(n_values) != 1 or not n_values or next(iter(n_values)) <= 1:
        raise ValueError("All head summaries must contain the same N>1 training patients")

    all_names = [str(head.head_name) for head in heads]
    if active_head_names is None:
        model_active = getattr(model, "active_encoder_heads", None)
        active = set(all_names if model_active is None else model_active)
    else:
        active = {str(name) for name in active_head_names}
    unknown = active - set(all_names)
    if unknown:
        raise ValueError(f"Unknown active head names: {sorted(unknown)}")
    if not active:
        raise ValueError("At least one active encoder head is required")

    block_means: list[Tensor] = []
    block_scales: list[float] = []
    active_normalized: list[Tensor] = []
    for head, block in zip(heads, blocks):
        mean = block.mean(dim=0)
        centered = block - mean
        scale_tensor = torch.sqrt(centered.square().sum(dim=1).mean()).clamp_min(float(eps))
        scale = float(scale_tensor)
        block_means.append(mean)
        block_scales.append(scale)
        width = int(block.shape[1])
        if str(head.head_name) in active:
            active_normalized.append(centered / scale_tensor)

    balanced = torch.cat(active_normalized, dim=1)
    latent_dim = int(getattr(model, "dim_latent"))
    max_rank = min(int(balanced.shape[0]), int(balanced.shape[1]))
    if latent_dim > max_rank:
        raise ValueError(
            f"latent_dim={latent_dim} exceeds covariance warm-start rank bound {max_rank}"
        )

    _u, singular_values, vh = torch.linalg.svd(balanced, full_matrices=False)
    directions = vh[:latent_dim].T
    scores = balanced @ directions
    score_sd = scores.std(dim=0, unbiased=False).clamp_min(float(eps))
    whitened_directions = directions / score_sd.unsqueeze(0)
    z = scores / score_sd.unsqueeze(0)

    # Expand the active balanced transform back onto the raw concatenated head
    # summaries, leaving inactive encoder blocks exactly zero.
    total_width = sum(int(block.shape[1]) for block in blocks)
    affine = torch.zeros(
        (total_width, latent_dim), dtype=torch.float64, device=balanced.device
    )
    raw_cursor = 0
    active_cursor = 0
    for head, block, scale in zip(heads, blocks, block_scales):
        width = int(block.shape[1])
        if str(head.head_name) in active:
            affine[raw_cursor : raw_cursor + width] = (
                whitened_directions[active_cursor : active_cursor + width] / float(scale)
            )
            active_cursor += width
        raw_cursor += width

    raw_mean = torch.cat(block_means, dim=0)
    bias = -(raw_mean @ affine)
    with torch.no_grad():
        encoder.loc.weight.copy_(affine.T.to(dtype=encoder.loc.weight.dtype, device=encoder.loc.weight.device))
        encoder.loc.bias.copy_(bias.to(dtype=encoder.loc.bias.dtype, device=encoder.loc.bias.device))

    by_name = {str(head.head_name): (head, data_i, enc) for head, data_i, enc in zip(heads, data, encoded_items)}
    if simple_head_names is None:
        simple_names = [
            str(head.head_name)
            for head in heads
            if not bool(getattr(head, "structured", False)) and hasattr(head, "loc")
        ]
    else:
        simple_names = [str(name) for name in simple_head_names]

    for name in simple_names:
        if name not in by_name:
            raise ValueError(f"Unknown simple head name {name!r}")
        head, data_i, _ = by_name[name]
        if not hasattr(head, "loc") or not isinstance(head.loc, nn.Linear):
            raise TypeError(f"Simple head {name!r} does not expose a linear .loc decoder")
        if not isinstance(data_i, Tensor) or data_i.ndim != 2:
            raise TypeError(f"Simple head {name!r} requires tensor data [N,D] for moment matching")
        coef, intercept = _ridge_affine(z, data_i.detach().to(dtype=torch.float64), float(ridge_alpha))
        with torch.no_grad():
            head.loc.weight.copy_(coef.T.to(dtype=head.loc.weight.dtype, device=head.loc.weight.device))
            head.loc.bias.copy_(intercept.to(dtype=head.loc.bias.dtype, device=head.loc.bias.device))

    supplied = dict(population_gate_probs or {})
    gate_sources: dict[str, str] = {}
    for head, encoded in zip(heads, encoded_items):
        if not bool(getattr(head, "structured", False)) or not hasattr(head, "mixture_logits"):
            continue
        name = str(head.head_name)
        if name in supplied:
            probs = _as_probability_vector(supplied[name], device=balanced.device)
            gate_sources[name] = "supplied"
        else:
            inferred = _infer_population_gate(encoded, int(head.dim))
            if inferred is None:
                continue
            probs = inferred.to(device=balanced.device)
            gate_sources[name] = "local_assignment_mean"
        if probs.numel() != int(head.dim):
            raise ValueError(
                f"Population gate for {name!r} has {probs.numel()} entries; expected {head.dim}"
            )
        logits = probs.clamp_min(1e-12).log()
        logits = logits - logits.mean()
        with torch.no_grad():
            head.mixture_logits.weight.zero_()
            head.mixture_logits.bias.copy_(
                logits.to(dtype=head.mixture_logits.bias.dtype, device=head.mixture_logits.bias.device)
            )

    info = {
        "method": "global_covariance",
        "active_head_names": [name for name in all_names if name in active],
        "block_scales": [float(value) for value in block_scales],
        "singular_values": [float(value) for value in singular_values[:latent_dim].detach().cpu()],
        "latent_score_sd": [float(value) for value in score_sd.detach().cpu()],
        "ridge_alpha": float(ridge_alpha),
        "gate_sources": gate_sources,
    }
    setattr(model, "_global_covariance_initialized", True)
    setattr(model, "_initialization_info", info)
    return info
