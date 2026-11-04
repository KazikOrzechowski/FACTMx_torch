"""Train-only, mask-aware adaptation of FACTMx v0.1.1 global covariance initialization.

The upstream ``initialize_global_covariance(model, data)`` assumes complete,
uncentered observations; it is not directly suitable for MissingViewFACTMx.
This adapter preserves its balanced SVD, simple-view ridge warm start and
configurable gate-weight/bias initialization, with three necessary corrections:

* Use observed patient/head summaries, imputing missing summaries with their
  train-only neutral means only for covariance estimation.
* Account for the encoder's subtraction of neutral centers in its bias.
* Fit simple Gaussian decoders using observed training targets only.

Weak singular directions are left at zero (rather than amplified by whitening).
No validation/test observations enter any calculation here.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


def seed_before_model(seed: int) -> None:
    import random

    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def initialize_masked_global_covariance(
    model: nn.Module,
    summaries: Sequence[np.ndarray],
    availability: np.ndarray,
    *,
    simple_targets: Mapping[str, np.ndarray] | None = None,
    population_gate_probs: Mapping[str, np.ndarray] | None = None,
    ridge_alpha: float = 1e-4,
    min_relative_singular: float = 1e-5,
    max_whitening_gain: float = 100.0,
    gate_weight_mode: str = "zero",
    gate_bias_mode: str = "population",
) -> dict[str, Any]:
    """Warm-start a v0.1.1 ``MissingViewFACTMx`` from TRAIN-ONLY summaries.

    ``summaries[j]`` has shape (N, head_dim_j); missing entries are ignored.
    ``availability`` is (N, n_heads); rows correspond to training patients only.
    ``simple_targets`` maps Gaussian head names to raw training (N, dim) arrays.
    ``gate_weight_mode`` controls zeroed vs already-drawn small random slopes.
    ``gate_bias_mode`` selects population log-priors or exactly zero logits.
    The output is JSON serializable and contains rank diagnostics and gate priors.
    """
    heads = list(model.heads)
    names = [str(h.head_name) for h in heads]
    if len(summaries) != len(heads):
        raise ValueError('Head count and summaries do not match')
    mask = np.asarray(availability, dtype=bool)
    if mask.ndim != 2 or mask.shape[1] != len(heads) or mask.shape[0] < 2:
        raise ValueError('availability must be [N,n_heads] with N>=2')
    if ridge_alpha <= 0 or min_relative_singular < 0 or max_whitening_gain <= 0:
        raise ValueError('Invalid ridge/whitening parameters')
    if gate_weight_mode not in {'zero', 'preserve_random'}:
        raise ValueError(f'Unsupported gate_weight_mode: {gate_weight_mode!r}')
    if gate_bias_mode not in {'population', 'zero'}:
        raise ValueError(f'Unsupported gate_bias_mode: {gate_bias_mode!r}')
    if not isinstance(model.encoder.loc, nn.Linear):
        raise TypeError('Global initializer requires a linear latent encoder')

    # Fit all numerical transforms on CPU in float64 for stability.
    n = int(mask.shape[0])
    blocks: list[Tensor] = []
    means: list[Tensor] = []
    scales: list[float] = []
    head_counts: dict[str, int] = {}
    for j, (name, head, source) in enumerate(zip(names, heads, summaries)):
        value = torch.as_tensor(np.asarray(source), dtype=torch.float64, device='cpu')
        if tuple(value.shape) != (n, int(head.dim)) and value.shape != (n, int(model.head_dims[j])):
            raise ValueError(f'{name}: unexpected encoder summary shape {tuple(value.shape)}, expected [N,{model.head_dims[j]}]')
        if not bool(mask[:, j].any()):
            raise ValueError(f'No observed training patients for {name}')
        observed = value[torch.as_tensor(mask[:, j])]
        if not bool(torch.isfinite(observed).all()):
            raise ValueError(f'{name}: nonfinite observed training summaries')
        observed_mean = observed.mean(dim=0)
        filled = value.clone()
        filled[~torch.as_tensor(mask[:, j])] = observed_mean
        block_mean = filled.mean(dim=0)
        centered = filled - block_mean
        scale = max(float(torch.sqrt(centered.square().sum(dim=1).mean())), 1e-8)
        blocks.append(filled)
        means.append(block_mean)
        scales.append(scale)
        head_counts[name] = int(mask[:, j].sum())

    active_head_names = getattr(model, 'active_encoder_heads', None)
    active = set(names if active_head_names is None else active_head_names)
    if not active or not active.issubset(set(names)):
        raise ValueError('Active heads must be a nonempty subset of supplied head names')
    balanced = torch.cat([(b - m) / s for b, m, s, name in zip(blocks, means, scales, names) if name in active], dim=1)
    k = int(model.dim_latent)
    if k > min(balanced.shape):
        raise ValueError(f'latent_dim={k} exceeds min(N, active summary width)={min(balanced.shape)}')
    _, sv, vh = torch.linalg.svd(balanced, full_matrices=False)
    if not bool(torch.isfinite(sv).all()):
        raise FloatingPointError('Nonfinite global covariance singular spectrum')
    threshold = max(1e-12, float(sv[0]) * float(min_relative_singular))
    rank = min(k, int((sv >= threshold).sum()))
    directions = vh[:rank].T
    scores = balanced @ directions
    score_sd = scores.std(dim=0, unbiased=False) if rank else torch.empty(0, dtype=torch.float64)
    whiten = 1.0 / score_sd.clamp_min(1.0 / float(max_whitening_gain))
    z = torch.zeros((n, k), dtype=torch.float64)
    if rank:
        z[:, :rank] = scores * whiten

    # Impute missing heads at their observed means; in masked inference missing
    # inputs contribute zero after centering.  Fix the upstream intercept to
    # account for the neutral-center subtraction.
    width = sum(model.head_dims)
    affine = torch.zeros((width, k), dtype=torch.float64)
    cursor = 0
    active_cursor = 0
    for name, dim, scale in zip(names, model.head_dims, scales):
        if name in active and rank:
            affine[cursor:cursor + dim, :rank] = (directions[active_cursor:active_cursor + dim] * whiten) / scale
        if name in active:
            active_cursor += dim
        cursor += dim
    raw_mean = torch.cat(means)
    if bool(getattr(model, 'neutral_centering', False)):
        center = model.encoder_center_initial.detach().cpu().to(dtype=torch.float64)
        if center.shape != raw_mean.shape:
            raise ValueError('Neutral center dimension mismatch')
        bias = -((raw_mean - center) @ affine)
    else:
        bias = -(raw_mean @ affine)

    # Validate all model edits before mutating parameters.
    targets = dict(simple_targets or {})
    ridge_updates: dict[str, tuple[Tensor, Tensor]] = {}
    for j, head in enumerate(heads):
        name = names[j]
        if name not in targets:
            continue
        if bool(getattr(head, 'structured', False)) or not isinstance(getattr(head, 'loc', None), nn.Linear):
            raise ValueError(f'{name}: simple ridge only supported for Gaussian linear decoder')
        y = torch.as_tensor(np.asarray(targets[name]), dtype=torch.float64)
        if y.shape != (n, int(head.loc.out_features)):
            raise ValueError(f'{name}: target shape {tuple(y.shape)} does not match train cohort')
        present = torch.as_tensor(mask[:, j])
        if int(present.sum()) < 2 or not bool(torch.isfinite(y[present]).all()):
            raise ValueError(f'{name}: insufficient observed, finite training targets')
        zo, yo = z[present], y[present]
        zm, ym = zo.mean(dim=0), yo.mean(dim=0)
        zc, yc = zo - zm, yo - ym
        gram = zc.T @ zc
        coeff = torch.linalg.solve(gram + ridge_alpha * torch.eye(k, dtype=torch.float64), zc.T @ yc)
        ridge_updates[name] = (coeff.T, ym - zm @ coeff)

    gate_updates: dict[str, Tensor] = {}
    gate_weight_diagnostics: dict[str, dict[str, float]] = {}
    supplied = dict(population_gate_probs or {})
    for head in heads:
        name = str(head.head_name)
        if not bool(getattr(head, 'structured', False)) or not hasattr(head, 'mixture_logits'):
            continue
        if name not in supplied:
            raise ValueError(f'Missing population mixture weights for structured head {name}; pass training-only independent-fit weights')
        probs = torch.as_tensor(np.asarray(supplied[name]), dtype=torch.float64).reshape(-1)
        if probs.shape != (int(head.dim),) or not bool(torch.isfinite(probs).all()) or bool((probs < 0).any()) or float(probs.sum()) <= 0:
            raise ValueError(f'Invalid independent population gate for {name}')
        probs = probs / probs.sum()
        logits = probs.clamp_min(1e-12).log()
        gate_updates[name] = logits - logits.mean()
        # initialize_model() has already seeded and drawn small random gate
        # weights (std=0.02). Retaining those tensors consumes no additional
        # RNG state and isolates the effect of zero vs nonzero gate slopes.
        existing_weights = head.mixture_logits.weight.detach()
        if not bool(torch.isfinite(existing_weights).all()):
            raise ValueError(f'{name}: existing gate weights are nonfinite')
        gate_weight_diagnostics[name] = {
            'std_before': float(existing_weights.float().std(unbiased=False).cpu()),
            'max_abs_before': float(existing_weights.abs().max().cpu()),
        }

    if not bool(torch.isfinite(affine).all()) or not bool(torch.isfinite(bias).all()):
        raise FloatingPointError('Global initializer produced nonfinite encoder weights')
    was_training = model.training
    with torch.no_grad():
        model.encoder.loc.weight.copy_(affine.T.to(model.encoder.loc.weight))
        model.encoder.loc.bias.copy_(bias.to(model.encoder.loc.bias))
        for head in heads:
            name = str(head.head_name)
            if name in ridge_updates:
                w, b = ridge_updates[name]
                head.loc.weight.copy_(w.to(head.loc.weight))
                head.loc.bias.copy_(b.to(head.loc.bias))
            if name in gate_updates:
                if gate_weight_mode == 'zero':
                    head.mixture_logits.weight.zero_()
                # preserve_random: keep exactly the weights initialized by
                # the legacy runner (normal with std=0.02). Bias policy is
                # independent: full global and random-gates use population
                # prior biases; global_legacy_gates uses zero (legacy) bias.
                if gate_bias_mode == 'zero':
                    head.mixture_logits.bias.zero_()
                else:
                    head.mixture_logits.bias.copy_(gate_updates[name].to(head.mixture_logits.bias))
    model.train(was_training)

    info = {
        'method': 'global_covariance_mask_aware_v0.1.1',
        'n_training_patients': n,
        'n_observed_by_head': head_counts,
        'active_head_names': [name for name in names if name in active],
        'block_scales': dict(zip(names, scales)),
        'singular_values': sv[:k].tolist(),
        'effective_initialized_rank': rank,
        'dropped_weak_directions': k - rank,
        'minimum_relative_singular': float(min_relative_singular),
        'max_whitening_gain': float(max_whitening_gain),
        'latent_score_sd': score_sd.tolist(),
        'ridge_alpha': float(ridge_alpha),
        'simple_ridge_heads': list(ridge_updates),
        'gate_sources': {
            name: 'uniform_zero_legacy' if gate_bias_mode == 'zero' else 'independent_train_fit_weights'
            for name in gate_updates
        },
        'gate_weight_mode': gate_weight_mode,
        'gate_bias_mode': gate_bias_mode,
        'gate_bias_diagnostics': {
            name: {
                'max_abs_after': float(next(h for h in heads if h.head_name == name).mixture_logits.bias.detach().abs().max().cpu()),
                'softmax_after': torch.softmax(next(h for h in heads if h.head_name == name).mixture_logits.bias.detach(), dim=0).cpu().tolist(),
            }
            for name in gate_updates
        },
        'gate_weight_diagnostics': {
            name: {
                **stats,
                'std_after': float(next(h for h in heads if h.head_name == name).mixture_logits.weight.detach().float().std(unbiased=False).cpu()),
                'max_abs_after': float(next(h for h in heads if h.head_name == name).mixture_logits.weight.detach().abs().max().cpu()),
            }
            for name, stats in gate_weight_diagnostics.items()
        },
        'neutral_centering_corrected': bool(getattr(model, 'neutral_centering', False)),
        'encoder_weight_max_abs': float(affine.abs().max()),
        'encoder_bias_max_abs': float(bias.abs().max()),
    }
    setattr(model, '_global_covariance_initialized', True)
    setattr(model, '_initialization_info', info)
    return info
