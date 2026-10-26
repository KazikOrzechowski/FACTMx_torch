"""Training utilities for the current PyTorch FACTMx implementation.

The default optimized regime combines three mechanisms that were diagnosed on
held-out structured prediction tasks:

1. a structured-first phase in which simple-view reconstruction is withheld;
2. a linear ramp that reintroduces the simple-view objective; and
3. encoder-gradient balancing, updated once per epoch by default.

The balancing rule uses only observed-data losses.  It never consumes labels,
simulator factors, true assignments, true component parameters, or any other
privileged information.

The implementation deliberately estimates balancing weights infrequently.  A
single update reuses two autograd traversals (one for the scaled simple loss and
one for the scaled structured loss) and the resulting weight is then reused for
all remaining minibatches in that epoch.  The EMA decay is adjusted to preserve
approximately the same smoothing that ``ema_beta_per_batch`` would have had if
updated every minibatch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class StructuredFirstBalanceConfig:
    """Configuration for structured-first encoder-gradient balancing.

    Parameters
    ----------
    structured_hold_epochs:
        Number of initial epochs for which the simple-view reconstruction
        contribution is exactly zero.
    simple_ramp_epochs:
        Number of epochs used to linearly introduce the simple-view objective
        after the structured-only hold.
    balance_encoder_gradients:
        Whether to adapt the simple-view multiplier from encoder gradient norms.
        When false, only the hold/ramp schedule is used.
    balance_update_every_epochs:
        Re-estimate gradient norms on the first minibatch every N epochs after
        the hold.  ``1`` is the validated low-overhead default.
    ema_beta_per_batch:
        Reference EMA coefficient for a hypothetical every-minibatch update.
        The actual coefficient is exponentiated by the number of minibatches
        between balance updates so changing update frequency does not silently
        change the smoothing time-scale.
    min_simple_multiplier, max_simple_multiplier:
        Bounds on the adaptive simple-view multiplier.
    target_gradient_ratio:
        Desired ratio of weighted simple encoder-gradient norm to structured
        encoder-gradient norm.  ``1`` targets equal norms.
    """

    structured_hold_epochs: int = 45
    simple_ramp_epochs: int = 45
    balance_encoder_gradients: bool = True
    balance_update_every_epochs: int = 1
    ema_beta_per_batch: float = 0.9
    min_simple_multiplier: float = 0.02
    max_simple_multiplier: float = 1.0
    target_gradient_ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.structured_hold_epochs < 0:
            raise ValueError("structured_hold_epochs must be >= 0")
        if self.simple_ramp_epochs < 0:
            raise ValueError("simple_ramp_epochs must be >= 0")
        if self.balance_update_every_epochs < 1:
            raise ValueError("balance_update_every_epochs must be >= 1")
        if not 0.0 <= self.ema_beta_per_batch < 1.0:
            raise ValueError("ema_beta_per_batch must be in [0, 1)")
        if self.min_simple_multiplier <= 0:
            raise ValueError("min_simple_multiplier must be > 0")
        if self.max_simple_multiplier < self.min_simple_multiplier:
            raise ValueError("max_simple_multiplier must be >= min_simple_multiplier")
        if self.target_gradient_ratio <= 0:
            raise ValueError("target_gradient_ratio must be > 0")

    @property
    def full_objective_start_epoch(self) -> int:
        """First epoch after the hold and linear ramp have both completed."""
        return self.structured_hold_epochs + self.simple_ramp_epochs


def simple_ramp_multiplier(epoch: int, hold_epochs: int, ramp_epochs: int) -> float:
    """Return the deterministic structured-first schedule multiplier.

    The first ``hold_epochs`` epochs return zero.  During the next
    ``ramp_epochs`` epochs the value increases linearly from ``1/ramp_epochs``
    to one.  Later epochs return one.
    """
    if epoch < hold_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    if epoch < hold_epochs + ramp_epochs:
        return float(epoch - hold_epochs + 1) / float(ramp_epochs)
    return 1.0


def kl_beta_for_epoch(epoch: int, final_beta: float, warmup_epochs: int) -> float:
    """Linear KL warm-up used by the benchmark and recommended training path."""
    if warmup_epochs <= 0:
        return float(final_beta)
    return float(final_beta) * min(1.0, float(epoch + 1) / float(warmup_epochs))


def encoder_gradient_norm_pair(
    simple_loss: Tensor,
    structured_loss: Tensor,
    encoder_parameters: Sequence[nn.Parameter],
) -> tuple[float, float]:
    """Return encoder gradient norms using exactly two autograd traversals.

    The graph is retained because the normal training backward pass follows.
    Gradients are detached immediately and are never written into ``.grad``.
    """
    params = tuple(p for p in encoder_parameters if p.requires_grad)
    if not params:
        return 0.0, 0.0

    # A model with only structured (or only simple) heads has an exactly
    # constant zero loss in the absent group. Such a value has no grad_fn;
    # allow_unused=True alone does not make autograd.grad accept it.
    g_simple = (
        torch.autograd.grad(simple_loss, params, retain_graph=True, allow_unused=True)
        if simple_loss.requires_grad else (None,) * len(params)
    )
    g_structured = (
        torch.autograd.grad(structured_loss, params, retain_graph=True, allow_unused=True)
        if structured_loss.requires_grad else (None,) * len(params)
    )

    simple_ss: Tensor | None = None
    structured_ss: Tensor | None = None
    for gs, gt in zip(g_simple, g_structured):
        if gs is not None:
            term = gs.detach().square().sum()
            simple_ss = term if simple_ss is None else simple_ss + term
        if gt is not None:
            term = gt.detach().square().sum()
            structured_ss = term if structured_ss is None else structured_ss + term

    simple_norm = 0.0 if simple_ss is None else float(torch.sqrt(simple_ss + 1e-30).cpu())
    structured_norm = (
        0.0 if structured_ss is None else float(torch.sqrt(structured_ss + 1e-30).cpu())
    )
    return simple_norm, structured_norm


def split_scaled_loss_components(
    model: nn.Module,
    components: Mapping[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    """Split already-computed FACTMx components into KL/simple/structured sums.

    ``model.loss_scales`` must contain the KL scale followed by one scale per
    head.  The head's ``structured`` flag determines which observed-data group
    receives each head loss.  This keeps balancing tied to the actual scaled
    training objective rather than to raw likelihood magnitudes.
    """
    heads = list(getattr(model, "heads"))
    scales = getattr(model, "loss_scales")
    if len(scales) != 1 + len(heads):
        raise ValueError("loss_scales must contain KL followed by one value per head")
    if "latent_kl" not in components:
        raise KeyError("components must contain 'latent_kl'")

    l_kl = components["latent_kl"] * scales[0]
    zero = l_kl.new_zeros(())
    l_simple = zero
    l_structured = zero

    for index, head in enumerate(heads, start=1):
        try:
            value = components[head.head_name] * scales[index]
        except KeyError as exc:
            raise KeyError(f"Missing loss component for head {head.head_name!r}") from exc
        if bool(getattr(head, "structured", False)):
            l_structured = l_structured + value
        else:
            l_simple = l_simple + value

    return l_kl, l_simple, l_structured


def combine_partitioned_objective(
    model: nn.Module,
    l_kl: Tensor,
    l_simple: Tensor,
    l_structured: Tensor,
    simple_multiplier: float,
) -> Tensor:
    """Combine partitioned losses while preserving FACTMx's mean-over-terms scale."""
    denominator = 1 + len(list(getattr(model, "heads")))
    return (l_kl + float(simple_multiplier) * l_simple + l_structured) / float(denominator)


class EncoderGradientBalancer:
    """Stateful, low-overhead adaptive simple-view loss multiplier."""

    def __init__(self, config: StructuredFirstBalanceConfig, n_batches_per_epoch: int) -> None:
        if n_batches_per_epoch < 1:
            raise ValueError("n_batches_per_epoch must be >= 1")
        self.config = config
        self.n_batches_per_epoch = int(n_batches_per_epoch)
        update_gap_batches = self.n_batches_per_epoch * config.balance_update_every_epochs
        self.ema_decay = config.ema_beta_per_batch ** update_gap_batches
        self.ema_log_multiplier: float | None = None
        self.current_adaptive_multiplier = 1.0
        self.balance_updates = 0

    def should_update(self, epoch: int, batch_index: int) -> bool:
        cfg = self.config
        if not cfg.balance_encoder_gradients:
            return False
        if epoch < cfg.structured_hold_epochs:
            return False
        if batch_index != 0:
            return False
        return (epoch - cfg.structured_hold_epochs) % cfg.balance_update_every_epochs == 0

    def schedule_multiplier(self, epoch: int) -> float:
        return simple_ramp_multiplier(
            epoch,
            self.config.structured_hold_epochs,
            self.config.simple_ramp_epochs,
        )

    def effective_simple_multiplier(self, epoch: int) -> float:
        schedule = self.schedule_multiplier(epoch)
        if schedule == 0.0:
            return 0.0
        adaptive = self.current_adaptive_multiplier if self.config.balance_encoder_gradients else 1.0
        return schedule * adaptive

    def maybe_update(
        self,
        *,
        epoch: int,
        batch_index: int,
        simple_loss: Tensor,
        structured_loss: Tensor,
        encoder_parameters: Sequence[nn.Parameter],
    ) -> dict[str, float | bool]:
        """Update the adaptive multiplier if this minibatch is scheduled.

        Returns diagnostics for logging.  No simulator truth or labels enter the
        calculation; only gradients of the two observed-data objective groups are
        used.
        """
        if epoch < self.config.structured_hold_epochs:
            self.current_adaptive_multiplier = 0.0
            return {
                "updated": False,
                "g_simple": float("nan"),
                "g_structured": float("nan"),
                "raw_multiplier": float("nan"),
                "adaptive_multiplier": 0.0,
            }

        # Once the hold ends, initialize to one until the first scheduled update.
        if self.current_adaptive_multiplier == 0.0:
            self.current_adaptive_multiplier = 1.0

        if not self.should_update(epoch, batch_index):
            return {
                "updated": False,
                "g_simple": float("nan"),
                "g_structured": float("nan"),
                "raw_multiplier": float("nan"),
                "adaptive_multiplier": self.current_adaptive_multiplier,
            }

        g_simple, g_structured = encoder_gradient_norm_pair(
            simple_loss,
            structured_loss,
            encoder_parameters,
        )
        raw = self.config.target_gradient_ratio * g_structured / (g_simple + 1e-12)
        raw = min(self.config.max_simple_multiplier, max(self.config.min_simple_multiplier, raw))
        log_raw = math.log(max(raw, 1e-12))

        if self.ema_log_multiplier is None:
            self.ema_log_multiplier = log_raw
        else:
            self.ema_log_multiplier = (
                self.ema_decay * self.ema_log_multiplier
                + (1.0 - self.ema_decay) * log_raw
            )

        adaptive = math.exp(self.ema_log_multiplier)
        adaptive = min(
            self.config.max_simple_multiplier,
            max(self.config.min_simple_multiplier, adaptive),
        )
        self.current_adaptive_multiplier = float(adaptive)
        self.balance_updates += 1

        return {
            "updated": True,
            "g_simple": float(g_simple),
            "g_structured": float(g_structured),
            "raw_multiplier": float(raw),
            "adaptive_multiplier": self.current_adaptive_multiplier,
        }
