"""Missing-view inference extensions for :mod:`factmx_torch`.

The :class:`MissingViewFACTMx` model keeps missingness out of the latent-mean
pathway by centering each head's encoder summary around a training-derived
neutral value and replacing missing summaries with zero *after* centering.
Optionally, view availability can increase posterior uncertainty through a
separate non-negative scale-only path.

Training-data center estimation and latent-consistency losses are deliberately
kept outside this model.  They are experiment/training-policy concerns rather
than part of the probabilistic model API.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import FACTMxModel


def _batch_size(value: Any) -> int:
    """Return the leading patient-batch dimension for supported head inputs."""
    if isinstance(value, Tensor):
        return int(value.shape[0])
    if isinstance(value, (tuple, list)) and value:
        return _batch_size(value[0])
    raise TypeError(f"Unsupported data container for batch-size inference: {type(value)!r}")


class MissingViewFACTMx(FACTMxModel):
    """FACTMx with neutral per-view encoder centers and optional scale-only masks.

    For each head, the observed encoder summary is centered by a training-only
    reference mean.  A missing or artificially masked head therefore contributes
    the zero *centered* vector, equivalent to imputing the current neutral
    encoder summary.  View availability is never concatenated into the latent
    mean pathway.

    When ``availability_to_scale=True``, missing views add a learned non-negative
    amount to the posterior scale parameterization.  This allows missingness to
    increase uncertainty without directly changing the posterior mean.

    Notes
    -----
    Encoder-center estimation is intentionally not implemented here; callers
    should estimate centers from training data only and pass them through
    :meth:`set_encoder_centers`.  Likewise, full-vs-masked latent consistency is
    a training objective external to the model and should default to weight 0
    unless an experiment explicitly enables it.

    Buffer and parameter names are kept stable so compatible historical
    ``state_dict`` files can be loaded strictly when the model is instantiated
    with the checkpoint's original configuration (in particular
    ``availability_to_scale=True`` when ``missing_scale_raw`` is present).
    """

    def __init__(
        self,
        *model_args: Any,
        neutral_centering: bool = True,
        center_ema: float = 0.99,
        availability_to_scale: bool = False,
        missing_scale_init: float = -2.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(*model_args, **kwargs)
        self.neutral_centering = bool(neutral_centering)
        self.center_ema = float(center_ema)
        if not 0.0 <= self.center_ema < 1.0:
            raise ValueError("center_ema must be in [0,1)")
        self.availability_to_scale = bool(availability_to_scale)
        self.missing_scale_init = float(missing_scale_init)

        total_dim = int(sum(self.head_dims))
        self.register_buffer("encoder_center_initial", torch.zeros(total_dim))
        self.register_buffer("encoder_center_refined", torch.zeros(total_dim))
        self.register_buffer(
            "encoder_centers_initialized", torch.tensor(False, dtype=torch.bool)
        )

        self._head_slices: list[slice] = []
        offset = 0
        for dim in self.head_dims:
            self._head_slices.append(slice(offset, offset + int(dim)))
            offset += int(dim)

        if self.availability_to_scale:
            # softplus(-2.5) ~= 0.079: a modest initial uncertainty increase
            # per missing view with a usable gradient.  The non-negative delta
            # ensures missingness cannot spuriously decrease uncertainty.
            self.missing_scale_raw = nn.Parameter(
                torch.full((len(self.heads), self.dim_latent), float(missing_scale_init))
            )
        else:
            self.register_parameter("missing_scale_raw", None)

    def _checkpoint_config(self) -> dict[str, Any]:
        config = super()._checkpoint_config()
        config.update(
            {
                "neutral_centering": self.neutral_centering,
                "center_ema": self.center_ema,
                "availability_to_scale": self.availability_to_scale,
                "missing_scale_init": self.missing_scale_init,
            }
        )
        return config

    def set_encoder_centers(
        self, initial: Tensor, refined: Tensor | None = None
    ) -> None:
        """Set training-derived neutral encoder centers.

        ``initial`` and ``refined`` are concatenated per-head encoder summaries.
        If no refined center is supplied, the initial center is reused.
        """
        initial = torch.as_tensor(
            initial,
            dtype=self.encoder_center_initial.dtype,
            device=self.encoder_center_initial.device,
        ).reshape(-1)
        if initial.numel() != self.encoder_center_initial.numel():
            raise ValueError("Initial encoder center has the wrong width")

        refined = (
            initial
            if refined is None
            else torch.as_tensor(
                refined,
                dtype=self.encoder_center_refined.dtype,
                device=self.encoder_center_refined.device,
            ).reshape(-1)
        )
        if refined.numel() != self.encoder_center_refined.numel():
            raise ValueError("Refined encoder center has the wrong width")

        with torch.no_grad():
            self.encoder_center_initial.copy_(initial)
            self.encoder_center_refined.copy_(refined)
            self.encoder_centers_initialized.fill_(True)

    def _update_center_from_batch(
        self,
        raw_inputs: Sequence[Tensor],
        observed_mask: Tensor,
        refined: bool,
        structured_only: bool = False,
    ) -> None:
        """Update neutral centers by EMA using only observed training heads."""
        if not self.neutral_centering:
            return
        center = self.encoder_center_refined if refined else self.encoder_center_initial
        momentum = float(self.center_ema)

        with torch.no_grad():
            for j, (head, raw) in enumerate(zip(self.heads, raw_inputs)):
                if structured_only and not head.structured:
                    continue
                use = observed_mask[:, j]
                if not bool(use.any()):
                    continue
                mean = raw[use].mean(dim=0)
                sl = self._head_slices[j]
                if bool(self.encoder_centers_initialized):
                    center[sl].mul_(momentum).add_(mean, alpha=1.0 - momentum)
                else:
                    # Preserve the historical first-update behavior:
                    # initialize directly from the first observed batch rather
                    # than shrinking that batch mean toward zero.
                    center[sl].copy_(mean)

                # Simple heads are unchanged by contextual refinement, so their
                # refined neutral center is synchronized with the initial one.
                if not head.structured and not refined:
                    self.encoder_center_refined[sl].copy_(center[sl])
            self.encoder_centers_initialized.fill_(True)

    def _center_and_mask(
        self,
        raw_inputs: Sequence[Tensor],
        encoder_mask: Tensor,
        refined: bool,
    ) -> list[Tensor]:
        center = self.encoder_center_refined if refined else self.encoder_center_initial
        centered: list[Tensor] = []
        for j, raw in enumerate(raw_inputs):
            sl = self._head_slices[j]
            value = raw - center[sl].to(raw.dtype) if self.neutral_centering else raw
            mask = encoder_mask[:, j].to(raw.dtype).unsqueeze(-1)
            centered.append(value * mask)
        return centered

    def _posterior(
        self, encoder_input: Tensor, encoder_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        # The scale-only missingness path is defined for LinearEncoder's diagonal
        # Normal parameterization.  Other encoder types retain their native
        # behavior while still benefiting from neutral centering/masking.
        if not hasattr(self.encoder, "loc") or not hasattr(self.encoder, "raw_scale"):
            return self.encoder(encoder_input)

        loc = self.encoder.loc(encoder_input)
        raw_scale = self.encoder.raw_scale.unsqueeze(0).expand(loc.shape[0], -1)
        if self.availability_to_scale and self.missing_scale_raw is not None:
            missing = (~encoder_mask).to(loc.dtype)
            raw_scale = raw_scale + missing @ F.softplus(self.missing_scale_raw)

        parameterization = getattr(self.encoder, "scale_parameterization", "softplus")
        eps = float(getattr(self.encoder, "eps", 1e-4))
        if parameterization == "softplus":
            scale = F.softplus(raw_scale) + eps
        elif parameterization == "exp":
            scale = torch.exp(raw_scale) + eps
        else:
            scale = F.relu(raw_scale) + eps
        return loc, scale

    def posterior_from_encoded(
        self,
        encoded: Sequence[Mapping[str, Tensor]],
        encoder_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute the initial posterior from already-computed head summaries.

        This enables a caller to compare full and artificially masked posterior
        means without performing a second structured encoding pass.
        """
        raw_inputs = [item["encoder_input"] for item in encoded]
        centered = self._center_and_mask(raw_inputs, encoder_mask, refined=False)
        return self._posterior(torch.cat(centered, dim=-1), encoder_mask)

    def infer_masked(
        self,
        data: Sequence[Any],
        encoder_mask: Tensor,
        deterministic: bool = False,
        refinement_steps: int | None = None,
        *,
        update_centers: bool = False,
        center_update_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, list[dict[str, Tensor]]]:
        """Infer a posterior while masking selected patient/view encoder inputs.

        ``encoder_mask`` has shape ``[batch, n_heads]`` and controls which heads
        contribute evidence to the latent encoder.  ``center_update_mask`` can be
        supplied separately so artificial dropout never contaminates the
        training-only neutral-center estimates.
        """
        if len(data) != len(self.heads):
            raise ValueError("One data item is required for each head")

        batch_n = _batch_size(data[0])
        expected_shape = (batch_n, len(self.heads))
        if tuple(encoder_mask.shape) != expected_shape:
            raise ValueError(
                f"encoder_mask must be [{batch_n},{len(self.heads)}], "
                f"got {tuple(encoder_mask.shape)}"
            )
        if encoder_mask.dtype != torch.bool:
            raise TypeError("encoder_mask must have dtype torch.bool")

        if center_update_mask is None:
            center_update_mask = encoder_mask
        if tuple(center_update_mask.shape) != expected_shape:
            raise ValueError("center_update_mask must have the same shape as encoder_mask")
        if center_update_mask.dtype != torch.bool:
            raise TypeError("center_update_mask must have dtype torch.bool")

        encoded = [
            head.encode(data_i, deterministic=deterministic)
            for head, data_i in zip(self.heads, data)
        ]
        raw_inputs = [item["encoder_input"] for item in encoded]
        if update_centers and self.training:
            self._update_center_from_batch(raw_inputs, center_update_mask, refined=False)

        steps = self.refinement_steps if refinement_steps is None else int(refinement_steps)
        if steps < 0:
            raise ValueError("refinement_steps must be non-negative")

        for step in range(steps + 1):
            centered = self._center_and_mask(raw_inputs, encoder_mask, refined=step > 0)
            loc, scale = self._posterior(torch.cat(centered, dim=-1), encoder_mask)
            latent = self._sample_latent(loc, scale, deterministic)
            if step == steps:
                break

            next_raw = list(raw_inputs)
            for j, (head, data_i) in enumerate(zip(self.heads, data)):
                if head.structured:
                    next_raw[j] = head.refined_encoder_input(latent, data_i)
            raw_inputs = next_raw
            if update_centers and self.training:
                self._update_center_from_batch(
                    raw_inputs,
                    center_update_mask,
                    refined=True,
                    structured_only=True,
                )

        # Keep the original local head encodings in the return object, matching
        # FACTMxModel.infer and the historical implementation.
        return latent, loc, scale, encoded
