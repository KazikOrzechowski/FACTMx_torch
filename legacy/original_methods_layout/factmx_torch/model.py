"""Top-level PyTorch FACTMx model and training loop."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .encoders import build_encoder
from .heads import BaseHead, build_head
from .training import (
    EncoderGradientBalancer,
    StructuredFirstBalanceConfig,
    combine_partitioned_objective,
    kl_beta_for_epoch,
    split_scaled_loss_components,
)


def _slice_data(value: Any, indices: Tensor) -> Any:
    if isinstance(value, Tensor):
        return value.index_select(0, indices)
    if isinstance(value, tuple):
        return tuple(_slice_data(item, indices) for item in value)
    if isinstance(value, list):
        return [_slice_data(item, indices) for item in value]
    raise TypeError(f"Unsupported data container: {type(value)!r}")


class FACTMxModel(nn.Module):
    """Multi-head variational latent-variable model.

    Parameters mirror the archived TensorFlow constructor where possible.
    ``refinement_steps=0`` is the faithful one-pass inference path.  Positive
    values alternate latent inference with structured posterior updates.
    """

    def __init__(
        self,
        dim_latent: int,
        heads_config: Sequence[Mapping[str, Any]],
        encoder_config: Mapping[str, Any] | None = None,
        beta: float = 1.0,
        loss_scales: Sequence[float] | None = None,
        refinement_steps: int = 0,
        name: str | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.dim_latent = int(dim_latent)
        self.beta = float(beta)
        self.refinement_steps = int(refinement_steps)
        self.name = name
        self.head_configs = copy.deepcopy(list(heads_config))
        self.encoder_config = copy.deepcopy(dict(encoder_config or {}))
        self.heads = nn.ModuleList(
            [build_head(config, dim_latent=self.dim_latent) for config in heads_config]
        )
        self.head_dims = [head.encoder_dim for head in self.heads]

        encoder_values = dict(encoder_config or {})
        encoder_type = encoder_values.pop("encoder_type", "Linear")
        encoder_values.pop("dim_latent", None)
        encoder_values.pop("head_dims", None)
        encoder_values.pop("prior_params", None)
        encoder_values.pop("layer_configs", None)
        self.encoder = build_encoder(
            encoder_type,
            dim_latent=self.dim_latent,
            head_dims=self.head_dims,
            **encoder_values,
        )

        if loss_scales is None:
            loss_scales = [1.0] * (1 + len(self.heads))
        if len(loss_scales) != 1 + len(self.heads):
            raise ValueError("loss_scales must contain KL followed by one value per head")
        self.register_buffer("loss_scales", torch.as_tensor(loss_scales, dtype=torch.float32))

    @staticmethod
    def _sample_latent(loc: Tensor, scale: Tensor, deterministic: bool) -> Tensor:
        if deterministic:
            return loc
        return loc + scale * torch.randn_like(scale)

    @staticmethod
    def _latent_kl(loc: Tensor, scale: Tensor) -> Tensor:
        variance = scale.square()
        return 0.5 * (loc.square() + variance - 1.0 - variance.log()).sum(-1).mean()

    def infer(
        self,
        data: Sequence[Any],
        deterministic: bool = False,
        refinement_steps: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, list[dict[str, Tensor]]]:
        if len(data) != len(self.heads):
            raise ValueError("One data item is required for each head")
        encoded = [
            head.encode(data_i, deterministic=deterministic)
            for head, data_i in zip(self.heads, data)
        ]
        steps = self.refinement_steps if refinement_steps is None else refinement_steps

        for step in range(steps + 1):
            encoder_input = torch.cat([item["encoder_input"] for item in encoded], dim=-1)
            loc, scale = self.encoder(encoder_input)
            latent = self._sample_latent(loc, scale, deterministic)
            if step == steps:
                break
            for index, (head, data_i) in enumerate(zip(self.heads, data)):
                if head.structured:
                    encoded[index]["encoder_input"] = head.refined_encoder_input(
                        latent, data_i
                    )
        return latent, loc, scale, encoded

    def get_latent_representation(
        self, data: Sequence[Any], refinement_steps: int | None = None
    ) -> Tensor:
        _, loc, _, _ = self.infer(
            data, deterministic=True, refinement_steps=refinement_steps
        )
        return loc

    def loss_components(
        self,
        data: Sequence[Any],
        deterministic_latent: bool = False,
    ) -> dict[str, Tensor]:
        latent, loc, scale, encoded = self.infer(
            data, deterministic=deterministic_latent
        )
        result: dict[str, Tensor] = {"latent_kl": self._latent_kl(loc, scale) * self.beta}
        for head, data_i, encoded_i in zip(self.heads, data, encoded):
            result[head.head_name] = head.loss(data_i, latent, encoded_i)
        return result

    def loss(self, data: Sequence[Any], deterministic_latent: bool = False) -> Tensor:
        components = self.loss_components(data, deterministic_latent=deterministic_latent)
        values = torch.stack(list(components.values()))
        # Mirrors reduce_mean(loss_scales * [KL, head losses]) in the archive.
        return (values * self.loss_scales).mean()

    def elbo(self, data: Sequence[Any]) -> Tensor:
        return -self.loss(data)

    def decode(self, latent: Tensor, data: Sequence[Any]) -> list[Any]:
        return [head.decode(latent, data_i) for head, data_i in zip(self.heads, data)]

    def full_pass(self, data: Sequence[Any], deterministic: bool = False) -> list[Any]:
        latent, _, _, _ = self.infer(data, deterministic=deterministic)
        return self.decode(latent, data)

    def fit(
        self,
        data: Sequence[Any],
        epochs: int = 1,
        batch_size: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 4e-3,
        shuffle: bool = True,
        temperature_update: float | None = None,
        min_temperature: float = 1e-2,
        max_grad_norm: float | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> list[float]:
        n = data[0].shape[0] if isinstance(data[0], Tensor) else data[0][0].shape[0]
        optimizer = optimizer or torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        history: list[float] = []
        self.train()
        for _epoch in range(epochs):
            order = torch.randperm(n, device=self.loss_scales.device) if shuffle else torch.arange(
                n, device=self.loss_scales.device
            )
            for start in range(0, n, batch_size):
                batch = _slice_data(data, order[start : start + batch_size])
                optimizer.zero_grad(set_to_none=True)
                objective = self.loss(batch)
                if not torch.isfinite(objective):
                    history.append(float("nan"))
                    return history
                objective.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                optimizer.step()
                history.append(float(objective.detach()))
            if temperature_update is not None:
                for head in self.heads:
                    if head.structured and head.temperature > min_temperature:
                        head.temperature = max(
                            min_temperature, head.temperature * temperature_update
                        )
        return history

    def fit_optimized(
        self,
        data: Sequence[Any],
        epochs: int = 160,
        batch_size: int = 64,
        lr: float = 3e-3,
        weight_decay: float = 1e-4,
        shuffle: bool = True,
        kl_final_beta: float = 0.1,
        kl_warmup_epochs: int = 30,
        structured_hold_epochs: int = 45,
        simple_ramp_epochs: int = 45,
        balance_encoder_gradients: bool = True,
        balance_update_every_epochs: int = 1,
        balance_ema_beta: float = 0.9,
        balance_min_simple_multiplier: float = 0.02,
        balance_max_simple_multiplier: float = 1.0,
        balance_target_gradient_ratio: float = 1.0,
        temperature_update: float | None = None,
        min_temperature: float = 1e-2,
        max_grad_norm: float | None = 10.0,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, Any]:
        """Train with structured-first warm-up and encoder-gradient balancing.

        This is the recommended observed-data-only optimization path.  The
        legacy :meth:`fit` method is intentionally retained unchanged for
        backwards compatibility.

        Gradient balancing compares the *scaled* simple-view and structured-view
        contributions at the shared encoder, updates the balancing coefficient
        on the first minibatch once per epoch by default, and reuses that
        coefficient for the remaining minibatches.  The EMA decay is corrected
        for the number of minibatches between updates.
        """
        n = data[0].shape[0] if isinstance(data[0], Tensor) else data[0][0].shape[0]
        if n <= 0:
            raise ValueError("Cannot fit FACTMx on an empty dataset")
        n_batches = (n + int(batch_size) - 1) // int(batch_size)

        balance_config = StructuredFirstBalanceConfig(
            structured_hold_epochs=int(structured_hold_epochs),
            simple_ramp_epochs=int(simple_ramp_epochs),
            balance_encoder_gradients=bool(balance_encoder_gradients),
            balance_update_every_epochs=int(balance_update_every_epochs),
            ema_beta_per_batch=float(balance_ema_beta),
            min_simple_multiplier=float(balance_min_simple_multiplier),
            max_simple_multiplier=float(balance_max_simple_multiplier),
            target_gradient_ratio=float(balance_target_gradient_ratio),
        )
        balancer = EncoderGradientBalancer(balance_config, n_batches_per_epoch=n_batches)
        encoder_parameters = [p for p in self.encoder.parameters() if p.requires_grad]

        optimizer = optimizer or torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )

        history: list[dict[str, float]] = []
        self.train()
        for epoch in range(int(epochs)):
            self.beta = kl_beta_for_epoch(epoch, kl_final_beta, kl_warmup_epochs)
            if shuffle:
                order = torch.randperm(n, device=self.loss_scales.device)
            else:
                order = torch.arange(n, device=self.loss_scales.device)

            epoch_losses: list[float] = []
            update_g_simple: list[float] = []
            update_g_structured: list[float] = []
            update_alpha: list[float] = []

            for batch_index, start in enumerate(range(0, n, batch_size)):
                batch = _slice_data(data, order[start : start + batch_size])
                optimizer.zero_grad(set_to_none=True)

                components = self.loss_components(batch, deterministic_latent=False)
                l_kl, l_simple, l_structured = split_scaled_loss_components(self, components)
                diagnostics = balancer.maybe_update(
                    epoch=epoch,
                    batch_index=batch_index,
                    simple_loss=l_simple,
                    structured_loss=l_structured,
                    encoder_parameters=encoder_parameters,
                )
                simple_multiplier = balancer.effective_simple_multiplier(epoch)
                objective = combine_partitioned_objective(
                    self, l_kl, l_simple, l_structured, simple_multiplier
                )
                if not torch.isfinite(objective):
                    raise FloatingPointError(f"Non-finite FACTMx objective at epoch {epoch}")

                objective.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                optimizer.step()
                epoch_losses.append(float(objective.detach().cpu()))

                if bool(diagnostics["updated"]):
                    update_g_simple.append(float(diagnostics["g_simple"]))
                    update_g_structured.append(float(diagnostics["g_structured"]))
                    update_alpha.append(float(diagnostics["adaptive_multiplier"]))

            history.append(
                {
                    "epoch": float(epoch),
                    "train_objective": float(sum(epoch_losses) / len(epoch_losses)),
                    "beta": float(self.beta),
                    "simple_schedule": float(balancer.schedule_multiplier(epoch)),
                    "adaptive_simple_multiplier": float(balancer.current_adaptive_multiplier),
                    "effective_simple_multiplier": float(balancer.effective_simple_multiplier(epoch)),
                    "g_simple": float(update_g_simple[-1]) if update_g_simple else float("nan"),
                    "g_structured": float(update_g_structured[-1]) if update_g_structured else float("nan"),
                }
            )

            if temperature_update is not None:
                for head in self.heads:
                    if head.structured and head.temperature > min_temperature:
                        head.temperature = max(
                            min_temperature, head.temperature * temperature_update
                        )

        return {
            "history": history,
            "balance_updates": int(balancer.balance_updates),
            "balance_ema_decay": float(balancer.ema_decay),
            "final_adaptive_simple_multiplier": float(balancer.current_adaptive_multiplier),
            "full_objective_start_epoch": int(balance_config.full_objective_start_epoch),
        }

    def save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                "dim_latent": self.dim_latent,
                "heads_config": self.head_configs,
                "encoder_config": self.encoder_config,
                "beta": self.beta,
                "loss_scales": self.loss_scales.detach().cpu().tolist(),
                "refinement_steps": self.refinement_steps,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: str, map_location: str | torch.device = "cpu") -> "FACTMxModel":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        state = payload.pop("state_dict")
        model = cls(**payload)
        model.load_state_dict(state)
        return model
