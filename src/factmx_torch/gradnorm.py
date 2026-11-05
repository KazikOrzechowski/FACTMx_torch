"""Chen et al. (ICML 2018) GradNorm for the three FACTMx simple-view losses.

Integrated in the FACTMx v0.1.4.dev1-derived research branch.
Use with the runner's per-head, already-scaled simple-view losses before the
existing outer simple-vs-H&E controller.

Reference: https://proceedings.mlr.press/v80/chen18a.html
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn


class MolecularGradNorm(nn.Module):
    """Learn strictly positive weights summing to the number of simple views.

    Model update: detach learned weights and differentiate the weighted sum of
    original FACTMx per-head losses through the network.

    Weight update: measure each UNWEIGHTED loss's L2 gradient on a specified
    shared encoder layer; apply the Chen L1 GradNorm objective only to weights,
    with a detached target, then project weights onto the positive simplex.

    By default, GradNorm uses L_i(t)/L_i(0) as its progress ratio. If a Gaussian
    NLL can be nonpositive or its learned variance makes NLL progress misleading,
    supply strictly positive, consistently normalized `progress_losses` in
    queue_update. This is an explicitly labeled adaptation of canonical GradNorm,
    not a change to the losses backpropagated through FACTMx.

    Parameters
    ----------
    task_names : ordered simple views, never include H&E or latent KL
    alpha : restoring-force exponent in Chen et al. (0 -> equal gradient norms)
    lr : independent optimizer learning rate for task weights (not network lr)
    warmup_calls : number of initial queue_update calls without weight updates;
        initial progress baselines are captured *after* the warmup.
    update_interval : update task weights every N eligible calls
    min_weight : small positive projection floor (weights are then renormalized)
    device : create weights on same device as training tensors/model
    """

    def __init__(
        self,
        task_names: Sequence[str] = ("rna", "cna", "methylation"),
        *,
        alpha: float = 1.5,
        lr: float = 0.025,
        warmup_calls: int = 0,
        update_interval: int = 1,
        min_weight: float = 1e-4,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        names = tuple(task_names)
        if len(names) < 2 or len(set(names)) != len(names):
            raise ValueError("Need at least two distinct modality names")
        if not (0 <= alpha < float("inf")):
            raise ValueError("alpha must be finite and nonnegative")
        if not (0 < lr < float("inf")):
            raise ValueError("lr must be finite and positive")
        if warmup_calls < 0 or update_interval < 1 or min_weight <= 0:
            raise ValueError("Invalid warmup_calls, update_interval or min_weight")
        if min_weight >= 1:
            raise ValueError("min_weight must be below the mean weight of 1")
        self.task_names = names
        self.alpha = float(alpha)
        self.lr = float(lr)
        self.warmup_calls = int(warmup_calls)
        self.update_interval = int(update_interval)
        self.min_weight = float(min_weight)
        self.weights = nn.Parameter(torch.ones(len(names), dtype=torch.float32, device=device))
        self.register_buffer("initial_progress_losses", torch.full((len(names),), float("nan"), device=device))
        self.register_buffer("calls", torch.zeros((), dtype=torch.long, device=device))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long, device=device))
        self._optimizer = torch.optim.Adam([self.weights], lr=float(lr))
        self._pending_grad: Tensor | None = None

    def _active(self, active: Mapping[str, bool] | None) -> dict[str, bool]:
        if active is None:
            return {n: True for n in self.task_names}
        unknown = set(active) - set(self.task_names)
        if unknown:
            raise KeyError(f"Unknown task(s) in active mask: {sorted(unknown)}")
        # Explicitly require per-view availability; don't infer 'missing' from zeros.
        missing = set(self.task_names) - set(active)
        if missing:
            raise KeyError(f"Missing activity flags for: {sorted(missing)}")
        return {n: bool(active[n]) for n in self.task_names}

    @staticmethod
    def _scalar(value: Tensor, label: str) -> Tensor:
        if not isinstance(value, Tensor) or value.numel() != 1:
            raise TypeError(f"{label} must be a scalar torch.Tensor")
        if not bool(torch.isfinite(value.detach()).all()):
            raise ValueError(f"{label} is not finite")
        return value.reshape(())

    def weighted_loss(
        self,
        losses: Mapping[str, Tensor],
        *,
        active: Mapping[str, bool] | None = None,
    ) -> Tensor:
        """Sum detached-weight * native per-view loss; NO KL or H&E included."""
        mask = self._active(active)
        terms = []
        for idx, name in enumerate(self.task_names):
            if mask[name]:
                if name not in losses:
                    raise KeyError(f"Missing active head loss: {name}")
                term = self._scalar(losses[name], f"loss[{name}]")
                # Clone so a weight-optimizer step never mutates a live model graph.
                terms.append(self.weights[idx].detach().clone() * term)
        if not terms:
            # A genuine H&E-only patient minibatch still updates H&E and KL.
            # Keep a differentiable zero simple-group term without inventing
            # observations or changing training-row availability masks.
            if not losses:
                raise ValueError("No molecular heads were configured")
            return next(iter(losses.values())) * 0.0
        return torch.stack(terms).sum()

    def queue_update(
        self,
        losses: Mapping[str, Tensor],
        shared_parameters: Sequence[nn.Parameter],
        *,
        progress_losses: Mapping[str, Tensor] | None = None,
        active: Mapping[str, bool] | None = None,
    ) -> dict[str, Any]:
        """Compute weight-only GradNorm step, without touching network .grad fields.

        Call AFTER the forward pass and BEFORE the network loss .backward().
        Call .step() AFTER the network optimizer step. Missing-view minibatches
        skip GradNorm updates (while the network may still train on present views).
        This is deliberate: Chen's relative training rate assumes fixed task set.
        """
        if self._pending_grad is not None:
            raise RuntimeError("Call step() before queueing another GradNorm update")
        self.calls.add_(1)
        call = int(self.calls.item())
        result: dict[str, Any] = {
            "call": call,
            "update": int(self.updates.item()),
            "weights": self.weight_dict(),
            "status": "skipped",
        }
        mask = self._active(active)
        if call <= self.warmup_calls:
            result["reason"] = "warmup"
            return result
        if not all(mask.values()):
            result["reason"] = "not_all_simple_views_observed"
            return result
        params = tuple(p for p in shared_parameters if p.requires_grad)
        if not params:
            raise ValueError("shared_parameters must include trainable shared encoder parameters")
        if progress_losses is None:
            progress_losses = losses
            result["progress_source"] = "native_scaled_nll"
        else:
            result["progress_source"] = "explicit_positive_proxy"

        raw_losses = [self._scalar(losses[n], f"loss[{n}]") for n in self.task_names]
        progress = torch.stack([
            self._scalar(progress_losses[n], f"progress_losses[{n}]").detach().to(
                device=self.weights.device, dtype=self.weights.dtype
            ) for n in self.task_names
        ])
        if not bool((progress > 0).all()):
            raise ValueError(
                "GradNorm relative training rates require positive progress losses. "
                "Provide positive standardized reconstruction errors as progress_losses "
                "if any native Gaussian NLL is nonpositive."
            )
        first = torch.isnan(self.initial_progress_losses)
        if bool(first.any()):
            with torch.no_grad():
                self.initial_progress_losses[first] = progress[first]
        relative_losses = progress / self.initial_progress_losses.clamp_min(1e-12)
        rate = relative_losses / relative_losses.mean().clamp_min(1e-12)
        result["loss_ratios"] = dict(zip(self.task_names, relative_losses.tolist()))
        result["inverse_relative_rates"] = dict(zip(self.task_names, rate.tolist()))
        result["initial_progress_losses"] = dict(zip(self.task_names, self.initial_progress_losses.tolist()))

        if ((call - self.warmup_calls - 1) % self.update_interval) != 0:
            result["reason"] = "update_interval"
            return result

        # Grad norms of ORIGINAL, normalized per-view NLLs on the SAME shared
        # parameters. The network graph is retained for the real backward pass.
        unweighted_norms = []
        for name, L in zip(self.task_names, raw_losses):
            if not L.requires_grad:
                raise RuntimeError(f"Head {name} loss has no computational graph")
            grad_list = torch.autograd.grad(
                L, params, retain_graph=True, create_graph=False, allow_unused=True
            )
            present = [g.detach().float() for g in grad_list if g is not None]
            if not present:
                raise RuntimeError(
                    f"Head {name} is not connected to the selected shared encoder "
                    "parameters; choose the actual last shared encoder layer"
                )
            sq = torch.stack([torch.sum(g * g) for g in present]).sum()
            unweighted_norms.append(torch.sqrt(sq.clamp_min(0)).to(self.weights.device))
        grad0 = torch.stack(unweighted_norms)
        if not bool(torch.isfinite(grad0).all()):
            raise ValueError("Nonfinite molecular encoder gradient norm")
        if bool((grad0 == 0).any()):
            result.update(reason="zero_shared_gradient", unweighted_grad_norms=dict(zip(self.task_names,grad0.tolist())))
            return result

        weighted_norms = self.weights * grad0
        # Canonical Chen target: stop gradients through both mean norm and rates.
        target = (weighted_norms.mean() * rate.pow(self.alpha)).detach()
        gradnorm_objective = (weighted_norms - target).abs().sum()
        gradient_w = torch.autograd.grad(gradnorm_objective, self.weights)[0]
        if not bool(torch.isfinite(gradient_w).all()):
            raise ValueError("Nonfinite GradNorm task-weight gradient")
        self._pending_grad = gradient_w.detach().clone()
        result.update(
            status="queued",
            gradnorm_objective=float(gradnorm_objective.detach().item()),
            unweighted_grad_norms=dict(zip(self.task_names, grad0.tolist())),
            weighted_grad_norms=dict(zip(self.task_names, weighted_norms.detach().tolist())),
            target_grad_norms=dict(zip(self.task_names, target.tolist())),
            weight_gradients=dict(zip(self.task_names, gradient_w.detach().tolist())),
        )
        return result

    def step(self) -> bool:
        """Apply the queued *weight-only* optimizer update; return update status."""
        if self._pending_grad is None:
            return False
        self._optimizer.zero_grad(set_to_none=True)
        self.weights.grad = self._pending_grad
        self._optimizer.step()
        with torch.no_grad():
            # Canonical positive weight renormalization: sum_i w_i == n_tasks.
            self.weights.clamp_(min=self.min_weight)
            self.weights.mul_(len(self.task_names) / self.weights.sum())
        self.weights.grad = None
        self._pending_grad = None
        self.updates.add_(1)
        return True

    def weight_dict(self) -> dict[str, float]:
        return dict(zip(self.task_names, [float(x) for x in self.weights.detach().tolist()]))

    def training_state_dict(self) -> dict[str, Any]:
        if self._pending_grad is not None:
            raise RuntimeError("Save controller checkpoints only between completed training steps")
        return {
            "task_names": self.task_names,
            "alpha": self.alpha,
            "lr": self.lr,
            "warmup_calls": self.warmup_calls,
            "update_interval": self.update_interval,
            "min_weight": self.min_weight,
            "module": self.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }

    def load_training_state_dict(self, saved: Mapping[str, Any]) -> None:
        if tuple(saved["task_names"]) != self.task_names:
            raise ValueError("Mismatch in GradNorm task names / order")
        for name in ("alpha", "lr", "warmup_calls", "update_interval", "min_weight"):
            if getattr(self, name) != saved[name]:
                raise ValueError(f"GradNorm hyperparameter mismatch: {name}")
        self.load_state_dict(saved["module"], strict=True)
        self._optimizer.load_state_dict(saved["optimizer"])
        self._pending_grad = None
