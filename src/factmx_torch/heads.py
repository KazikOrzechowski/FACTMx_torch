"""Observation heads for the PyTorch FACTMx implementation.

The default structured-head settings reproduce the archived TensorFlow
objective.  Alternative settings expose the stabilization and collapsed-
mixture formulations used by the diagnostic experiment.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _orthogonal_linear(layer: nn.Linear) -> None:
    nn.init.orthogonal_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class BaseHead(nn.Module):
    """Common interface for simple and structured FACTMx heads."""

    structured = False

    def __init__(self, dim: int, dim_latent: int, head_name: str) -> None:
        super().__init__()
        self.dim = int(dim)
        self.dim_latent = int(dim_latent)
        self.head_name = head_name
        self.encoder_dim = self.dim

    def encode(self, data: Any, deterministic: bool = False) -> dict[str, Tensor]:
        raise NotImplementedError

    def loss(self, data: Any, latent: Tensor, encoded: Mapping[str, Tensor]) -> Tensor:
        raise NotImplementedError


class BernoulliHead(BaseHead):
    head_type = "Bernoulli"

    def __init__(self, dim: int, dim_latent: int, head_name: str, **_: Any) -> None:
        super().__init__(dim, dim_latent, head_name)
        self.logits = nn.Linear(dim_latent, dim)

    def encode(self, data: Tensor, deterministic: bool = False) -> dict[str, Tensor]:
        return {"encoder_input": data}

    def decode_params(self, latent: Tensor) -> Tensor:
        return self.logits(latent)

    def decode(self, latent: Tensor, data: Tensor | None = None) -> Tensor:
        return torch.distributions.Bernoulli(logits=self.decode_params(latent)).sample()

    def loss(self, data: Tensor, latent: Tensor, encoded: Mapping[str, Tensor]) -> Tensor:
        return F.binary_cross_entropy_with_logits(self.decode_params(latent), data)


class MultinomialHead(BaseHead):
    head_type = "Multinomial"

    def __init__(
        self,
        dim_pos: int,
        dim_cat: int,
        dim: int,
        dim_latent: int,
        head_name: str,
        eps: float = 1e-3,
        encode_logits: bool = True,
        **_: Any,
    ) -> None:
        super().__init__(dim, dim_latent, head_name)
        self.dim_pos = int(dim_pos)
        self.dim_cat = int(dim_cat)
        self.eps = eps
        self.encode_logits = encode_logits
        self.decoder = nn.Linear(dim_latent, dim_pos * dim_cat)
        self.encoder_dim = dim_pos * dim_cat

    def encode(
        self, data: tuple[Tensor, Tensor], deterministic: bool = False
    ) -> dict[str, Tensor]:
        observations, counts = data
        if self.encode_logits:
            values = observations / counts.unsqueeze(-1).clamp_min(1.0) + self.eps
            values = values.log()
        else:
            values = observations
        return {"encoder_input": values.reshape(values.shape[0], -1)}

    def decode_params(self, latent: Tensor) -> Tensor:
        logits = self.decoder(latent)
        logits = torch.logaddexp(logits, logits.new_full((), math.log(self.eps)))
        return logits.reshape(-1, self.dim_pos, self.dim_cat)

    def loss(
        self,
        data: tuple[Tensor, Tensor],
        latent: Tensor,
        encoded: Mapping[str, Tensor],
    ) -> Tensor:
        observations, counts = data
        log_probs = F.log_softmax(self.decode_params(latent), dim=-1)
        log_coeff = torch.lgamma(counts + 1) - torch.lgamma(observations + 1).sum(-1)
        log_likelihood = log_coeff + (observations * log_probs).sum(-1)
        # The archive sums over positions and divides only by batch size.
        return -log_likelihood.sum() / observations.shape[0]

    def decode(self, latent: Tensor, data: tuple[Tensor, Tensor]) -> Tensor:
        _, counts = data
        probs = torch.softmax(self.decode_params(latent), dim=-1)
        samples = []
        for row_probs, row_counts in zip(probs, counts):
            samples.append(
                torch.stack(
                    [
                        torch.distributions.Multinomial(int(n.item()), p).sample()
                        for p, n in zip(row_probs, row_counts)
                    ]
                )
            )
        return torch.stack(samples)


class MultiNormalHead(BaseHead):
    head_type = "MultiNormal"

    def __init__(
        self,
        dim: int,
        dim_latent: int,
        head_name: str,
        eps: float = 1e-3,
        **_: Any,
    ) -> None:
        super().__init__(dim, dim_latent, head_name)
        self.eps = eps
        self.loc = nn.Linear(dim_latent, dim)
        _orthogonal_linear(self.loc)
        self.log_scale = nn.Parameter(torch.full((dim,), math.log(eps)))

    def encode(self, data: Tensor, deterministic: bool = False) -> dict[str, Tensor]:
        return {"encoder_input": data}

    def decode_params(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        return self.loc(latent), torch.exp(self.log_scale) + self.eps

    def loss(self, data: Tensor, latent: Tensor, encoded: Mapping[str, Tensor]) -> Tensor:
        loc, scale = self.decode_params(latent)
        return -torch.distributions.Independent(
            torch.distributions.Normal(loc, scale), 1
        ).log_prob(data).mean()

    def decode(self, latent: Tensor, data: Tensor | None = None) -> Tensor:
        loc, scale = self.decode_params(latent)
        return torch.distributions.Normal(loc, scale).sample()


class DeepSetContextEncoder(nn.Module):
    """Permutation-invariant view-local context encoder.

    This is deliberately small and head-local: ``rho(mean(phi(x_j)))`` maps a
    patient's context observations to a fixed-width embedding.  A small non-zero
    initialized projection from that embedding to component logits
    avoids a first-step gradient block while keeping the initial residual modest.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.phi = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"DeepSetContextEncoder expects [batch,set,features], got {tuple(x.shape)}"
            )
        if x.shape[1] == 0:
            return x.new_zeros((x.shape[0], self.hidden_dim))
        return self.rho(self.phi(x).mean(dim=1))


class StructuredMixtureHead(BaseHead):
    """Shared assignment machinery for GMM and topic heads."""

    structured = True

    def __init__(
        self,
        dim: int,
        dim_latent: int,
        head_name: str,
        temperature: float = 1e-4,
        eps: float = 1e-3,
        assignment_estimator: str = "relaxed",
        objective: str = "faithful_elbo",
        agreement_weight: float = 1.0,
        distillation_weight: float = 0.0,
        contextual_gate: bool = False,
        contextual_prior_strength: float = 4.0,
        contextual_steps: int = 2,
        contextual_encoder: str = "none",
        contextual_hidden_dim: int = 64,
        contextual_residual_scale: float = 1.0,
        contextual_residual_init_std: float = 0.01,
        contextual_distillation_weight: float = 0.0,
    ) -> None:
        super().__init__(dim, dim_latent, head_name)
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.assignment_estimator = assignment_estimator
        self.objective = objective
        self.agreement_weight = float(agreement_weight)
        self.distillation_weight = float(distillation_weight)
        self.contextual_gate = bool(contextual_gate)
        self.contextual_prior_strength = float(contextual_prior_strength)
        self.contextual_steps = int(contextual_steps)
        self.contextual_encoder = str(contextual_encoder).lower()
        self.contextual_hidden_dim = int(contextual_hidden_dim)
        self.contextual_residual_scale = float(contextual_residual_scale)
        self.contextual_residual_init_std = float(contextual_residual_init_std)
        self.contextual_distillation_weight = float(contextual_distillation_weight)
        if self.contextual_residual_init_std <= 0:
            raise ValueError("contextual_residual_init_std must be positive")
        if self.contextual_distillation_weight < 0:
            raise ValueError("contextual_distillation_weight must be non-negative")
        if self.contextual_encoder not in {"none", "deepsets"}:
            raise ValueError("contextual_encoder must be 'none' or 'deepsets'")
        if self.contextual_hidden_dim <= 0:
            raise ValueError("contextual_hidden_dim must be positive")
        self.context_set_encoder: DeepSetContextEncoder | None = None
        self.context_to_logits: nn.Linear | None = None
        if self.contextual_prior_strength < 0:
            raise ValueError("contextual_prior_strength must be non-negative")
        if self.contextual_steps < 0:
            raise ValueError("contextual_steps must be non-negative")
        self.mixture_logits = nn.Linear(dim_latent, dim)
        nn.init.normal_(self.mixture_logits.weight, std=0.05)
        nn.init.ones_(self.mixture_logits.bias)

    def component_log_likelihood(self, data: Tensor) -> Tensor:
        raise NotImplementedError

    def local_assignment_logits(self, data: Tensor) -> Tensor:
        raise NotImplementedError

    def decode_mixture_logits(self, latent: Tensor) -> Tensor:
        log_probs = F.log_softmax(self.mixture_logits(latent), dim=-1)
        return torch.logaddexp(log_probs, log_probs.new_full((), math.log(self.eps)))

    def _assignment_weights(self, logits: Tensor) -> Tensor:
        if self.assignment_estimator == "relaxed":
            return F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=-1)
        if self.assignment_estimator == "straight_through":
            return F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
        if self.assignment_estimator == "mean":
            return F.softmax(logits, dim=-1)
        raise ValueError(f"Unknown assignment estimator: {self.assignment_estimator}")

    def encode(self, data: Tensor, deterministic: bool = False) -> dict[str, Tensor]:
        local_logits = self.local_assignment_logits(data)
        assignment = (
            F.softmax(local_logits, dim=-1)
            if deterministic
            else self._assignment_weights(local_logits)
        )
        proportions = assignment.mean(dim=1) + self.eps
        return {
            "encoder_input": proportions.log(),
            "assignment": assignment,
            "local_logits": local_logits,
        }

    def prior_gate_probs(self, latent: Tensor) -> Tensor:
        """Patient mixture proportions decoded from the shared latent state."""
        return F.softmax(self.decode_mixture_logits(latent), dim=-1)

    def _build_context_encoder(self, input_dim: int) -> None:
        if self.contextual_encoder == "deepsets":
            self.context_set_encoder = DeepSetContextEncoder(
                input_dim=int(input_dim), hidden_dim=self.contextual_hidden_dim
            )
            self.context_to_logits = nn.Linear(self.contextual_hidden_dim, self.dim)
            # Use a small non-zero output map.  Exact zeros would block the
            # first-step gradient into the Deep Sets encoder (the same zero-map
            # pathology previously observed for structured->encoder coupling).
            nn.init.normal_(
                self.context_to_logits.weight, std=self.contextual_residual_init_std
            )
            nn.init.zeros_(self.context_to_logits.bias)

    def context_encoder_input(self, context_data: Tensor) -> Tensor:
        """Representation consumed by the neural set encoder.

        Subclasses can override this without changing their probabilistic
        emission likelihood.  In particular, Topic heads use normalized word
        frequencies for neural context while retaining raw counts for the
        multinomial mixture likelihood.
        """
        return context_data

    def contextual_prior_probs(self, latent: Tensor, context_data: Tensor) -> Tensor:
        """Latent gate optionally corrected by a learned Deep Sets residual."""
        if self.contextual_encoder == "none":
            return self.prior_gate_probs(latent)
        if self.context_set_encoder is None or self.context_to_logits is None:
            raise RuntimeError("Context encoder was not initialized by the structured head")
        context_x = self.context_encoder_input(context_data)
        embedding = self.context_set_encoder(context_x)
        delta = self.context_to_logits(embedding)
        # Softmax is invariant to a common shift; centering removes that
        # unidentifiable degree of freedom and makes the residual interpretable.
        delta = delta - delta.mean(dim=-1, keepdim=True)
        base_logits = self.decode_mixture_logits(latent)
        return F.softmax(
            base_logits + self.contextual_residual_scale * delta, dim=-1
        )

    def contextual_gate_probs(
        self,
        latent: Tensor,
        context_data: Tensor,
        prior_strength: float | None = None,
        steps: int | None = None,
    ) -> Tensor:
        """Infer a view-local patient gate from this view's context observations.

        The latent-decoded gate is treated as a prior mean.  Context observations
        contribute soft component counts, producing an empirical-Bayes posterior
        mean that is iterated a small number of times::

            pi0 = softmax(W z)
            r_jk ∝ pi_tk p(x_j | k)
            pi_{t+1,k} = (kappa pi0_k + sum_j r_jk) / (kappa + n_context)

        This update is differentiable and uses context observations only.  Query
        observations must never be passed here during held-out evaluation.
        """
        prior = self.contextual_prior_probs(latent, context_data)
        if not self.contextual_gate:
            return prior

        kappa = self.contextual_prior_strength if prior_strength is None else float(prior_strength)
        n_steps = self.contextual_steps if steps is None else int(steps)
        if kappa < 0:
            raise ValueError("prior_strength must be non-negative")
        if n_steps <= 0 or (context_data[0] if isinstance(context_data, (tuple, list)) else context_data).shape[1] == 0:
            return prior

        component_ll = self.component_log_likelihood(context_data)
        gate = prior
        n_context = float(context_data[0].shape[1] if isinstance(context_data, (tuple, list)) else context_data.shape[1])
        for _ in range(n_steps):
            log_gate = gate.clamp_min(1e-12).log()
            responsibilities = F.softmax(log_gate.unsqueeze(1) + component_ll, dim=-1)
            gate = (kappa * prior + responsibilities.sum(dim=1)) / (kappa + n_context)
            gate = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return gate

    def contextual_distillation_loss(self, latent: Tensor, context_data: Tensor) -> Tensor:
        """KL from a detached contextual posterior to the latent-only gate.

        The analytic context posterior acts as an observed-data teacher.  Its
        probabilities are detached before the KL is formed, so this term sends
        gradients only through the latent-decoded patient gate (and therefore
        through ``z`` / the shared encoder), not back through the contextual
        responsibility update or the mixture emission parameters.

        Query observations are never used here: ``context_data`` must be the
        same context set available to held-out patient inference.
        """
        if not self.contextual_gate or self.contextual_distillation_weight <= 0.0:
            return latent.new_zeros(())

        with torch.no_grad():
            target = self.contextual_gate_probs(latent, context_data).detach()
        student = self.prior_gate_probs(latent)
        target_log = target.clamp_min(1e-12).log()
        student_log = student.clamp_min(1e-12).log()
        return (target * (target_log - student_log)).sum(dim=-1).mean()

    def contextual_distillation_loss_from_target(
        self, latent: Tensor, detached_target: Tensor
    ) -> Tensor:
        """KL(detached contextual target || latent-only gate)."""
        student = self.prior_gate_probs(latent)
        target = detached_target.detach()
        target_log = target.clamp_min(1e-12).log()
        student_log = student.clamp_min(1e-12).log()
        return (target * (target_log - student_log)).sum(dim=-1).mean()

    def posterior_logits(
        self,
        latent: Tensor,
        data: Tensor,
        gate_probs: Tensor | None = None,
    ) -> Tensor:
        gate = self.prior_gate_probs(latent) if gate_probs is None else gate_probs
        return gate.clamp_min(1e-12).log().unsqueeze(1) + self.component_log_likelihood(data)

    def posterior_probs(
        self,
        latent: Tensor,
        data: Tensor,
        gate_probs: Tensor | None = None,
    ) -> Tensor:
        return F.softmax(self.posterior_logits(latent, data, gate_probs=gate_probs), dim=-1)

    def predictive_gate_probs(self, latent: Tensor, context_data: Tensor) -> Tensor:
        """Gate used for held-out query prediction."""
        return self.contextual_gate_probs(latent, context_data)

    def predictive_loss(
        self,
        query_data: Tensor,
        latent: Tensor,
        context_data: Tensor,
        encoded_query: Mapping[str, Tensor] | None = None,
        include_context_distillation: bool = False,
    ) -> Tensor:
        """Score held-out query observations using a context-only local gate."""
        gate = self.predictive_gate_probs(latent, context_data)
        component_ll = self.component_log_likelihood(query_data)

        if self.objective == "marginal":
            joint_logits = gate.clamp_min(1e-12).log().unsqueeze(1) + component_ll
            marginal_nll = -torch.logsumexp(joint_logits, dim=-1).mean()
            if self.distillation_weight and encoded_query is not None:
                target = F.softmax(joint_logits.detach(), dim=-1)
                local_log_probs = F.log_softmax(encoded_query["local_logits"], dim=-1)
                distillation = (
                    target * (target.clamp_min(1e-12).log() - local_log_probs)
                ).sum(-1).mean()
                marginal_nll = marginal_nll + self.distillation_weight * distillation
            if include_context_distillation and self.contextual_distillation_weight > 0.0:
                marginal_nll = marginal_nll + (
                    self.contextual_distillation_weight
                    * self.contextual_distillation_loss_from_target(latent, gate)
                )
            return marginal_nll

        # The context/query benchmark uses the exact marginal objective.  Keep a
        # faithful-ELBO fallback for API completeness without changing its local
        # assignment term.
        if self.objective == "faithful_elbo":
            if encoded_query is None:
                encoded_query = self.encode(query_data, deterministic=True)
            local_log_probs = F.log_softmax(encoded_query["local_logits"], dim=-1)
            local_probs = local_log_probs.exp()
            agreement_kl = (
                local_probs
                * (local_log_probs - gate.clamp_min(1e-12).log().unsqueeze(1))
            ).sum(-1).mean()
            expected_nll = -(encoded_query["assignment"] * component_ll).sum(-1).mean()
            objective = expected_nll + self.agreement_weight * agreement_kl
            if include_context_distillation and self.contextual_distillation_weight > 0.0:
                objective = objective + (
                    self.contextual_distillation_weight
                    * self.contextual_distillation_loss_from_target(latent, gate)
                )
            return objective

        raise ValueError(f"Unknown structured objective: {self.objective}")

    def refined_encoder_input(self, latent: Tensor, data: Tensor) -> Tensor:
        if self.contextual_gate:
            proportions = self.contextual_gate_probs(latent, data) + self.eps
        else:
            proportions = self.posterior_probs(latent, data).mean(dim=1) + self.eps
        return proportions.log()

    def decode(self, latent: Tensor, data: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        prior_logits = self.decode_mixture_logits(latent)
        assignment_logits = prior_logits.unsqueeze(1) + self.component_log_likelihood(data)
        sample = self._assignment_weights(assignment_logits)
        return sample, assignment_logits, prior_logits.unsqueeze(1)

    def loss(self, data: Tensor, latent: Tensor, encoded: Mapping[str, Tensor]) -> Tensor:
        prior_logits = self.decode_mixture_logits(latent)
        prior_log_probs = F.log_softmax(prior_logits, dim=-1)
        component_ll = self.component_log_likelihood(data)

        if self.objective == "faithful_elbo":
            local_log_probs = F.log_softmax(encoded["local_logits"], dim=-1)
            local_probs = local_log_probs.exp()
            agreement_kl = (
                local_probs
                * (local_log_probs - prior_log_probs.unsqueeze(1))
            ).sum(-1).mean()
            expected_nll = -(encoded["assignment"] * component_ll).sum(-1).mean()
            return expected_nll + self.agreement_weight * agreement_kl

        if self.objective == "marginal":
            joint_logits = prior_log_probs.unsqueeze(1) + component_ll
            marginal_nll = -torch.logsumexp(joint_logits, dim=-1).mean()
            if self.distillation_weight:
                target = F.softmax(joint_logits.detach(), dim=-1)
                local_log_probs = F.log_softmax(encoded["local_logits"], dim=-1)
                distillation = (
                    target * (target.clamp_min(1e-12).log() - local_log_probs)
                ).sum(-1).mean()
                marginal_nll = marginal_nll + self.distillation_weight * distillation
            if self.contextual_distillation_weight > 0.0:
                marginal_nll = marginal_nll + (
                    self.contextual_distillation_weight
                    * self.contextual_distillation_loss(latent, data)
                )
            return marginal_nll

        raise ValueError(f"Unknown structured objective: {self.objective}")


class GMMHead(StructuredMixtureHead):
    """Gaussian-mixture head with patient-dependent component proportions."""

    head_type = "GMM"

    def __init__(
        self,
        dim: int,
        dim_latent: int,
        dim_normal: int,
        head_name: str,
        mixture_params: Mapping[str, Any] | None = None,
        cov_eps: float = 1e-1,
        max_n_perturb_factor: int = 2,
        covariance_parameterization: str = "faithful_relu",
        **kwargs: Any,
    ) -> None:
        super().__init__(dim, dim_latent, head_name, **kwargs)
        self.dim_normal = int(dim_normal)
        self.cov_eps = float(cov_eps)
        self.covariance_parameterization = covariance_parameterization
        mixture_params = dict(mixture_params or {})

        loc_value = mixture_params.get("loc", "random")
        if isinstance(loc_value, str) and loc_value == "random":
            locs = torch.empty(dim, dim_normal)
            nn.init.orthogonal_(locs)
        else:
            locs = torch.as_tensor(loc_value, dtype=torch.float32)
        self.mixture_locs = nn.Parameter(locs.clone())

        cov_value = mixture_params.get("log_cov_diag", 0.0)
        if isinstance(cov_value, (float, int)):
            cov_value = torch.full((dim, dim_normal), float(cov_value))
        else:
            cov_value = torch.as_tensor(cov_value, dtype=torch.float32)
        self.mixture_log_covs = nn.Parameter(cov_value.clone())

        rank = max(1, min(dim_normal, max_n_perturb_factor))
        factor_value = mixture_params.get("cov_perturb_factor")
        if factor_value is None:
            factor = torch.randn(dim, dim_normal, rank) * 0.05
        else:
            factor = torch.as_tensor(factor_value, dtype=torch.float32)
        self.mixture_cov_perturb = nn.Parameter(factor.clone())

        self.encoder_classifier = nn.Linear(dim_normal, dim)
        self._build_context_encoder(self.dim_normal)

    def local_assignment_logits(self, data: Tensor) -> Tensor:
        return self.encoder_classifier(data)

    def covariance_parts(self) -> tuple[Tensor, Tensor]:
        if self.covariance_parameterization == "faithful_relu":
            cov_diag = F.relu(self.mixture_log_covs) + self.cov_eps
        elif self.covariance_parameterization == "softplus":
            cov_diag = F.softplus(self.mixture_log_covs) + 1e-4
        else:
            raise ValueError(
                f"Unknown covariance parameterization: {self.covariance_parameterization}"
            )
        return cov_diag, self.mixture_cov_perturb

    def component_distribution(self) -> torch.distributions.Distribution:
        cov_diag, cov_factor = self.covariance_parts()
        return torch.distributions.LowRankMultivariateNormal(
            self.mixture_locs, cov_factor, cov_diag
        )

    def component_log_likelihood(self, data: Tensor) -> Tensor:
        return self.component_distribution().log_prob(data.unsqueeze(-2))


class TopicModelHead(StructuredMixtureHead):
    """Mixture-of-multinomials topic head."""

    head_type = "TopicModel"

    def __init__(
        self,
        dim: int,
        dim_latent: int,
        dim_words: int,
        head_name: str,
        topic_profiles: Any | None = None,
        normalize_profiles: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(dim, dim_latent, head_name, **kwargs)
        self.dim_words = int(dim_words)
        self.normalize_profiles = bool(normalize_profiles)
        if topic_profiles is None:
            profiles = torch.randn(dim_words, dim) * 0.05
        else:
            profiles = torch.as_tensor(topic_profiles, dtype=torch.float32)
        self.topic_profiles_trainable = nn.Parameter(profiles.clone())
        self.encoder_classifier = nn.Linear(dim_words, dim)
        nn.init.ones_(self.encoder_classifier.bias)
        self._build_context_encoder(self.dim_words)

    def context_encoder_input(self, context_data: Tensor) -> Tensor:
        totals = context_data.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return context_data / totals

    def local_assignment_logits(self, data: Tensor) -> Tensor:
        return self.encoder_classifier(data)

    def log_topic_profiles(self) -> Tensor:
        if self.normalize_profiles:
            return F.log_softmax(self.topic_profiles_trainable, dim=0)
        # Faithful bug-compatible path: the archive treats raw values as log p(w|k).
        return self.topic_profiles_trainable

    def topic_profiles(self) -> Tensor:
        return self.log_topic_profiles().exp()

    def component_log_likelihood(self, data: Tensor) -> Tensor:
        return torch.matmul(data, self.log_topic_profiles())



class ZINBMixtureHead(StructuredMixtureHead):
    """Count mixture with NB / gene-ZINB / hierarchical gene-component ZINB.

    One structured view is a tuple of patient-batched tensors:

    * raw UMI counts ``[patients, cells, genes]`` (float tensor of integers);
    * **external**, full-transcriptome size factors ``[patients, cells]``,
      normalized to a reference depth (typical value near 1);
    * integer technical-library indices ``[patients, cells]`` in
      ``0 .. n_libraries-1``. This third item is optional for one library.

    Every library uses the same genes in the same order. No library covariate
    enters the patient mixture gate. An emission offset is mean-centered across
    libraries for identifiability and L2-shrunk to zero. The head assumes a
    fixed number of sampled cells per patient (no padded-cell mask).

    ``inflation`` specifies the *maximum supported* model, while
    :meth:`set_inflation_mode` selects the active nested submodel. The active
    setting is a buffer and survives state_dict/checkpoint round-trips.
    """

    head_type = "ZINBMixture"
    _INFLATION_LEVELS = {"none": 0, "gene": 1, "gene_component": 2}

    def __init__(
        self,
        dim: int,
        dim_latent: int,
        dim_genes: int,
        head_name: str,
        n_libraries: int = 1,
        inflation: str = "gene",
        initial_inflation: str | None = None,
        initial_pi: float = 0.01,
        initial_dispersion: float = 10.0,
        min_dispersion: float = 1e-2,
        max_dispersion: float = 1e4,
        component_inflation_sd: float = 0.35,
        component_inflation_penalty: float = 1.0,
        gene_inflation_prior_strength: float = 0.05,
        gene_inflation_prior_beta: float = 20.0,
        dispersion_prior_strength: float = 0.01,
        batch_effect_sd: float = 0.25,
        batch_effect_penalty: float = 1.0,
        normalize_by_genes: bool = False,
        validate_counts: bool = True,
        **kwargs: Any,
    ) -> None:
        # Exact marginalization over whole-cell mixture components is essential:
        # a nearly hard Gumbel estimator would make ZINB fitting very unstable.
        if kwargs.pop("objective", "marginal") != "marginal":
            raise ValueError("ZINBMixture requires objective='marginal'")
        # Unlike the legacy structured heads, avoid almost-hard Gumbel draws
        # at the default temperature in the local patient-encoder summary.
        kwargs.setdefault("assignment_estimator", "mean")
        super().__init__(dim, dim_latent, head_name, objective="marginal", **kwargs)
        if dim_genes <= 0 or n_libraries <= 0:
            raise ValueError("dim_genes and n_libraries must be positive")
        if inflation not in self._INFLATION_LEVELS:
            raise ValueError("inflation must be 'none', 'gene' or 'gene_component'")
        if not (0 < initial_pi < 1):
            raise ValueError("initial_pi must lie in (0, 1)")
        if not (0 < min_dispersion < initial_dispersion < max_dispersion):
            raise ValueError("dispersion bounds must contain initial_dispersion")
        if component_inflation_sd <= 0 or batch_effect_sd <= 0:
            raise ValueError("regularization standard deviations must be positive")
        if (component_inflation_penalty < 0 or batch_effect_penalty < 0
                or gene_inflation_prior_strength < 0 or dispersion_prior_strength < 0):
            raise ValueError("regularization strengths must be non-negative")
        if gene_inflation_prior_beta < 1:
            raise ValueError("gene_inflation_prior_beta must be >= 1")
        self.dim_genes = int(dim_genes)
        self.n_libraries = int(n_libraries)
        self.max_inflation = inflation
        self.min_dispersion = float(min_dispersion)
        self.max_dispersion = float(max_dispersion)
        self.component_inflation_sd = float(component_inflation_sd)
        self.component_inflation_penalty = float(component_inflation_penalty)
        self.gene_inflation_prior_strength = float(gene_inflation_prior_strength)
        self.gene_inflation_prior_beta = float(gene_inflation_prior_beta)
        self.dispersion_prior_strength = float(dispersion_prior_strength)
        self.initial_dispersion = float(initial_dispersion)
        self.batch_effect_sd = float(batch_effect_sd)
        self.batch_effect_penalty = float(batch_effect_penalty)
        self.normalize_by_genes = bool(normalize_by_genes)
        self.validate_counts = bool(validate_counts)
        self.log_component_mean = nn.Parameter(torch.zeros(dim, dim_genes))
        dispersion_raw = math.log(math.expm1(initial_dispersion))
        self.raw_dispersion = nn.Parameter(torch.full((dim_genes,), dispersion_raw))
        pi_logits = math.log(initial_pi / (1.0 - initial_pi))
        self.gene_inflation_logits = (
            nn.Parameter(torch.full((dim_genes,), pi_logits))
            if self._INFLATION_LEVELS[inflation] >= 1 else None
        )
        self.component_inflation_deviations = (
            nn.Parameter(torch.zeros(dim, dim_genes))
            if inflation == "gene_component" else None
        )
        self.batch_log_effects = (
            nn.Parameter(torch.zeros(n_libraries, dim_genes))
            if n_libraries > 1 else None
        )
        self.encoder_classifier = nn.Linear(dim_genes, dim)
        nn.init.zeros_(self.encoder_classifier.bias)
        self._build_context_encoder(dim_genes)
        self.register_buffer("_inflation_level", torch.zeros((), dtype=torch.long))
        self.set_inflation_mode(initial_inflation or inflation)

    @property
    def inflation_mode(self) -> str:
        level = int(self._inflation_level.item())
        return next(name for name, number in self._INFLATION_LEVELS.items() if number == level)

    def set_inflation_mode(self, mode: str) -> None:
        """Switch from NB warm start to gene or hierarchical ZINB in-place."""
        if mode not in self._INFLATION_LEVELS:
            raise ValueError(f"Unknown inflation mode: {mode}")
        if self._INFLATION_LEVELS[mode] > self._INFLATION_LEVELS[self.max_inflation]:
            raise ValueError(f"{mode} exceeds configured maximum {self.max_inflation}")
        self._inflation_level.fill_(self._INFLATION_LEVELS[mode])

    def unpack_data(self, data: Any) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(data, (tuple, list)) or len(data) not in (2, 3):
            raise ValueError(
                "ZINBMixture data must be (raw_counts, full_count_size_factors, library_ids)"
            )
        counts, size_factors = data[:2]
        if not isinstance(counts, Tensor) or counts.ndim != 3 or counts.shape[-1] != self.dim_genes:
            raise ValueError("counts must have shape [patients, cells, dim_genes]")
        if counts.shape[1] == 0:
            raise ValueError("ZINBMixture requires at least one actual cell per patient")
        if not counts.is_floating_point():
            counts = counts.to(torch.float32)
        if not isinstance(size_factors, Tensor):
            raise TypeError("size_factors must be a Tensor")
        if size_factors.ndim == 3 and size_factors.shape[-1] == 1:
            size_factors = size_factors[..., 0]
        if size_factors.shape != counts.shape[:2]:
            raise ValueError("size_factors must have shape [patients, cells]")
        size_factors = size_factors.to(device=counts.device, dtype=counts.dtype)
        if len(data) == 3:
            library_ids = data[2]
            if not isinstance(library_ids, Tensor) or library_ids.shape != counts.shape[:2]:
                raise ValueError("library_ids must have shape [patients, cells]")
            if library_ids.is_floating_point():
                if not torch.all(library_ids == library_ids.round()):
                    raise ValueError("library_ids must contain integer indices")
            library_ids = library_ids.to(device=counts.device, dtype=torch.long)
        else:
            if self.n_libraries != 1:
                raise ValueError("library_ids are required with n_libraries > 1")
            library_ids = torch.zeros(counts.shape[:2], dtype=torch.long, device=counts.device)
        if self.validate_counts:
            if not torch.all(torch.isfinite(counts)) or torch.any(counts < 0):
                raise ValueError("counts must be finite and non-negative")
            if not torch.all(counts == counts.round()):
                raise ValueError("counts must be *raw integer UMI counts*, not normalized data")
        if not torch.all(torch.isfinite(size_factors)) or torch.any(size_factors <= 0):
            raise ValueError("size_factors must be finite and strictly positive")
        if torch.any(library_ids < 0) or torch.any(library_ids >= self.n_libraries):
            raise ValueError("library_ids outside the configured library range")
        return counts, size_factors, library_ids

    def context_encoder_input(self, context_data: Any) -> Tensor:
        counts, size_factors, library_ids = self.unpack_data(context_data)
        # Emission-estimated gene/library technical shifts must also be removed
        # from the *local encoder* summary, otherwise the patient posterior can
        # absorb batch despite a corrected emission model. Detach offsets here
        # so the encoder cannot train technical offsets simply to help itself.
        correction = self.batch_effects()[library_ids].detach().clamp(-8.0, 8.0)
        corrected_rate = counts / size_factors.unsqueeze(-1) * torch.exp(-correction)
        return torch.log1p(corrected_rate)

    def local_assignment_logits(self, data: Any) -> Tensor:
        return self.encoder_classifier(self.context_encoder_input(data))

    def dispersion(self) -> Tensor:
        return F.softplus(self.raw_dispersion).clamp(self.min_dispersion, self.max_dispersion)

    def batch_effects(self) -> Tensor:
        if self.batch_log_effects is None:
            return self.log_component_mean.new_zeros((1, self.dim_genes))
        return self.batch_log_effects - self.batch_log_effects.mean(dim=0, keepdim=True)

    def inflation_logits(self) -> Tensor | None:
        if self.inflation_mode == "none":
            return None
        assert self.gene_inflation_logits is not None
        logits = self.gene_inflation_logits.unsqueeze(0).expand(self.dim, -1)
        if self.inflation_mode == "gene_component":
            assert self.component_inflation_deviations is not None
            # Across-component centering removes a redundant additive intercept.
            deviations = self.component_inflation_deviations
            logits = logits + deviations - deviations.mean(dim=0, keepdim=True)
        return logits

    def component_log_likelihood(self, data: Any) -> Tensor:
        """Exact per-cell log p(count-vector | component), [N,C,K]."""
        counts, size_factors, library_ids = self.unpack_data(data)
        x = counts.unsqueeze(-2)  # [patients, cells, 1, genes]
        log_mu = (
            size_factors.log().unsqueeze(-1).unsqueeze(-1)
            + self.log_component_mean[None, None, :, :]
            + self.batch_effects()[library_ids].unsqueeze(-2)
        )
        # Bound exponentiation only, preserving finite values at extreme depths.
        mu = torch.exp(log_mu.clamp(-18.0, 18.0))
        theta = self.dispersion()
        log_theta = theta.log()
        log_theta_mu = torch.log(theta + mu)
        log_nb = (
            torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1.0)
            + theta * (log_theta - log_theta_mu)
            + x * (log_mu.clamp(-18.0, 18.0) - log_theta_mu)
        )
        inflation = self.inflation_logits()
        if inflation is not None:
            logits = inflation[None, None, :, :]
            log_pi = F.logsigmoid(logits)
            log_no_pi = F.logsigmoid(-logits)
            log_nb_zero = theta * (log_theta - log_theta_mu)
            log_zero = torch.logaddexp(log_pi, log_no_pi + log_nb_zero)
            log_nb = torch.where(x == 0, log_zero, log_no_pi + log_nb)
        result = log_nb.sum(dim=-1)
        if self.normalize_by_genes:
            result = result / self.dim_genes
        return result

    def regularization_loss(self) -> Tensor:
        result = self.log_component_mean.new_zeros(())
        # Weak log-dispersion shrinkage mitigates competition between low theta
        # and large pi when observed zeros do not identify them separately.
        if self.dispersion_prior_strength > 0:
            result = result + (
                self.dispersion_prior_strength
                * (self.dispersion().log() - math.log(self.initial_dispersion)).square().mean()
            )
        # A weak Beta(1, beta) prior discourages unjustified extra zeros.
        if self.inflation_mode != "none" and self.gene_inflation_prior_strength > 0:
            assert self.gene_inflation_logits is not None
            result = result + (
                self.gene_inflation_prior_strength * (self.gene_inflation_prior_beta - 1)
                * F.softplus(self.gene_inflation_logits).mean()
            )
        if self.batch_log_effects is not None and self.batch_effect_penalty > 0:
            result = result + (
                self.batch_effect_penalty / (2 * self.batch_effect_sd**2)
                * self.batch_effects().square().mean()
            )
        if self.inflation_mode == "gene_component" and self.component_inflation_penalty > 0:
            deviations = self.component_inflation_deviations
            assert deviations is not None
            centered = deviations - deviations.mean(dim=0, keepdim=True)
            result = result + (
                self.component_inflation_penalty / (2 * self.component_inflation_sd**2)
                * centered.square().mean()
            )
        return result

    def loss(self, data: Any, latent: Tensor, encoded: Mapping[str, Tensor]) -> Tensor:
        return super().loss(data, latent, encoded) + self.regularization_loss()

    @torch.no_grad()
    def initialize_from_counts(
        self,
        data: Any,
        assignments: Tensor | None = None,
        *,
        max_init_cells: int = 5000,
        kmeans_iterations: int = 10,
        seed: int = 0,
        prior_exposure: float = 2.0,
        gate_from_assignments: bool = True,
    ) -> dict[str, Any]:
        """Observed training-count initialization (no held-out data or labels).

        Optionally accept training-only cell responsibilities ``[N,C,K]``.
        Otherwise run a small, seeded torch k-means on subsampled normalized
        log-counts. The resulting component mean profiles are exposure-weighted.
        NB dispersion remains at its configured stable initial value; refine it
        during training, rather than fitting a poorly identified ZINB moment model.
        """
        counts, sf, _ = self.unpack_data(data)
        x = counts.reshape(-1, self.dim_genes)
        exposure = sf.reshape(-1)
        n_cells = int(x.shape[0])
        if n_cells < self.dim:
            raise ValueError("Need at least as many training cells as components")
        if max_init_cells < self.dim or kmeans_iterations < 1 or prior_exposure <= 0:
            raise ValueError("max_init_cells>=components, iterations>=1, prior_exposure>0 required")
        generator = torch.Generator(device=x.device).manual_seed(seed)
        ids = torch.randperm(n_cells, generator=generator, device=x.device)[:max_init_cells]
        sample_x = x[ids]
        sample_sf = exposure[ids]
        features = torch.log1p(sample_x / sample_sf[:, None])
        if assignments is None:
            centers = features[torch.randperm(len(ids), generator=generator, device=x.device)[:self.dim]].clone()
            for _ in range(kmeans_iterations):
                distances = torch.cdist(features[None], centers[None]).squeeze(0)
                labels = distances.argmin(dim=-1)
                for k in range(self.dim):
                    selected = labels == k
                    if selected.any():
                        centers[k] = features[selected].mean(dim=0)
            responsibilities = F.one_hot(labels, self.dim).to(sample_x.dtype)
        else:
            if assignments.shape != (*counts.shape[:2], self.dim):
                raise ValueError("assignments must be [patients, cells, components]")
            responsibilities = assignments.reshape(-1, self.dim)[ids].to(sample_x.dtype)
            if (not torch.all(torch.isfinite(responsibilities))
                    or torch.any(responsibilities < 0)
                    or torch.any(responsibilities.sum(-1) <= 0)):
                raise ValueError("assignments must be finite, nonnegative and have positive row sums")
            responsibilities = responsibilities / responsibilities.sum(-1, keepdim=True)
        global_rate = sample_x.sum(0) / sample_sf.sum().clamp_min(1e-8)
        numerators = responsibilities.T @ sample_x + prior_exposure * global_rate[None]
        denominators = responsibilities.T @ sample_sf + prior_exposure
        rate = (numerators / denominators[:, None]).clamp_min(1e-7)
        self.log_component_mean.copy_(rate.log())
        mixture_probs = (responsibilities.sum(0) + 1.0) / (len(ids) + self.dim)
        if gate_from_assignments:
            self.mixture_logits.weight.zero_()
            self.mixture_logits.bias.copy_(mixture_probs.log())
        # Set classifier directions near data-driven centers without large logits.
        directions = features.new_zeros((self.dim, self.dim_genes))
        for k in range(self.dim):
            weight = responsibilities[:, k].sum()
            directions[k] = (responsibilities[:, k, None] * features).sum(0) / weight.clamp_min(1.0)
        directions -= directions.mean(0, keepdim=True)
        directions /= directions.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.encoder_classifier.weight.copy_(directions * 0.1)
        self.encoder_classifier.bias.copy_(mixture_probs.log())
        return {"cells_used": int(len(ids)), "component_probs": mixture_probs.cpu().tolist(),
                "method": "supplied_responsibilities" if assignments is not None else "kmeans"}

    @torch.no_grad()
    def sample_counts(self, latent: Tensor, size_factors: Tensor, library_ids: Tensor) -> Tensor:
        """Sample raw-like UMI counts for given patient/cell covariates."""
        if size_factors.ndim != 2 or size_factors.shape[0] != latent.shape[0]:
            raise ValueError("size_factors must be [patients,cells]")
        if library_ids.shape != size_factors.shape:
            raise ValueError("library_ids must match size_factors")
        if (torch.any(library_ids < 0) or torch.any(library_ids >= self.n_libraries)
                or torch.any(library_ids != library_ids.long())):
            raise ValueError("library_ids must be integers in the configured library range")
        if not torch.all(torch.isfinite(size_factors)) or torch.any(size_factors <= 0):
            raise ValueError("size_factors must be finite and strictly positive")
        gate = self.prior_gate_probs(latent)
        n, c = size_factors.shape
        choices = torch.multinomial(gate, c, replacement=True)
        effect = self.batch_effects()[library_ids.long()]
        log_mu = size_factors.log()[..., None] + self.log_component_mean[choices] + effect
        mu = log_mu.clamp(-18, 18).exp()
        theta = self.dispersion()
        rate = torch.distributions.Gamma(theta, theta / mu).sample()
        sampled = torch.poisson(rate)
        logits = self.inflation_logits()
        if logits is not None:
            p = torch.sigmoid(logits[choices])
            sampled = sampled.masked_fill(torch.rand_like(p) < p, 0.)
        return sampled

def build_head(config: Mapping[str, Any], dim_latent: int) -> BaseHead:
    """Build a head from an archived-style configuration dictionary."""
    values = copy.deepcopy(dict(config))
    head_type = values.pop("head_type")
    values.pop("dim_latent", None)
    # The supplied notebook uses this name, although it is absent from the archive.
    if head_type == "GMM_prop":
        head_type = "GMM"
    mapping = {
        "Bernoulli": BernoulliHead,
        "Multinomial": MultinomialHead,
        "MultiNormal": MultiNormalHead,
        "GMM": GMMHead,
        "TopicModel": TopicModelHead,
        "ZINBMixture": ZINBMixtureHead,
    }
    try:
        cls = mapping[head_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported head type: {head_type}") from exc
    # Keras serialization details cannot be reused by torch.nn; default linear maps
    # are the faithful equivalent for every configuration in the supplied notebook.
    values.pop("layer_configs", None)
    values.pop("l1_scale", None)
    return cls(dim_latent=dim_latent, **values)
