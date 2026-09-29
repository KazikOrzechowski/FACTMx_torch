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
        if n_steps <= 0 or context_data.shape[1] == 0:
            return prior

        component_ll = self.component_log_likelihood(context_data)
        gate = prior
        n_context = float(context_data.shape[1])
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
