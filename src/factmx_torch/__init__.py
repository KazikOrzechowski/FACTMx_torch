"""FACTMx: multimodal latent-variable models implemented in PyTorch."""

from ._version import __version__
from .encoders import AttentionEncoder, LinearEncoder, MeanEncoder, build_encoder
from .heads import (
    BaseHead,
    BernoulliHead,
    GMMHead,
    MultiNormalHead,
    MultinomialHead,
    StructuredMixtureHead,
    TopicModelHead,
    build_head,
)
from .initialization import initialize_global_covariance
from .missing_view import MissingViewFACTMx
from .model import FACTMxModel
from .training import (
    EncoderGradientBalancer,
    StructuredFirstBalanceConfig,
    combine_partitioned_objective,
    encoder_gradient_norm_pair,
    kl_beta_for_epoch,
    simple_ramp_multiplier,
    split_scaled_loss_components,
)

__all__ = [
    "__version__",
    "FACTMxModel",
    "MissingViewFACTMx",
    "initialize_global_covariance",
    "BaseHead",
    "BernoulliHead",
    "MultinomialHead",
    "MultiNormalHead",
    "StructuredMixtureHead",
    "GMMHead",
    "TopicModelHead",
    "build_head",
    "LinearEncoder",
    "MeanEncoder",
    "AttentionEncoder",
    "build_encoder",
    "EncoderGradientBalancer",
    "StructuredFirstBalanceConfig",
    "combine_partitioned_objective",
    "encoder_gradient_norm_pair",
    "kl_beta_for_epoch",
    "simple_ramp_multiplier",
    "split_scaled_loss_components",
]
