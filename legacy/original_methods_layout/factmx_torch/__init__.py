"""Current PyTorch FACTMx implementation used by the simulation benchmark."""

from .model import FACTMxModel
from .missing_view import MissingViewFACTMx
from .heads import build_head
from .encoders import build_encoder
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
    "FACTMxModel",
    "MissingViewFACTMx",
    "build_head",
    "build_encoder",
    "EncoderGradientBalancer",
    "StructuredFirstBalanceConfig",
    "combine_partitioned_objective",
    "encoder_gradient_norm_pair",
    "kl_beta_for_epoch",
    "simple_ramp_multiplier",
    "split_scaled_loss_components",
]
