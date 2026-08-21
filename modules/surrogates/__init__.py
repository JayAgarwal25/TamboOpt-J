"""The neural networks: shower-response surrogates and the reconstruction net.

`fnn` holds the primary encoding shared by every stage plus the pre-DeepSets
FNNSurrogate; `deepsets` is the live per-species surrogate, `dual` combines the
e/mu pair into one physical event, and `recon` maps a response back to primary
labels.
"""

from .deepsets import DeepSetsSurrogate, build_surrogate_from_ckpt
from .dual import DualSpeciesSurrogate, combine_species_outputs, load_dual_surrogate
from .fnn import FNNSurrogate, compute_normalization, encode_primary
from .recon import DeepSetsRecon, Reconstruction, build_recon_from_ckpt

__all__ = ["encode_primary", "compute_normalization", "FNNSurrogate",
           "DeepSetsSurrogate", "build_surrogate_from_ckpt",
           "DualSpeciesSurrogate", "combine_species_outputs", "load_dual_surrogate",
           "Reconstruction", "DeepSetsRecon", "build_recon_from_ckpt"]
