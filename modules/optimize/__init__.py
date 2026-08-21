"""The Step-4 objective: what the layout optimizers maximise.

`objective` assembles the surrogate + recon forward pass into U(x, y) and holds
the term weights and thresholds that tune it; `utility` holds the individual U
terms it sums.
"""

from .objective import (CAP_E, CAP_PHI, CAP_THETA, DISTINCT_SCALE,
                        LAYOUT_THRESHOLD, OFFMESH_PENALTY_W, PARTICLE_SCALE,
                        PENALTY_LINEAR_AT, PENALTY_ONSET_FRAC,
                        RECONSTRUCT_THRESHOLD, TAU_LAYOUT, TAU_RECONSTRUCT,
                        W_DIV, W_E, W_PHI, W_PR, W_THETA,
                        activation_of_xy, align_to_reference, assign,
                        consecutive_cos_distance, load_models, offmesh_penalty,
                        overlap_multiplicity, primary_to_physical_labels,
                        utility_of_xy)
from .utility import U_E, U_PR, U_angle, reconstructability

__all__ = [
    # callables
    "utility_of_xy", "activation_of_xy", "offmesh_penalty",
    "overlap_multiplicity", "primary_to_physical_labels", "load_models",
    "assign", "align_to_reference", "consecutive_cos_distance",
    "reconstructability", "U_PR", "U_E", "U_angle",
    # objective tuning constants (the 04 scripts read and print these)
    "W_THETA", "W_PHI", "W_E", "W_PR", "W_DIV",
    "CAP_THETA", "CAP_PHI", "CAP_E",
    "LAYOUT_THRESHOLD", "RECONSTRUCT_THRESHOLD", "TAU_LAYOUT", "TAU_RECONSTRUCT",
    "OFFMESH_PENALTY_W", "PENALTY_ONSET_FRAC", "PENALTY_LINEAR_AT",
    "PARTICLE_SCALE", "DISTINCT_SCALE",
]
