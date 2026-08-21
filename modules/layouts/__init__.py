"""Detector layout generation.

`strategies` are the named generators a dataset build draws from; `learnable`
is the differentiable (x, y) parameterization the Step-4 optimizers descend on.
"""

from .strategies import (ACTIVE_STRATEGIES, layout_center_gaussian, layout_grid,
                         layout_latin_hypercube, layout_rings,
                         layout_uniform_random)
from .learnable import LearnableXY

__all__ = ["ACTIVE_STRATEGIES", "layout_grid", "layout_center_gaussian",
           "layout_rings", "layout_uniform_random", "layout_latin_hypercube",
           "LearnableXY"]
