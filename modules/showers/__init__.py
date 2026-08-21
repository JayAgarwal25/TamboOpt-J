"""Shower corpora and the detector response kernel.

`kernel` is the ground truth every training label comes from; `tau` loads the
tau primaries; `generate` bridges to the AllShowers generator in the sibling
TAMBO-opt repo (see its docstring — it injects sys.path on import).
"""

from .kernel import GetCounts_planeaware
from .tau import load_tau_primaries

__all__ = ["GetCounts_planeaware", "load_tau_primaries"]
