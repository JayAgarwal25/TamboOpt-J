"""Shared pieces for the evaluators here.

`true_utility` is the scoring module every evaluator compares against. It lives
in the plots tree, whose layout is not the same in every checkout, so the search
happens once here instead of five scripts each naming a path.

`_corpus_paths` and `build_primaries` live here rather than in that module
because they are specific to evaluating a corpus other than the training one.
`primary.pt` is row-aligned to the corpus Step 1 consumed, so slicing its first B
rows for a different corpus pairs clouds with the wrong events' ground truth,
which is worse than leakage. These rebuild the encoding from corpus metadata plus
the decay-position sidecar instead.
"""

import importlib.util as _ilu
import os

import numpy as np
import torch

from modules_v6.constants import HELDOUT_SHOWER_CACHE_PATH, HELDOUT_POSITIONS_PATH
from modules_v6.fnn_surrogate import encode_primary
from modules_v6.fnn_surrogate_ne import _load_positions_sidecar

# Where the scoring module sits, newest layout first. Both are real: the plots
# tree was regrouped by subject, and older checkouts still have the flat name.
_TRUE_UTILITY_CANDIDATES = (
    ("plots", "layouts", "true_utility.py"),
    ("plots", "eval_true_utility.py"),
)


def load_true_utility(root):
    """Import the scoring module by path, wherever it lives in this checkout."""
    for parts in _TRUE_UTILITY_CANDIDATES:
        path = os.path.join(root, *parts)
        if os.path.exists(path):
            spec = _ilu.spec_from_file_location("_true_utility", path)
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit(
        "[eval] no scoring module found; looked for "
        + " and ".join(os.path.join(*p) for p in _TRUE_UTILITY_CANDIDATES)
        + " under " + root)


def _corpus_paths(corpus_override):
    """(corpus, positions) paths: the held-out pair by default, or an override
    plus its own `<corpus>_positions.pt` sidecar, the naming rule Step 1 uses.

    Held-out by default because every pipeline stage reads the training corpus:
    the surrogates split it internally and the stage-4 sweep does not split at
    all, so scoring a layout there scores it on the events that produced it.
    """
    if not corpus_override:
        return HELDOUT_SHOWER_CACHE_PATH, HELDOUT_POSITIONS_PATH
    return corpus_override, os.path.splitext(corpus_override)[0] + "_positions.pt"


def build_primaries(corpus_path, B, mountain):
    """Encode the first `B` events' primaries straight from a shower corpus.

    Mirrors the Step-1 encoding so a held-out corpus stays byte-comparable with
    the training path; verified at max abs diff 0 across all 8 columns.
    """
    import showerdata

    meta = showerdata.load_inc_particles(corpus_path)
    keep = np.arange(0, B)          # electron block; the muon row shares the primary
    dirs = torch.as_tensor(meta.directions[keep], dtype=torch.float32)
    energs = torch.as_tensor(meta.energies[keep], dtype=torch.float32)
    pdg = torch.as_tensor(meta.pdg[keep], dtype=torch.long)
    positions = _load_positions_sidecar(corpus_path, keep)
    array_center = torch.as_tensor(mountain.centroids_ENU,
                                   dtype=torch.float32).mean(dim=0)
    return encode_primary(dirs, energs, pdg, positions, array_center)
