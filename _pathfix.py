"""Single source of truth for locating the repo root on sys.path.

Any entry-point script one level below repo root (scripts/*.py, etc.) should do:

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _pathfix import V6_ROOT  # noqa: F401 — idempotent, registers repo root once

Scripts directly at repo root (or already-existing modules/plots code,
which live one level down but were never moved) do not need this at all.
"""
import os
import sys

V6_ROOT = os.path.dirname(os.path.abspath(__file__))
if V6_ROOT not in sys.path:
    sys.path.insert(0, V6_ROOT)

# Keep every .pyc in one tree instead of scattering __pycache__/ dirs through
# the source. The env var is what actually matters (it is read at interpreter
# startup, so it also covers modules imported before this file); this assignment
# is the fallback for interactive sessions and notebooks that never set it.
PYCACHE_DIR = os.path.join(V6_ROOT, ".pycache")
os.environ.setdefault("PYTHONPYCACHEPREFIX", PYCACHE_DIR)
if not sys.pycache_prefix:
    sys.pycache_prefix = PYCACHE_DIR
