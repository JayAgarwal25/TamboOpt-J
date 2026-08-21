"""Where these analyses read their inputs from and write their results to.

Every script here used to name an absolute path to a layout produced by a
particular stage-4 run. Those paths pinned the analyses to layouts optimized
against an older mountain, so re-running after the geometry changed would have
snapped stale coordinates onto the new mesh and reported the result as current.

Layouts are therefore named here, once. Override the base directory with
$TAMBO_LAYOUT_BASE; individual scripts that take a --layout argument still win
over both.
"""

import os

# Default: `v6_runs/` sits beside the repository checkout. Derived rather than
# written out, so no account name is baked into the file.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../<repo>/<v6 dir>
_RUNS = os.path.join(os.path.dirname(os.path.dirname(_REPO)), "v6_runs")
RUNS = _RUNS   # public: stage-4 run outputs root

BASE = os.environ.get("TAMBO_LAYOUT_BASE",
                      os.path.join(_RUNS, "phase3_2026-08", "run_04_opt"))

# The two schemes stage 4 currently produces. PRIMARY is the one a single-layout
# analysis should use; SECONDARY exists so cross-layout checks have a second,
# independently initialised optimum to compare against.
PRIMARY_NAME   = "opt_lbfgs_ensemble_p3_grid"
SECONDARY_NAME = "opt_lbfgs_ensemble_p3_center"


def layout_path(name: str, base: str = None) -> str:
    """Absolute path to one layout_best.pt, checked to exist."""
    p = os.path.join(base or BASE, name, "layout_best.pt")
    if not os.path.exists(p):
        raise SystemExit(
            f"[layouts] no layout at {p}\n"
            f"  Set $TAMBO_LAYOUT_BASE to the directory holding the stage-4 "
            f"output folders, or pass an explicit --layout where the script "
            f"supports one.")
    return p


def primary(base: str = None) -> str:
    return layout_path(PRIMARY_NAME, base)


def secondary(base: str = None) -> str:
    return layout_path(SECONDARY_NAME, base)


# ── Results ──────────────────────────────────────────────────────────────────
# Results used to be written next to the code, so every run left the repository
# dirty and the analysis output was versioned alongside the analysis. They go
# beside the other run outputs instead. Override with $TAMBO_LANDSCAPE_OUT.
_DEFAULT_OUT = os.path.join(_RUNS, "landscape_analysis")


def results_dir(create: bool = True) -> str:
    d = os.environ.get("TAMBO_LANDSCAPE_OUT", _DEFAULT_OUT)
    if create:
        os.makedirs(d, exist_ok=True)
    return d
