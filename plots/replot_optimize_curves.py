"""Redraw `optimize_curves.png` + `utility_components.png` from a finished run.

Stage 4 writes both PNGs at the end of the run, so a change to the plotting
code otherwise means re-optimizing. Everything the two figures need is already
in `optimize_log.json` — the per-step U logs and the W-smoothed gradient cosine
series — EXCEPT the raw per-step gradients, which are only ever held in memory
(the L-BFGS resume checkpoint that carried them is deleted on success). So
panel 2 is redrawn from the stored smoothed series and loses its faint raw
underlay; panel 1 is exact.

    python plots/replot_optimize_curves.py                     # every scheme dir
    python plots/replot_optimize_curves.py --run-dir DIR_A DIR_B
"""
import argparse
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(_HERE))

import modules  # noqa: F401 — package import; keeps modules on the path
import plots.opt_plotting as _plt
from modules.constants import OPT_FOLDER

# Mirrors GRAD_COS_WINDOW in 04_optimize_lbfgs_ensemble.py; importing that
# module just for one int would drag in the whole stage-4 setup. Only a label
# here — the series in the log were already smoothed with the run's own value.
GRAD_COS_WINDOW = 10


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opt-suffix", default="_full_corpus")
    ap.add_argument("--run-dir", nargs="+", help="scheme dir(s); overrides --opt-suffix")
    args = ap.parse_args()

    dirs = args.run_dir or sorted(
        d for d in glob.glob(OPT_FOLDER + "_lbfgs_ensemble" + args.opt_suffix + "_*")
        if os.path.exists(os.path.join(d, "optimize_log.json")))
    if not dirs:
        raise SystemExit(f"no optimize_log.json under {OPT_FOLDER}"
                         f"_lbfgs_ensemble{args.opt_suffix}_*")

    for d in dirs:
        with open(os.path.join(d, "optimize_log.json")) as f:
            log = json.load(f)
        print(f"[replot] {os.path.basename(d)}  scheme={log['scheme']}  "
              f"best U={log['best_U']:+.3f}")
        _plt.plot_curves_lbfgs(
            log["adam_logs"], log["lbfgs_logs"], None, None,
            os.path.join(d, "optimize_curves.png"),
            grad_cos_window=log.get("config", {}).get("grad_cos_window",
                                                      GRAD_COS_WINDOW),
            cos_smoothed=log.get("grad_cos_consecutive"),
            scoring_logs=log.get("scoring_logs"))
        _plt.plot_components_lbfgs(
            log["adam_logs"], log["lbfgs_logs"],
            os.path.join(d, "utility_components.png"))


if __name__ == "__main__":
    main()
