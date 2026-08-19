"""Per-event composite utility vs. primary tau energy, for a saved optimized layout.

`opt_core.utility_of_xy` returns a single batch-averaged U — that's what the
optimizer selects on, but it hides whether the layout helps uniformly across the
energy range or is buying its average from one end. This script re-derives the
SAME composite (identical weights, caps, thresholds — `modules_v6/opt_core.py`'s
W_THETA/W_PHI/W_E/W_DIV, CAP_*, LAYOUT_THRESHOLD, RECONSTRUCT_THRESHOLD,
TAU_LAYOUT, TAU_RECONSTRUCT) but stops short of the final `torch.mean` inside
`U_E`/`U_angle`, keeping one U value per event instead. Since mean is linear,
`mean(per_event_U) == utility_of_xy(...)`'s scalar U exactly — checked at
runtime against the unmodified function as a correctness guard, not just an
assertion in a comment.

Reuses `plots/eval_true_utility.py`'s event loader (held-out corpus, unseen by
Steps 1-4) and model loader, so the numbers here are on the same footing as
that script's baseline/optimized comparison — just broken out per event and
plotted against each event's true tau energy instead of collapsed to one number.

Output: `utility_vs_energy.png` next to each `layout_best.pt` passed via
`--run-dir` (mirrors `plots/replot_optimize_curves.py`'s convention of writing
back into the run's own directory).

    cd TambOpt
    python plots/eval_utility_vs_energy.py --run-dir "RUN_DIR_A" "RUN_DIR_B"
    python plots/eval_utility_vs_energy.py --run-dir "RUN_DIR" --n-events 20000
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))          # plots/
_V6 = os.path.dirname(_HERE)
for _p in (_V6, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import modules_v6  # noqa: F401 — sys.path injection for v3 + v4
from modules_v6.tr_geometry_ne import project_to_mountain_ne
from modules_v6.legacy_core.tr_geometry import load_tr_mountain
from modules_v6.opt_core import (
    primary_to_physical_labels, load_models,
    W_THETA, W_PHI, W_E, W_DIV, CAP_THETA, CAP_PHI, CAP_E,
    LAYOUT_THRESHOLD, RECONSTRUCT_THRESHOLD, TAU_LAYOUT, TAU_RECONSTRUCT,
)
from modules_v6.constants import (
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
)
# modules_v6/__init__ injected the v3 (`modules`) path on package import.
from modules_v6.legacy_core.utility_functions import reconstructability, _soft_cap
from eval_true_utility import load_events, utility_of_xy


# log10(1e9): GeV -> eV shift applied to the plotted x-values (not a label
# annotation) -- see score_run's comment where it's used.
GEV_TO_EV_LOG10 = 9.0


def _snap(mountain, e, n):
    e, n = project_to_mountain_ne(mountain, e.float().reshape(-1), n.float().reshape(-1))
    return e.float(), n.float()


def load_layout(path, mountain):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    e, n = (raw["x"], raw["y"]) if isinstance(raw, dict) else (raw[:, 0], raw[:, 1])
    return _snap(mountain, e, n)


@torch.no_grad()
def utility_per_event(x_det, y_det, primary_batch, fnn, recon):
    """(U_i, E_true_gev, r) — per-event decomposition of `utility_of_xy`.

    Mirrors that function line for line up to the point where `U_E`/`U_angle`
    collapse their `r * inv_err` term with `torch.mean`; here that reduction is
    simply omitted, so every return is (B,) instead of a scalar. Same weights,
    same soft caps, same reconstructability thresholds — nothing about the
    objective changes, only whether the batch dimension survives.
    """
    B = primary_batch.shape[0]
    xy_per_det = torch.stack([x_det, y_det], dim=-1)
    xy_batch = xy_per_det.unsqueeze(0).expand(B, -1, -1)

    pred_ET = fnn(primary_batch, xy_batch)
    E_pred_det = pred_ET[..., 0]
    T_pred_det = pred_ET[..., 1]
    recon_feats = torch.stack(
        [xy_batch[..., 0], xy_batch[..., 1], E_pred_det, T_pred_det], dim=-1)
    pred = recon(recon_feats)
    E_pred_phys, theta_pred, phi_pred = primary_to_physical_labels(pred)
    E_pred_phys = E_pred_phys.clamp(min=1.0)
    E_true, theta_true, phi_true = primary_to_physical_labels(primary_batch)

    r = reconstructability(
        torch.expm1(E_pred_det), layout_threshold=LAYOUT_THRESHOLD,
        tau_layout=TAU_LAYOUT, reconstruct_threshold=RECONSTRUCT_THRESHOLD,
        tau_reconstruct=TAU_RECONSTRUCT)

    inv_theta = _soft_cap(1.0 / ((theta_pred - theta_true) ** 2 + .001), CAP_THETA)
    inv_phi   = _soft_cap(1.0 / ((phi_pred   - phi_true)   ** 2 + .001), CAP_PHI)
    inv_e     = _soft_cap(1.0 / ((torch.log10(E_pred_phys)
                                  - torch.log10(E_true)) ** 2 + .01), CAP_E)

    U_i = (W_THETA * r * inv_theta + W_PHI * r * inv_phi
          + W_E * r * inv_e) / W_DIV
    return U_i, E_true, r


def _plot(ax, log_e, U, title, n_bins=40, min_count=20):
    """Hexbin density of per-event U over log10(E_tau), linear colour clipped
    at p95 (the convention `plots/02_plot_nn_target_vs_pred.py::_scatter` uses
    for the same reason: counts span orders of magnitude, so a plain linear
    scale lets the single densest cell own the whole ramp). A binned mean +/-
    std curve on top answers the actual question -- does per-event utility
    trend with energy, and how much does it vary at fixed energy."""
    lo, hi = float(log_e.min()), float(log_e.max())
    y_hi = float(np.percentile(U, 99.5))
    hb = ax.hexbin(log_e, np.clip(U, None, y_hi), gridsize=70, cmap="viridis",
                   mincnt=1, extent=(lo, hi, 0.0, y_hi))
    counts = np.asarray(hb.get_array())
    if counts.size:
        hb.set_clim(0.0, float(np.percentile(counts, 95.0)))
    plt.colorbar(hb, ax=ax, label="count  (linear, clipped at p95)",
                 pad=0.02, fraction=0.046)

    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(log_e, edges[1:-1]), 0, n_bins - 1)
    ctr, mean, sd = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() < min_count:
            continue
        ctr.append(0.5 * (edges[b] + edges[b + 1]))
        mean.append(float(U[m].mean()))
        sd.append(float(U[m].std()))
    ctr, mean, sd = map(np.asarray, (ctr, mean, sd))
    ax.fill_between(ctr, mean - sd, mean + sd, color="#ff2d95", alpha=0.18)
    ax.plot(ctr, mean, color="#ff2d95", marker="o", ms=3.5, lw=2.0,
            label="mean U per energy-bin  (±1 std)")

    ax.set_xlim(lo, hi); ax.set_ylim(0.0, y_hi)
    # `log_e` already has GEV_TO_EV_LOG10 added in by the caller, so these tick
    # numbers ARE log10(E/eV) -- the label states what the numbers already are,
    # not a conversion still owed.
    ax.set_xlabel(r"$\log_{10}(E_\tau \,/\, \mathrm{eV})$")
    ax.set_ylabel("per-event utility U")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)


def score_run(run_dir, prim, fnn, recon, mountain, device, formats=("png",)):
    layout_path = os.path.join(run_dir, "layout_best.pt")
    e_o, n_o = load_layout(layout_path, mountain)
    x, y = e_o.to(device), n_o.to(device)

    U_i, E_true, r = utility_per_event(x, y, prim, fnn, recon)
    U_i_np = U_i.cpu().numpy()
    # E_true is in GeV (opt_core.primary_to_physical_labels' own docstring names
    # its return E_GeV). +GEV_TO_EV_LOG10 shifts the plotted numbers into eV --
    # log10(E/eV) is the convention this field actually quotes cosmic-ray-scale
    # energies in (this corpus spans ~1e14-1e17 eV) -- so the axis ticks are the
    # log of E in eV directly, not GeV relabelled.
    log_e = torch.log10(E_true).cpu().numpy() + GEV_TO_EV_LOG10

    # Correctness guard: mean is linear, so this must reproduce utility_of_xy's
    # own scalar U to floating-point precision. A mismatch means the per-event
    # decomposition above drifted from the composite it claims to mirror.
    U_ref, _, _ = utility_of_xy(x, y, prim, fnn, recon)
    U_ref = float(U_ref.item())
    U_mean = float(U_i_np.mean())
    print(f"[{os.path.basename(run_dir)}] mean(per-event U) = {U_mean:.4f}  "
          f"utility_of_xy U = {U_ref:.4f}  "
          f"(diff {abs(U_mean - U_ref):.2e}, should be ~0)")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    _plot(ax, log_e, U_i_np,
         f"{os.path.basename(run_dir)}\nN={U_i_np.size:,}  mean U={U_mean:.3f}  "
         f"mean r={float(r.mean()):.3f}")
    fig.tight_layout()
    out = os.path.join(run_dir, "utility_vs_energy.png")
    for fmt in formats:
        path = out if fmt == "png" else os.path.splitext(out)[0] + f".{fmt}"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        print(f"[save] {path}")
    plt.close(fig)
    return U_mean


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", nargs="+", required=True,
                    help="one or more optimizer run dirs, each containing layout_best.pt; "
                         "utility_vs_energy.png is written back into each")
    ap.add_argument("--n-events", type=int, default=5120,
                    help="fixed primary/cloud batch size, shared across all --run-dir "
                         "(same held-out events score every layout)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("utility vs tau energy — per-event composite, held-out corpus")
    print("=" * 72)
    print(f"device : {device}")

    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                                east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX,
                                n_planes=N_PLANES)
    _elec, _muon, B, n_pairs, prim = load_events(args.n_events, device, mountain)
    print(f"events : {B} of {n_pairs} pairs")

    fnn, recon = load_models(device)

    for run_dir in args.run_dir:
        if not os.path.exists(os.path.join(run_dir, "layout_best.pt")):
            print(f"[skip] no layout_best.pt in {run_dir}")
            continue
        score_run(run_dir, prim, fnn, recon, mountain, device)


if __name__ == "__main__":
    main()
