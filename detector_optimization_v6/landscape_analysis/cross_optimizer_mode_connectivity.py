#!/usr/bin/env python3
"""
Cross-optimizer-family extension of mode_connectivity.py.

Every landscape-analysis experiment so far (grid scan, detector removal,
full-space slice, mode connectivity, the 5-seed L-BFGS study) has used ONLY
L-BFGS-derived layouts. The original motivation for the whole plateau
investigation was that structurally UNRELATED optimizers -- L-BFGS
(gradient), GES (evolutionary strategy), EvoGrad (fitness-weighted
evolutionary), DE (differential evolution), v5's (mu+lambda)-ES -- all
converge to similar U. That observation is separate evidence and still valid
on its own, but it does NOT mean their specific optima sit on the SAME
connected plateau as L-BFGS's -- that has never actually been tested. This
script closes that gap directly.

Hungarian-aligns each of these independently-found optima to the L-BFGS-best
reference (same method as mode_connectivity.py, opt_core.align_to_reference),
then linearly interpolates from L-BFGS to each of them, checking whether U
stays high along the path (same connected plateau) or dips (real barrier,
separate basin).

GES is included for contrast despite being flagged untrustworthy elsewhere in
this project (Hessian probe: GES-best sits on a numerical singularity, not a
genuine smooth optimum, see feedback_trust_lbfgs_over_ges) -- specifically to
see whether that numerical pathology shows up as a real interpolation
barrier, or whether GES's answer is connected to L-BFGS's anyway despite
sitting on a kink.

v5's (mu+lambda)-ES has two candidate runs; the earliest (20260622_041128)
predates the (North,East) vs (North,Up) coordinate-convention fix elsewhere
in this project and is excluded (its saved "East" values fall entirely
outside the mountain's real East span -- that's the elevation/Up scale
instead). 20260624_061237 has valid-range positions and its saved U=200.24
matches the known-good optimizer-comparison figure already in memory
(v5 (mu+lambda)-ES: 200.2), confirming it's a genuine, valid comparison point.
A built-in coordinate-range check (see COMPARISON_LAYOUTS loop below) flags
and skips any layout whose positions fall outside plausible mountain bounds,
so this class of issue doesn't silently corrupt future runs of this script.

DE's saved U (52-54, drastically worse than even random layouts ~150-170) is
suspicious -- re-evaluated here with the CURRENT frozen FNN+recon on fresh
batches (not trusting the possibly-stale saved value) before deciding whether
DE is a meaningful comparison point at all.
"""
import os, sys, json, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _V6)
import layouts as _layouts  # noqa: E402  (input/output locations)
import modules_v6  # noqa: F401

from modules_v6.constants import (
    N_DETECTORS, GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
    TRAINING_DATASET_FOLDER, FNN_FOLDER, RECON_FOLDER,
)
from modules_v4.tr_geometry import load_tr_mountain
from modules_v6.opt_core import utility_of_xy, load_models, align_to_reference
from modules_v6.tr_geometry_ne import project_to_mountain_ne

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Results live beside the other run outputs, not next to the code.
HERE = _layouts.results_dir()
RUN_BASE = _layouts.RUNS
BATCH_SEED_BASE = 1000
BATCH_SIZE = 512
N_BATCHES = 6
N_STEPS = 41

V5_RUN_BASE = os.path.join(os.path.dirname(_layouts.RUNS), "v5_es_runs")

COMPARISON_LAYOUTS = {
    "evograd": f"{RUN_BASE}/test_v6_run_04_optimize_evograd_combined/layout_best.pt",
    "de": f"{RUN_BASE}/test_v6_run_04_optimize_de_ensemble_grid/layout_best.pt",   # de_ensemble_combined is empty
    "ges": f"{RUN_BASE}/test_v6_run_04_optimize_ges_combined/layout_best.pt",
    # v5's (mu+lambda)-ES: 20260622_041128 predates the (North,East) vs (North,Up)
    # coordinate-convention fix (its saved "y"/East values range 2368-3901, which
    # doesn't overlap the mountain's real East span at all -- that's the elevation
    # scale, so that run is invalid and excluded. 20260624_061237 has valid-range
    # positions and its saved U=200.24 matches the known-good optimizer-comparison
    # figure in memory (v5 (mu+lambda)-ES: 200.2), confirming it's the right one.
    "mu_lambda_es": f"{V5_RUN_BASE}/20260624_061237/layout_best.pt",
}
REFERENCE_LAYOUT = f"{RUN_BASE}/test_v6_run_04_optimize_lbfgs_ensemble_ds_combined/layout_best.pt"

print("=" * 70)
print("Cross-optimizer-family mode connectivity: L-BFGS vs. EvoGrad, DE, GES")
print("=" * 70)

fnn, recon = load_models(DEVICE, fnn_folder=FNN_FOLDER, recon_dir=RECON_FOLDER + "_deepsets")
mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES)

primary_all = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt"),
                         weights_only=False).float()
n_total = primary_all.shape[0]


def fresh_batch(seed):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n_total, (BATCH_SIZE,), generator=g)
    return primary_all[idx].to(DEVICE)


BATCHES = [fresh_batch(BATCH_SEED_BASE + b) for b in range(N_BATCHES)]


@torch.no_grad()
def eval_U(x, y):
    Us = [float(utility_of_xy(x.to(DEVICE), y.to(DEVICE), p, fnn, recon)[0].item()) for p in BATCHES]
    return float(np.mean(Us))


def load_layout(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["x"].float().reshape(-1), d["y"].float().reshape(-1), float(d["U"])


ref_x, ref_y, ref_U_saved = load_layout(REFERENCE_LAYOUT)
ref_U_re = eval_U(ref_x, ref_y)
print(f"\nL-BFGS (reference) layout: U saved={ref_U_saved:.4f}  re-evaluated={ref_U_re:.4f}")

results = {"reference": dict(U_saved=ref_U_saved, U_reeval=ref_U_re)}

for tag, path in COMPARISON_LAYOUTS.items():
    cmp_x, cmp_y, cmp_U_saved = load_layout(path)

    # Coordinate-convention sanity check: catches layouts saved in the old
    # (North, Up) convention (pre coordinate-fix, see v5's 20260622_041128) --
    # its "East" values would actually be elevation-scaled and fall well
    # outside the mountain's real East span.
    n_lo, n_hi = float(mountain.n_min), float(mountain.n_max)
    e_lo, e_hi = float(mountain.east_lo), float(mountain.east_hi)
    cx_lo, cx_hi = float(cmp_x.min()), float(cmp_x.max())
    cy_lo, cy_hi = float(cmp_y.min()), float(cmp_y.max())
    n_pad, e_pad = 0.25 * (n_hi - n_lo), 0.25 * (e_hi - e_lo)
    coords_plausible = (cx_lo > n_lo - n_pad and cx_hi < n_hi + n_pad and
                         cy_lo > e_lo - e_pad and cy_hi < e_hi + e_pad)
    print(f"\n[{tag}] layout: N range=[{cx_lo:.0f},{cx_hi:.0f}]  E range=[{cy_lo:.0f},{cy_hi:.0f}]  "
          f"(mountain N=[{n_lo:.0f},{n_hi:.0f}]  E=[{e_lo:.0f},{e_hi:.0f}])")
    if not coords_plausible:
        print(f"[{tag}] WARNING: position range falls well outside the mountain's real "
              f"(East, North) bounds -- likely a stale (North, Up) coordinate-convention "
              f"layout, NOT a valid comparison point. Skipping.")
        results[tag] = dict(skipped=True, reason="coordinates outside plausible (N,E) bounds")
        continue

    cmp_U_re = eval_U(cmp_x, cmp_y)
    print(f"[{tag}] U saved={cmp_U_saved:.4f}  re-evaluated (current FNN+recon, "
          f"fresh batches)={cmp_U_re:.4f}")
    if cmp_U_re < 0.5 * ref_U_re:
        print(f"[{tag}] WARNING: re-evaluated U is far below the reference layout and below "
              f"the known random-layout range (~150-170) -- this optimum may not be a "
              f"meaningful comparison point (stale checkpoint mismatch or a genuinely failed "
              f"run), interpret its connectivity result with that in mind.")

    raw_disp = ((ref_x - cmp_x) ** 2 + (ref_y - cmp_y) ** 2).sqrt().mean()

    xy_stack = np.stack([
        np.stack([ref_x.numpy(), ref_y.numpy()], axis=-1),
        np.stack([cmp_x.numpy(), cmp_y.numpy()], axis=-1),
    ], axis=0)  # (2, n_det, 2)
    aligned, perms = align_to_reference(xy_stack, ref_idx=0)
    cmp_x_aligned = torch.as_tensor(aligned[1, :, 0], dtype=torch.float32)
    cmp_y_aligned = torch.as_tensor(aligned[1, :, 1], dtype=torch.float32)

    aligned_disp = ((ref_x - cmp_x_aligned) ** 2 + (ref_y - cmp_y_aligned) ** 2).sqrt().mean()
    cmp_U_aligned_check = eval_U(cmp_x_aligned, cmp_y_aligned)
    print(f"[{tag}] mean displacement: raw(unaligned)={raw_disp:.1f}m  "
          f"AFTER Hungarian alignment={aligned_disp:.1f}m")
    print(f"[{tag}] U after alignment (sanity check, should match re-evaluated): "
          f"{cmp_U_aligned_check:.4f}")

    print(f"[{tag}] sweeping {N_STEPS} points along the straight line from L-BFGS "
          f"to (aligned) {tag} ...")
    t0 = time.time()
    ts = np.linspace(0.0, 1.0, N_STEPS)
    U_path = np.zeros(N_STEPS, dtype=np.float64)
    disp_path = np.zeros(N_STEPS, dtype=np.float64)
    for k, t in enumerate(ts):
        e_t = (1 - t) * ref_x + t * cmp_x_aligned
        n_t = (1 - t) * ref_y + t * cmp_y_aligned
        e_proj, n_proj = project_to_mountain_ne(mountain, e_t, n_t)
        U_path[k] = eval_U(e_proj, n_proj)
        disp_path[k] = float(((e_proj - e_t) ** 2 + (n_proj - n_t) ** 2).sqrt().mean())
    dt = time.time() - t0
    print(f"[{tag}] done in {dt:.0f}s")

    min_along_path = float(U_path.min())
    min_t = float(ts[np.argmin(U_path)])
    endpoints_mean = 0.5 * (U_path[0] + U_path[-1])
    dip = endpoints_mean - min_along_path
    print(f"[{tag}] endpoints: U(t=0, L-BFGS)={U_path[0]:.3f}  U(t=1, {tag})={U_path[-1]:.3f}")
    print(f"[{tag}] minimum along path: U={min_along_path:.3f} at t={min_t:.3f}")
    print(f"[{tag}] dip below endpoints' average: {dip:.3f} ({100*dip/endpoints_mean:.2f}%)")
    print(f"[{tag}] mean mountain-projection correction along path: {disp_path.mean():.1f}m")

    results[tag] = dict(
        U_saved=cmp_U_saved, U_reeval=cmp_U_re,
        raw_displacement_m=float(raw_disp), aligned_displacement_m=float(aligned_disp),
        U_aligned_check=cmp_U_aligned_check,
        ts=ts.tolist(), U_path=U_path.tolist(), disp_path=disp_path.tolist(),
        min_along_path=min_along_path, min_t=min_t, dip_below_endpoints=dip,
        dip_pct=100 * dip / endpoints_mean,
    )

out_path = os.path.join(HERE, "cross_optimizer_mode_connectivity_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for tag in COMPARISON_LAYOUTS:
    r = results[tag]
    if r.get("skipped"):
        continue
    axes[0].plot(r["ts"], r["U_path"], "-o", markersize=3, label=f"L-BFGS -> {tag}")
    axes[1].plot(r["ts"], r["disp_path"], "-o", markersize=3, label=tag)
axes[0].set_xlabel("t (0=L-BFGS, 1=other optimizer)")
axes[0].set_ylabel("U")
axes[0].set_title("U along straight-line path, L-BFGS to each other optimizer's optimum")
axes[0].legend(fontsize=8)
axes[1].set_xlabel("t")
axes[1].set_ylabel("mean mountain-projection correction (m)")
axes[1].set_title("How far off-surface the straight line wanders")
axes[1].legend(fontsize=8)
fig.tight_layout()
out_png = os.path.join(HERE, "cross_optimizer_mode_connectivity.png")
fig.savefig(out_png, dpi=150)
print(f"[plot] wrote {out_png}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
for tag in COMPARISON_LAYOUTS:
    r = results[tag]
    if r.get("skipped"):
        print(f"  L-BFGS <-> {tag:12s}: SKIPPED ({r['reason']})")
        continue
    verdict = ("ONE connected plateau" if r["dip_pct"] < 5.0 else
               "a real barrier -- separate basins")
    caveat = " (but re-evaluated U is suspiciously low, interpret with caution)" \
        if r["U_reeval"] < 0.5 * ref_U_re else ""
    print(f"  L-BFGS <-> {tag:12s}: dip={r['dip_pct']:.2f}%  -> consistent with {verdict}{caveat}")
