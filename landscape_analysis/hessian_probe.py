#!/usr/bin/env python3
"""
Hessian curvature probe for TAMBO v6 detector optimization.

At N_DETECTORS=100 (200 free (x,y) parameters), the full Hessian of the
composite utility U w.r.t. detector positions is small enough to form
EXACTLY via 200 Hessian-vector products (double backward through the frozen
surrogate), rather than approximating the spectrum with Lanczos/power
iteration as is standard for million-parameter models.

Computes the full eigenspectrum at:
  - GES-best optimized layout   (U~210, from landscape_analysis.py Q2)
  - L-BFGS DS-best optimized layout (U~209, ~260m Hungarian-displaced from GES)
  - a random layout              (baseline)

Reports: eigenvalue range, count of positive/negative/~flat directions,
condition number over the nonzero eigenvalues. Answers directly: how flat is
the "plateau" (in how many directions), and does curvature differ between the
two same-U/different-basin optima found earlier.
"""
import sys, os, json, time
import numpy as np
import torch

_V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _V6)
from _pathfix import V6_ROOT  # noqa: F401 — idempotent, registers v6 root

import layouts as _layouts  # noqa: E402  (input/output locations)
import modules  # noqa: F401 — package import; keeps modules on the path

from modules.constants import (
    N_DETECTORS, GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
    TRAINING_DATASET_FOLDER, FNN_FOLDER, RECON_FOLDER,
)
from modules.geometry import load_tr_mountain
from modules.optimize import utility_of_xy, load_models
from modules.layouts import layout_uniform_random

DEVICE = torch.device("cpu")
RUN_BASE = _layouts.RUNS
SEED = 42
BATCH_SIZE = 512
D = 2 * N_DETECTORS

print("=" * 70)
print("TAMBO v6 Hessian Curvature Probe")
print("=" * 70)
print(f"Device: {DEVICE}   D={D} (N_DETECTORS={N_DETECTORS})")

# ── Load models / geometry / primaries (same recipe as landscape_analysis.py)
print("\nLoading models...")
t0 = time.time()
fnn, recon = load_models(DEVICE, fnn_folder=FNN_FOLDER, recon_dir=RECON_FOLDER + "_deepsets")
print(f"Models loaded in {time.time()-t0:.1f}s")

mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES)

primary_all = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt"),
                         weights_only=False).float()
n_total = primary_all.shape[0]
g = torch.Generator().manual_seed(SEED)
idx_fixed = torch.randint(0, n_total, (BATCH_SIZE,), generator=g)
primary_fixed = primary_all[idx_fixed].to(DEVICE)
print(f"Fixed batch: {BATCH_SIZE} primaries (seed={SEED})")


def load_layout(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["x"].float().reshape(-1), d["y"].float().reshape(-1), float(d["U"])


def exact_hessian(x, y, primary_batch, label=""):
    """Exact D x D Hessian of U via D Hessian-vector products (double backward)."""
    xy = torch.cat([x.clone(), y.clone()]).to(DEVICE).requires_grad_(True)
    t0 = time.time()
    U, r, _ = utility_of_xy(xy[:N_DETECTORS], xy[N_DETECTORS:], primary_batch, fnn, recon)
    grad = torch.autograd.grad(U, xy, create_graph=True)[0]   # (D,)

    H = torch.zeros(D, D)
    for i in range(D):
        e_i = torch.zeros(D)
        e_i[i] = 1.0
        Hi = torch.autograd.grad(grad, xy, grad_outputs=e_i, retain_graph=True)[0]
        H[i] = Hi.detach()
    H = 0.5 * (H + H.T)  # symmetrize away numerical asymmetry
    dt = time.time() - t0

    eigvals = np.linalg.eigvalsh(H.numpy())
    scale = np.abs(eigvals).max()
    n_pos = int((eigvals > 1e-6 * scale).sum())
    n_neg = int((eigvals < -1e-6 * scale).sum())
    n_flat = D - n_pos - n_neg
    nonzero = eigvals[np.abs(eigvals) > 1e-6 * scale]
    cond = float(np.abs(nonzero).max() / np.abs(nonzero).min()) if len(nonzero) else float("nan")

    print(f"\n[{label}] U={U.item():.4f}  ||grad||={grad.norm().item():.4f}  "
          f"Hessian computed in {dt:.1f}s")
    print(f"  eig range: [{eigvals.min():.4f}, {eigvals.max():.4f}]")
    print(f"  # positive={n_pos}  # negative={n_neg}  # ~flat={n_flat}  (D={D})")
    print(f"  condition number (nonzero eigs): {cond:.3e}")

    return dict(U=float(U.item()), grad_norm=float(grad.norm().item()),
                eigvals=eigvals.tolist(), n_pos=n_pos, n_neg=n_neg,
                n_flat=n_flat, cond=cond, time_s=dt)


print("\nLoading layouts...")
ges_x, ges_y, ges_U_saved = load_layout(
    f"{RUN_BASE}/test_v6_run_04_optimize_ges_combined/layout_best.pt")
lbfgs_x, lbfgs_y, lbfgs_U_saved = load_layout(
    f"{RUN_BASE}/test_v6_run_04_optimize_lbfgs_ensemble_ds_combined/layout_best.pt")
rx_rand, ry_rand = layout_uniform_random(mountain, rng=np.random.default_rng(0))
print(f"GES best U (saved): {ges_U_saved:.4f}")
print(f"L-BFGS DS best U (saved): {lbfgs_U_saved:.4f}")

results = {}
results["random"]     = exact_hessian(rx_rand, ry_rand, primary_fixed, "Random layout")
results["ges_best"]   = exact_hessian(ges_x, ges_y, primary_fixed, "GES best")
results["lbfgs_best"] = exact_hessian(lbfgs_x, lbfgs_y, primary_fixed, "L-BFGS DS best")

out_path = os.path.join(_layouts.results_dir(), "hessian_probe_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved full spectra to {out_path}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for k in ["random", "ges_best", "lbfgs_best"]:
    r = results[k]
    print(f"  {k:12s}: U={r['U']:8.3f}  eig=[{min(r['eigvals']):10.3f}, {max(r['eigvals']):10.3f}]  "
          f"flat={r['n_flat']:3d}/{D}  cond={r['cond']:.2e}")

print()
print("Conclusions:")
rand_flat  = results["random"]["n_flat"]
ges_flat   = results["ges_best"]["n_flat"]
lbfgs_flat = results["lbfgs_best"]["n_flat"]
if ges_flat > rand_flat + 20:
    print(f"  [FLATTER AT OPTIMUM] optimized layouts have far more ~zero-curvature "
          f"directions ({ges_flat}/{D}) than random ({rand_flat}/{D}) => broad plateau confirmed quantitatively.")
else:
    print(f"  [SIMILAR CURVATURE] optimized ({ges_flat}/{D}) vs random ({rand_flat}/{D}) "
          f"flat-direction counts are close => 'plateau' may be an artifact of the earlier 1D probes, not a true broad flat basin.")

if abs(ges_flat - lbfgs_flat) < 15 and results["ges_best"]["cond"] / results["lbfgs_best"]["cond"] < 3:
    print("  [CONSISTENT BASINS] GES-best and L-BFGS-best have similar curvature "
          "signatures despite ~260m displacement => supports same-plateau / mode-connected interpretation.")
else:
    print("  [DIFFERENT CURVATURE] GES-best and L-BFGS-best have meaningfully different "
          "curvature signatures => may be genuinely distinct basins, not just displaced points on one plateau.")
