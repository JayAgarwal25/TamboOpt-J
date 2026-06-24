"""Cross-evaluation: 7 optimizer-found layouts x 2 recon networks.

All layouts are post-coordinate-bug-fix (commit 639fe94) and stored in
(North, East) convention. FNN expects (North, East) inputs.

Layouts evaluated:
  Zlatan DE         -- Zlatan's DE population run, 50k primaries, reference U=138.87
  Jay L-BFGS + MLP  -- Jay's L-BFGS with flat MLP recon,  U=60.6  at 512 prim
  Jay L-BFGS + DS   -- Jay's L-BFGS with DeepSets recon,  U=208.9 at 512 prim
  Jay ES + MLP      -- Jay's ES with flat MLP recon,       U=69.0  at 512 prim
  Jay ES + DS       -- Jay's ES with DeepSets recon,       U=200.2 at 512 prim
  Jay CMA-ES + MLP  -- Jay's CMA-ES with flat MLP recon,  U=49.3  at 512 prim
  Jay CMA-ES + DS   -- Jay's CMA-ES with DeepSets recon,  U=189.1 at 512 prim

Recon networks:
  flat_MLP  -- test_v6_run_03_recentered,          val_total=0.126
  DeepSets  -- test_v6_run_03_recentered_deepsets, val_total=0.001118

Primary batch: 512, seed=42, same FNN for all.

Run from detector_optimization_v6/:
    python eval_cross_comparison.py
"""
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch

import modules_v6
from modules_v6.dual_surrogate import load_dual_surrogate
from modules_v6.reconstruction import build_recon_from_ckpt
from modules_v6.constants import (
    N_DETECTORS, PRIMARY_DIM,
    FNN_FOLDER, RUN_LOCATION,
    TRAINING_DATASET_FOLDER,
    LOG_E_MIN, LOG_E_MAX,
)
from modules.utility_functions import reconstructability, U_E, U_angle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

W_THETA = 1e2
W_PHI   = 1e2
W_E     = 2.5e2
W_DIV   = 1e3
LAYOUT_THRESHOLD      = 5e-2
RECONSTRUCT_THRESHOLD = 10.0
BATCH_PRIMARIES = 512
SEED = 42

JAGARWAL    = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal"
ZDIMITROV   = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6"

LAYOUT_SPECS = [
    ("Zlatan DE",
     f"{ZDIMITROV}/optimization_runs_200k_dual/test_v6_run_04_optimize_de_population/layout_best.pt"),
    ("Jay L-BFGS+MLP",
     f"{JAGARWAL}/v6_runs/test_v6_run_04_optimize_lbfgs_ensemble_mlp_combined/layout_best.pt"),
    ("Jay L-BFGS+DS",
     f"{JAGARWAL}/v6_runs/test_v6_run_04_optimize_lbfgs_ensemble_ds_combined/layout_best.pt"),
    ("Jay ES+MLP",
     f"{JAGARWAL}/v5_es_runs/20260624_053311/layout_best.pt"),
    ("Jay ES+DS",
     f"{JAGARWAL}/v5_es_runs/20260624_061237/layout_best.pt"),
    ("Jay CMA-ES+MLP",
     f"{JAGARWAL}/v5_es_runs/cmaes_20260624_053315/layout_best.pt"),
    ("Jay CMA-ES+DS",
     f"{JAGARWAL}/v5_es_runs/cmaes_20260624_061214/layout_best.pt"),
]


def primary_to_physical_labels(primary):
    dir_z = primary[:, 2].clamp(-1.0, 1.0)
    log_e_norm = primary[:, 3]
    log_e = log_e_norm * (LOG_E_MAX - LOG_E_MIN) + LOG_E_MIN
    E_gev = torch.exp(log_e) - 1.0
    theta = torch.arccos(dir_z)
    phi = torch.atan2(primary[:, 1], primary[:, 0])
    phi = torch.where(phi < 0, phi + 2 * math.pi, phi)
    return E_gev, theta, phi


def utility_of_xy(x_det, y_det, primary_batch, fnn, recon):
    B = primary_batch.shape[0]
    xy_per_det = torch.stack([x_det, y_det], dim=-1)
    xy_batch   = xy_per_det.unsqueeze(0).expand(B, -1, -1)
    pred_ET    = fnn(primary_batch, xy_batch)
    E_pred_det = pred_ET[..., 0]
    T_pred_det = pred_ET[..., 1]
    recon_feats = torch.stack(
        [xy_batch[..., 0], xy_batch[..., 1], E_pred_det, T_pred_det], dim=-1,
    )
    pred = recon(recon_feats)
    E_pred_phys, theta_pred, phi_pred = primary_to_physical_labels(pred)
    E_pred_phys = E_pred_phys.clamp(min=1.0)
    E_true, theta_true, phi_true = primary_to_physical_labels(primary_batch)
    r = reconstructability(
        torch.expm1(E_pred_det),
        layout_threshold=LAYOUT_THRESHOLD,
        reconstruct_threshold=RECONSTRUCT_THRESHOLD,
    )
    u_theta = U_angle(theta_pred, theta_true, r)
    u_phi   = U_angle(phi_pred,   phi_true,   r)
    u_e     = U_E(E_pred_phys, E_true, r)
    U = (W_THETA * u_theta + W_PHI * u_phi + W_E * u_e) / W_DIV
    return float(U.item()), {
        "u_theta": float(W_THETA * u_theta / W_DIV),
        "u_phi":   float(W_PHI   * u_phi   / W_DIV),
        "u_e":     float(W_E     * u_e     / W_DIV),
        "r_mean":  float(r.mean().item()),
    }


def load_layout(path, label):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        x = raw["x"].float().reshape(-1)
        y = raw["y"].float().reshape(-1)
    else:
        raw = raw.float()
        assert raw.ndim == 2 and raw.shape[1] == 2, f"unexpected shape {raw.shape}"
        x, y = raw[:, 0], raw[:, 1]
    print(f"  [{label}] N=[{x.min():.0f},{x.max():.0f}] E=[{y.min():.0f},{y.max():.0f}]")
    return x.to(DEVICE), y.to(DEVICE)


def main():
    print("=" * 80)
    print("eval_cross_comparison.py -- 7 layouts x 2 recons (all post-coordinate-fix NE)")
    print(f"device: {DEVICE}")
    print("=" * 80)

    primary_all = torch.load(
        os.path.join(TRAINING_DATASET_FOLDER, "primary.pt"), weights_only=False,
    ).float()
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randint(0, primary_all.shape[0], (BATCH_PRIMARIES,), generator=g)
    primary_batch = primary_all[idx].to(DEVICE)
    print(f"[load] {primary_all.shape[0]} primaries, fixed batch={BATCH_PRIMARIES} seed={SEED}")

    print(f"\n[load] FNN from {FNN_FOLDER}")
    fnn = load_dual_surrogate(FNN_FOLDER, DEVICE)

    recon_paths = {
        "flat_MLP": os.path.join(RUN_LOCATION, "test_v6_run_03_recentered",          "recon.pt"),
        "DeepSets": os.path.join(RUN_LOCATION, "test_v6_run_03_recentered_deepsets", "recon.pt"),
    }
    recons = {}
    for name, path in recon_paths.items():
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        recon = build_recon_from_ckpt(ckpt, N_DETECTORS, DEVICE)
        recon.eval()
        cfg = ckpt.get("config", {})
        vt = cfg.get("val_total", ckpt.get("val_total", "?"))
        print(f"  [recon '{name}'] val_total={vt}")
    recons["flat_MLP"] = recon  # overwrite with last; reload properly
    # Reload correctly
    recons = {}
    for name, path in recon_paths.items():
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        r = build_recon_from_ckpt(ckpt, N_DETECTORS, DEVICE)
        r.eval()
        recons[name] = r

    print("\n[load] Layouts:")
    layouts = {}
    for label, path in LAYOUT_SPECS:
        layouts[label] = load_layout(path, label)

    recon_names  = list(recons.keys())
    layout_names = list(layouts.keys())

    # Full results table: rows=layouts, cols=recons
    results  = {}
    parts_all = {}
    print(f"\n{'=' * 80}")
    print("EVALUATING...")
    for lname, (x_det, y_det) in layouts.items():
        for rname, recon in recons.items():
            with torch.no_grad():
                U, parts = utility_of_xy(x_det, y_det, primary_batch, fnn, recon)
            results[(lname, rname)] = U
            parts_all[(lname, rname)] = parts
            print(f"  {lname:<20} + {rname:<10} -> U={U:.3f}  "
                  f"u_th={parts['u_theta']:.3f} u_ph={parts['u_phi']:.3f} "
                  f"u_e={parts['u_e']:.3f} r={parts['r_mean']:.3f}")

    # Summary table
    print(f"\n{'=' * 80}")
    col_w = 12
    header = f"{'Layout (opt recon)':<22}" + "".join(f"{r:>{col_w}}" for r in recon_names)
    print(header)
    print("-" * (22 + col_w * len(recon_names)))
    for lname in layout_names:
        row = f"{lname:<22}" + "".join(f"{results[(lname, r)]:>{col_w}.3f}" for r in recon_names)
        print(row)
    print("-" * (22 + col_w * len(recon_names)))

    # DS/MLP ratio column
    print(f"\n{'DS/MLP ratio per layout':}")
    for lname in layout_names:
        mlp_u = results[(lname, "flat_MLP")]
        ds_u  = results[(lname, "DeepSets")]
        ratio = ds_u / max(mlp_u, 1e-6)
        print(f"  {lname:<22}  MLP={mlp_u:.3f}  DS={ds_u:.3f}  DS/MLP={ratio:.2f}x")

    # Cross-eval analysis: does DS-optimized layout score higher with MLP than MLP-optimized?
    print(f"\n{'=' * 80}")
    print("CROSS-EVAL INTERPRETATION:")
    for opt_method, mlp_key, ds_key in [
        ("L-BFGS", "Jay L-BFGS+MLP", "Jay L-BFGS+DS"),
        ("ES",     "Jay ES+MLP",      "Jay ES+DS"),
        ("CMA-ES", "Jay CMA-ES+MLP",  "Jay CMA-ES+DS"),
    ]:
        mlp_with_mlp = results[(mlp_key, "flat_MLP")]
        mlp_with_ds  = results[(mlp_key, "DeepSets")]
        ds_with_mlp  = results[(ds_key,  "flat_MLP")]
        ds_with_ds   = results[(ds_key,  "DeepSets")]
        print(f"\n  {opt_method}:")
        print(f"    MLP-opt layout:  MLP-eval={mlp_with_mlp:.3f}  DS-eval={mlp_with_ds:.3f}")
        print(f"    DS-opt  layout:  MLP-eval={ds_with_mlp:.3f}  DS-eval={ds_with_ds:.3f}")
        if ds_with_mlp > mlp_with_mlp + 1.0:
            print(f"    -> DS-opt layout genuinely better: +{ds_with_mlp - mlp_with_mlp:.3f} on MLP eval")
        elif abs(ds_with_mlp - mlp_with_mlp) <= 1.0:
            print(f"    -> Layouts reach similar MLP utility; DS recon scale explains the difference")
        else:
            print(f"    -> MLP-opt layout better on MLP eval; DS optimizer may have overfit to DS recon")

    print("=" * 80)


if __name__ == "__main__":
    main()
