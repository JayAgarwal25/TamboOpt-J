"""2x2 controlled utility evaluation: layout x recon.

Evaluates utility U for all combinations of:
  - Jay's best layout   (L_star_r0.pt, found by buggy lbfgs_ensemble — (North, Up) coords)
  - Zlatan's best layout (zdimitrov/.../test_v6_run_04_optimize_de_population/layout_best.pt,
                          found with correct DE — (North, East) coords)
  x
  - Flat MLP recon  (test_v6_run_03_recentered/recon.pt,    val=0.126)
  - DeepSets recon  (test_v6_run_03_recentered_deepsets/recon.pt, val=0.001118)

COORDINATE NOTE: The FNN expects (North, East) inputs. Jay's layout is stored in
(North, Up) because lbfgs_ensemble.py had a coordinate bug (now fixed). Jay's layout
is converted to (North, East) by snapping each detector to the nearest mountain
centroid in NE space (project_to_mountain_ne with Up treated as if it were East).
This conversion is approximate but gives the most faithful NE version of Jay's layout.

Zlatan's layout is already in (North, East) and is evaluated as-is.

Primary batch: 512 primaries sampled with seed=42 from the training primary.pt,
               matching the fixed batch used by lbfgs_refine in 04_optimize_lbfgs_ensemble.py.

Run from detector_optimization_v6/:
    python eval_recon_comparison.py
"""
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch

import modules_v6  # injects v3/v4 into sys.path
from modules_v6.dual_surrogate import load_dual_surrogate
from modules_v6.reconstruction import build_recon_from_ckpt
from modules_v6.tr_geometry_ne import project_to_mountain_ne
from modules_v6.constants import (
    N_DETECTORS, PRIMARY_DIM,
    FNN_FOLDER, RUN_LOCATION,
    TRAINING_DATASET_FOLDER,
    LOG_E_MIN, LOG_E_MAX,
    GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
)
from modules_v4.tr_geometry import load_tr_mountain
from modules.utility_functions import reconstructability, U_E, U_angle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

W_THETA = 1e2
W_PHI   = 1e2
W_E     = 2.5e2
W_DIV   = 1e3
LAYOUT_THRESHOLD      = 5e-2
RECONSTRUCT_THRESHOLD = 10.0
LBFGS_BATCH_PRIMARIES = 512
SEED = 42

ZDIMITROV_V6 = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6"


def primary_to_physical_labels(primary: torch.Tensor):
    dir_x = primary[:, 0]
    dir_y = primary[:, 1]
    dir_z = primary[:, 2].clamp(-1.0, 1.0)
    log_e_norm = primary[:, 3]
    log_e = log_e_norm * (LOG_E_MAX - LOG_E_MIN) + LOG_E_MIN
    E_gev = torch.exp(log_e) - 1.0
    theta = torch.arccos(dir_z)
    phi   = torch.atan2(dir_y, dir_x)
    two_pi = 2.0 * math.pi
    phi = torch.where(phi < 0, phi + two_pi, phi)
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
    u_e     = U_E    (E_pred_phys, E_true,    r)
    U = (W_THETA * u_theta + W_PHI * u_phi + W_E * u_e) / W_DIV
    return float(U.item()), dict(
        u_theta=float(W_THETA * u_theta / W_DIV),
        u_phi=float(W_PHI   * u_phi   / W_DIV),
        u_e=float(W_E     * u_e     / W_DIV),
        r_mean=float(r.mean().item()),
    )


def load_layout(path, label, mountain=None, is_north_up=False):
    """Load a layout from path.

    If is_north_up=True, the file stores (North, Up) and must be converted to
    (North, East) by snapping each detector to the nearest mountain centroid in
    NE space. This is the case for layouts saved by the old (buggy) lbfgs_ensemble.
    """
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        x = raw["x"].float().reshape(-1)
        y = raw["y"].float().reshape(-1)
    else:
        raw = raw.float()
        assert raw.ndim == 2 and raw.shape[1] == 2, f"unexpected layout shape {raw.shape}"
        x = raw[:, 0]
        y = raw[:, 1]

    if is_north_up:
        # y currently holds Up (elevation ~2442-3886m). The FNN expects East (-2019 to +1182m).
        # Convert by finding each detector's nearest mountain centroid in NE space.
        # project_to_mountain_ne snaps any point whose "East" value is far from all
        # centroids to the nearest centroid — since Up≈3000 >> East≈0, all detectors
        # snap to centroids with East closest to their Up value, which gives an NE
        # layout that preserves the North distribution but corrects the East channel.
        print(f"[layout] {label}: (North, Up) stored — converting to (North, East)")
        print(f"  N in [{x.min():.1f}, {x.max():.1f}]  Up in [{y.min():.1f}, {y.max():.1f}]")
        x_ne, y_ne = project_to_mountain_ne(mountain, x, y)
        print(f"  -> N in [{x_ne.min():.1f}, {x_ne.max():.1f}]  E in [{y_ne.min():.1f}, {y_ne.max():.1f}]")
        return x_ne.to(DEVICE), y_ne.to(DEVICE)

    print(f"[layout] {label}: N in [{x.min():.1f}, {x.max():.1f}]  "
          f"E in [{y.min():.1f}, {y.max():.1f}]  shape=({len(x)},)")
    return x.to(DEVICE), y.to(DEVICE)


def main():
    print("=" * 72)
    print("eval_recon_comparison.py — 2x2 layout x recon utility table")
    print(f"device: {DEVICE}")
    print("=" * 72)

    # ── Primary batch ────────────────────────────────────────────────────────
    primary_all = torch.load(
        os.path.join(TRAINING_DATASET_FOLDER, "primary.pt"), weights_only=False,
    ).float()
    n_total = primary_all.shape[0]
    print(f"[load] {n_total} primaries from training dataset")

    g = torch.Generator().manual_seed(SEED)
    idx = torch.randint(0, n_total, (LBFGS_BATCH_PRIMARIES,), generator=g)
    primary_batch = primary_all[idx].to(DEVICE)
    print(f"[sample] fixed batch of {LBFGS_BATCH_PRIMARIES} primaries (seed={SEED})")

    # ── Mountain (needed for North+Up → North+East conversion) ───────────────
    print(f"\n[load] mountain geometry from {GEOMETRY_PATH}")
    mountain = load_tr_mountain(
        GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
        east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES,
    )
    print(f"[mountain] N=[{mountain.n_min:.0f}, {mountain.n_max:.0f}]  "
          f"E=[{mountain.east_lo:.0f}, {mountain.east_hi:.0f}]")

    # ── FNN (shared across all evaluations) ──────────────────────────────────
    print(f"\n[load] FNN from {FNN_FOLDER}")
    fnn = load_dual_surrogate(FNN_FOLDER, DEVICE)

    # ── Recons ───────────────────────────────────────────────────────────────
    recon_paths = {
        "flat_MLP":  os.path.join(RUN_LOCATION, "test_v6_run_03_recentered",          "recon.pt"),
        "DeepSets":  os.path.join(RUN_LOCATION, "test_v6_run_03_recentered_deepsets", "recon.pt"),
    }
    recons = {}
    for name, path in recon_paths.items():
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        recon = build_recon_from_ckpt(ckpt, N_DETECTORS, DEVICE)
        recon.eval()
        cfg = ckpt.get("config", {})
        print(f"[load] recon '{name}': model_type={cfg.get('model_type','mlp')}  "
              f"val_total={cfg.get('val_total', ckpt.get('val_total','?'))}")
        recons[name] = recon

    # ── Layouts ──────────────────────────────────────────────────────────────
    # Jay's layout: saved by old lbfgs_ensemble which used (North, Up) — must convert to NE.
    # Zlatan's layout: saved by DE which uses correct (North, East) — load as-is.
    print()
    layouts = {}
    layouts["Jay (buggy Up→NE conv)"] = load_layout(
        os.path.join(RUN_LOCATION, "L_star_r0.pt"),
        label="Jay (buggy Up→NE conv)",
        mountain=mountain,
        is_north_up=True,
    )
    layouts["Zlatan (correct NE)"] = load_layout(
        os.path.join(ZDIMITROV_V6, "test_v6_run_04_optimize_de_population", "layout_best.pt"),
        label="Zlatan (correct NE)",
    )

    # ── Evaluation ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("RESULTS  (512 primaries, seed=42, same FNN for all)")
    print(f"{'=' * 72}")
    print(f"{'Layout':<28} {'Recon':<12} {'U_total':>9} {'u_theta':>9} {'u_phi':>9} {'u_e':>9} {'r_mean':>8}")
    print("-" * 72)

    results = {}
    for layout_name, (x_det, y_det) in layouts.items():
        for recon_name, recon in recons.items():
            with torch.no_grad():
                U, parts = utility_of_xy(x_det, y_det, primary_batch, fnn, recon)
            results[(layout_name, recon_name)] = U
            print(f"{layout_name:<28} {recon_name:<12} {U:>9.3f} "
                  f"{parts['u_theta']:>9.3f} {parts['u_phi']:>9.3f} "
                  f"{parts['u_e']:>9.3f} {parts['r_mean']:>8.3f}")
    print("-" * 72)

    # ── 2x2 table ────────────────────────────────────────────────────────────
    layout_names = list(layouts.keys())
    recon_names  = list(recons.keys())
    print("\n2x2 TABLE  (U_total)")
    header = f"{'':28}" + "".join(f"{r:>14}" for r in recon_names)
    print(header)
    for l in layout_names:
        row = f"{l:<28}" + "".join(f"{results[(l, r)]:>14.3f}" for r in recon_names)
        print(row)

    print()
    # Interpretation
    jay_mlp      = results[("Jay (buggy Up→NE conv)",  "flat_MLP")]
    jay_ds       = results[("Jay (buggy Up→NE conv)",  "DeepSets")]
    zlatan_mlp   = results[("Zlatan (correct NE)",     "flat_MLP")]
    zlatan_ds    = results[("Zlatan (correct NE)",     "DeepSets")]

    print("Interpretation:")
    print(f"  Jay layout:    MLP={jay_mlp:.3f}  DS={jay_ds:.3f}  "
          f"ratio MLP/DS={jay_mlp/max(jay_ds,1e-6):.2f}x")
    print(f"  Zlatan layout: MLP={zlatan_mlp:.3f}  DS={zlatan_ds:.3f}  "
          f"ratio MLP/DS={zlatan_mlp/max(zlatan_ds,1e-6):.2f}x")
    if jay_mlp > jay_ds and zlatan_mlp > zlatan_ds:
        print("  -> Flat MLP recon gives higher U for BOTH layouts.")
        print("     The recon is the primary cause of Jay's low U (not the layout).")
    elif jay_ds < zlatan_ds:
        print("  -> DeepSets recon degrades Jay's layout more than Zlatan's.")
        print("     Consistent with OOD sensitivity hypothesis.")
    elif jay_mlp < zlatan_mlp:
        diff = zlatan_mlp - jay_mlp
        print(f"  -> With flat MLP recon, Jay's layout is still {diff:.3f} below Zlatan's.")
        print("     Zlatan's optimizer found a genuinely better region.")
    print("=" * 72)


if __name__ == "__main__":
    main()
