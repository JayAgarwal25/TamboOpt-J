"""Optimize detector positions via gradient descent through FNN + recon.

Loads the frozen FNN (step 2) and frozen reconstruction NN (step 3),
initializes a `LearnableXY` layout on the mountain surface, and iterates:

    x_det, y_det = xy_module()
    xy_per_det  = stack(x_det, y_det)                    # (100, 2)
    xy_batch    = xy_per_det broadcast over primary batch
    E_pred, T_pred = FNN(primary_batch, xy_batch)        # (B, 100, 2)
    recon_input  = (x, y, E_pred, T_pred) per detector
    norm_preds   = Reconstruction(recon_input_flat)       # (B, 3) tanh
    E_pred_phys, θ_pred, φ_pred = DenormalizeLabels(norm_preds)
    r            = reconstructability(E_pred)            # (B,) gate
    U = (1e2·U_θ + 1e2·U_φ + 1e3·U_E + 5e5·U_PR) / 1e3
    loss = -U
    loss.backward()
    optimizer.step()
    project_to_mountain()

Run:

    cd TambOpt/detector_optimization_v6
    python 04_optimize.py

Artifacts in `outputs/v6_run_04_optimize/`:
    layout_best.pt         best-U (x, y) snapshot
    layout_final.pt        last-epoch (x, y)
    xy_trajectory.pt       periodic snapshots
    optimize_log.json      per-epoch utility breakdown + grad norms
    optimize_curves.png    U and components over epochs
    layout_before_after.png top-down mountain with init vs. final
"""
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch

import modules_v6   # sys.path injection for v3 + v4
from modules_v6.fnn_surrogate import (
    FNNSurrogate
)
from modules_v6.constants import (
    N_DETECTORS, PRIMARY_DIM,
    GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
    TRAINING_DATASET_FOLDER, FNN_FOLDER, RECON_FOLDER,
    LOG_E_MIN, LOG_E_MAX,
)
from modules.reconstruction      import Reconstruction, DenormalizeLabels
from modules.utility_functions   import reconstructability, U_E, U_angle, U_PR
from modules.layout_optimization import LearnableXY
from modules_v4.tr_geometry      import load_tr_mountain


# ── Config ───────────────────────────────────────────────────────────────────
OPT_DIR = os.path.join(_HERE, "outputs", "v6_run_04_optimize")

N_OPT_EPOCHS       = 1000000 
PRIMARIES_PER_STEP = 256
LR                 = 1              # v3/v4 use lr=10; MLP Jacobian is larger
GRAD_CLIP          = 100.0
LAYOUT_INIT_SCHEME = "grid"           # grid | center | random

# Utility composite weights (match v4: (1e2·U_θ + 1e2·U_φ + 1e3·U_E + 5e5·U_PR) / 1e3)
W_THETA = 1e2
W_PHI   = 1e2
W_E     = 1e3
W_PR    = 5e5
W_DIV   = 1e3

# reconstructability thresholds (v4 uses reconstruct_threshold=10; layout default is 5e-2)
LAYOUT_THRESHOLD      = 5e-2
RECONSTRUCT_THRESHOLD = 10.0

LOG_EVERY  = 10
SAVE_EVERY = 50
SEED       = 42
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def primary_to_physical_labels(primary: torch.Tensor):
    """(B, 5) -> (E_GeV, θ_rad, φ_rad)."""
    dir_x = primary[:, 0]
    dir_y = primary[:, 1]
    dir_z = primary[:, 2].clamp(-1.0, 1.0)
    log_e_norm = primary[:, 3]

    log_e = log_e_norm * (LOG_E_MAX - LOG_E_MIN) + LOG_E_MIN
    E_gev = torch.pow(10.0, log_e)
    theta = torch.arccos(dir_z)
    phi   = torch.atan2(dir_y, dir_x)
    two_pi = 2.0 * math.pi
    phi = torch.where(phi < 0, phi + two_pi, phi)
    return E_gev, theta, phi


def _plot_curves(log, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep = [e["epoch"] for e in log]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(ep, [e["U"] for e in log], label="U")
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("U (composite)")
        axes[0].set_title("utility"); axes[0].grid(alpha=0.3); axes[0].legend()
        axes[1].plot(ep, [e["u_theta"] for e in log], label="U_\u03b8")
        axes[1].plot(ep, [e["u_phi"]   for e in log], label="U_\u03c6")
        axes[1].plot(ep, [e["u_e"]     for e in log], label="U_E")
        axes[1].plot(ep, [e["r_mean"]  for e in log], label="r_mean", linestyle="--")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("component")
        axes[1].set_title("components"); axes[1].grid(alpha=0.3); axes[1].legend()
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] skipped ({exc!r})")


def _plot_layout(x_final, y_final, N_init, U_init, mountain, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 6))
        ax.scatter(mountain.centroids_NUE[:, 0], mountain.centroids_NUE[:, 1],
                   s=2, c="lightgray", alpha=0.6, label="mountain")
        ax.scatter(N_init, U_init, s=28, c="tab:blue", label="init", alpha=0.8, edgecolors="none")
        ax.scatter(x_final, y_final, s=28, c="tab:red",  label="final", alpha=0.95, edgecolors="none")
        ax.set_xlabel("North [m]"); ax.set_ylabel("Up [m]")
        ax.set_aspect("equal"); ax.legend()
        ax.set_title("layout before / after")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] skipped ({exc!r})")


def main():
    os.makedirs(OPT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 72)
    print("v6/04_optimize.py")
    print("=" * 72)
    print(f"geometry         : {GEOMETRY_PATH}")
    print(f"training dataset : {TRAINING_DATASET_FOLDER}")
    print(f"fnn checkpoint   : {FNN_FOLDER}")
    print(f"recon checkpoint : {RECON_FOLDER}")

    print(f"opt dir         : {OPT_DIR}")
    print(f"device          : {DEVICE}")
    print(f"epochs          : {N_OPT_EPOCHS}")
    print(f"primaries/step  : {PRIMARIES_PER_STEP}")
    print(f"grad clip       : {GRAD_CLIP}")
    print(f"layout init     : {LAYOUT_INIT_SCHEME}")

    # Load corpus (primaries only — we sample batches each epoch)
    primary_all = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()
    norm_stats  = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "norm_stats.pt"))
    n_total_primaries = int(primary_all.shape[0])
    print(f"[load] {n_total_primaries} primaries")

    # Frozen FNN
    fnn_ckpt = torch.load(os.path.join(FNN_FOLDER, "fnn.pt"), map_location=DEVICE)
    fnn = FNNSurrogate(n_det=N_DETECTORS, primary_dim=PRIMARY_DIM,
                       hidden=512, dropout=0.1).to(DEVICE)
    fnn.load_state_dict(fnn_ckpt["state_dict"])
    fnn.set_normalization(norm_stats)
    fnn.eval()
    for p in fnn.parameters():
        p.requires_grad_(False)
    print(f"[load] fnn.pt    epoch={fnn_ckpt.get('epoch','?')}  "
          f"val={fnn_ckpt.get('val_total', fnn_ckpt.get('val','?'))}")

    # Frozen reconstruction
    recon_ckpt = torch.load(os.path.join(RECON_FOLDER, "recon.pt"), map_location=DEVICE)
    recon_feat = int(recon_ckpt.get("input_features", 4))
    recon_nd   = int(recon_ckpt.get("num_detectors", N_DETECTORS))
    cfg = recon_ckpt.get("config", {})
    recon = Reconstruction(
        input_features=recon_feat,
        num_detectors=recon_nd,
        hidden_lay1=int(cfg.get("hidden_lay1", 256)),
        hidden_lay2=int(cfg.get("hidden_lay2", 128)),
        hidden_lay3=int(cfg.get("hidden_lay3", 32)),
        output_dim=int(cfg.get("output_dim", 3)),
    ).to(DEVICE)
    recon.load_state_dict(recon_ckpt["state_dict"])
    recon.eval()
    for p in recon.parameters():
        p.requires_grad_(False)
    # Frozen z-score stats for the recon input (v4 pattern). Required because
    # v3's Reconstruction has no internal normalization and mountain-scale xy
    # would otherwise saturate its Tanh head.
    if "input_mean" not in recon_ckpt or "input_std" not in recon_ckpt:
        raise RuntimeError(
            "recon.pt is missing 'input_mean'/'input_std'. "
            "Retrain with the updated 03_train_recon.py."
        )
    recon_in_mean = recon_ckpt["input_mean"].to(DEVICE)
    recon_in_std  = recon_ckpt["input_std"].to(DEVICE)
    print(f"[load] recon.pt  epoch={recon_ckpt.get('epoch','?')}  "
          f"val={recon_ckpt.get('val_total', recon_ckpt.get('val','?'))}  "
          f"feats/det={recon_feat}")

    # Mountain + initial layout
    mountain = load_tr_mountain(
        GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
        east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES,
    )
    N_init_np, U_init_np = mountain.sample_initial_layout(
        n_units=N_DETECTORS, scheme=LAYOUT_INIT_SCHEME,
    )
    N_init = torch.as_tensor(N_init_np, dtype=torch.float32)
    U_init = torch.as_tensor(U_init_np, dtype=torch.float32)
    N_init, U_init = mountain.project_to_mountain(N_init, U_init)
    print(f"[layout] init  N in [{N_init.min():.1f}, {N_init.max():.1f}]  "
          f"Up in [{U_init.min():.1f}, {U_init.max():.1f}]")

    # Keep a copy of the init coordinates for plotting at the end
    N_init_plot = N_init.numpy().copy()
    U_init_plot = U_init.numpy().copy()

    xy_module = LearnableXY(N_init, U_init, device=str(DEVICE))
    xy_module.to(DEVICE)

    optimizer = torch.optim.Adam(xy_module.parameters(), lr=LR)

    log = []
    best_u     = -float("inf")
    best_epoch = -1
    traj = []                                   # (epoch, (100, 2)) snapshots

    for epoch in range(N_OPT_EPOCHS):
        t_epoch = time.time()

        # Sample a random primary batch from the corpus
        idx = torch.randint(0, n_total_primaries, (PRIMARIES_PER_STEP,))
        primary_batch = primary_all[idx].to(DEVICE)
        E_true, theta_true, phi_true = primary_to_physical_labels(primary_batch)

        # Current layout (differentiable)
        x_det, y_det = xy_module()
        xy_per_det = torch.stack([x_det, y_det], dim=-1)                     # (100, 2)
        xy_batch   = xy_per_det.unsqueeze(0).expand(PRIMARIES_PER_STEP, -1, -1)  # (B, 100, 2)

        # FNN forward (frozen)
        pred_ET = fnn(primary_batch, xy_batch)                                # (B, 100, 2)
        E_pred_det = pred_ET[..., 0]                                          # (B, 100)
        T_pred_det = pred_ET[..., 1]                                          # (B, 100)

        # Reconstruction forward (frozen): (x, y, E, T) per detector
        recon_feats = torch.stack(
            [xy_batch[..., 0], xy_batch[..., 1], E_pred_det, T_pred_det],
            dim=-1,
        )                                                                    # (B, 100, 4)
        recon_input = recon_feats.reshape(PRIMARIES_PER_STEP, -1)             # (B, 400)
        recon_input = (recon_input - recon_in_mean) / recon_in_std            # frozen z-score
        pred_norm = recon(recon_input)                                        # (B, 3) tanh
        E_norm, theta_norm, phi_norm = pred_norm[:, 0], pred_norm[:, 1], pred_norm[:, 2]

        # Denormalize + clamp E to the training support [1e5, 1e8] GeV. The
        # clamp prevents log10 of negative numbers when Tanh emits values <0
        # for the E channel. Clamp is differentiable on the active side.
        E_pred_phys, theta_pred, phi_pred = DenormalizeLabels(E_norm, theta_norm, phi_norm)
        E_pred_phys = E_pred_phys.clamp(min=1.0)

        # Reconstructability from the predicted per-detector counts (physical space)
        r = reconstructability(
            torch.expm1(E_pred_det),
            layout_threshold=LAYOUT_THRESHOLD,
            reconstruct_threshold=RECONSTRUCT_THRESHOLD,
        )

        # Composite utility (matches v4's 4-term composite / 1e3)
        u_theta = U_angle(theta_pred, theta_true, r)
        u_phi   = U_angle(phi_pred,   phi_true,   r)
        u_e     = U_E    (E_pred_phys, E_true,    r)
        u_pr    = U_PR(r)
        U = (W_THETA * u_theta + W_PHI * u_phi + W_E * u_e ) / W_DIV
        loss = -U

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(xy_module.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        # Project the updated coordinates back onto the mountain surface
        with torch.no_grad():
            N_cpu  = xy_module.x.detach().cpu()
            Up_cpu = xy_module.y.detach().cpu()
            N_new, Up_new = mountain.project_to_mountain(N_cpu, Up_cpu)
            xy_module.x.data.copy_(N_new.to(DEVICE).to(xy_module.x.dtype))
            xy_module.y.data.copy_(Up_new.to(DEVICE).to(xy_module.y.dtype))

        dt = time.time() - t_epoch
        u_val    = float(U.item())
        r_mean   = float(r.mean().item())
        gn       = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)
        log.append(dict(
            epoch=epoch + 1,
            U=u_val,
            u_theta=float(u_theta.item()), u_phi=float(u_phi.item()),
            u_e=float(u_e.item()), u_pr=float(u_pr.item()),
            r_mean=r_mean, grad_norm=gn, dt=dt,
        ))

        if epoch % LOG_EVERY == 0 or epoch == N_OPT_EPOCHS - 1:
            print(f"[epoch {epoch+1:4d}/{N_OPT_EPOCHS}] "
                  f"U={u_val:+.3f} (θ={u_theta.item():.2f} φ={u_phi.item():.2f} "
                  f"E={u_e.item():.2f} PR={u_pr.item():.2f})  "
                  f"r={r_mean:.3f}  |g|={gn:.2f}  {dt:.2f}s")

        if u_val > best_u:
            best_u     = u_val
            best_epoch = epoch + 1
            torch.save({
                "x": xy_module.x.detach().cpu(),
                "y": xy_module.y.detach().cpu(),
                "epoch": epoch + 1,
                "U": u_val,
            }, os.path.join(OPT_DIR, "layout_best.pt"))

        if (epoch + 1) % SAVE_EVERY == 0 or epoch == N_OPT_EPOCHS - 1:
            snap = torch.stack([
                xy_module.x.detach().cpu(),
                xy_module.y.detach().cpu(),
            ], dim=-1)                                                         # (100, 2)
            traj.append((epoch + 1, snap))

    # Final artifacts
    x_final = xy_module.x.detach().cpu().numpy()
    y_final = xy_module.y.detach().cpu().numpy()
    torch.save({
        "x": torch.as_tensor(x_final),
        "y": torch.as_tensor(y_final),
        "epoch": N_OPT_EPOCHS,
        "U_final": log[-1]["U"] if log else None,
    }, os.path.join(OPT_DIR, "layout_final.pt"))

    if traj:
        snaps = torch.stack([s for _, s in traj], dim=0)                      # (K, 100, 2)
        epochs = torch.tensor([e for e, _ in traj], dtype=torch.long)
        torch.save({"epochs": epochs, "xy": snaps},
                   os.path.join(OPT_DIR, "xy_trajectory.pt"))

    with open(os.path.join(OPT_DIR, "optimize_log.json"), "w") as f:
        json.dump({
            "log": log,
            "best_U": best_u,
            "best_epoch": best_epoch,
            "config": dict(
                n_opt_epochs=N_OPT_EPOCHS,
                primaries_per_step=PRIMARIES_PER_STEP,
                lr=LR, grad_clip=GRAD_CLIP,
                layout_init_scheme=LAYOUT_INIT_SCHEME,
                w_theta=W_THETA, w_phi=W_PHI, w_e=W_E, w_pr=W_PR, w_div=W_DIV,
                layout_threshold=LAYOUT_THRESHOLD,
                reconstruct_threshold=RECONSTRUCT_THRESHOLD,
                seed=SEED,
            ),
        }, f, indent=2)

    _plot_curves(log, os.path.join(OPT_DIR, "optimize_curves.png"))
    _plot_layout(
        x_final, y_final, N_init_plot, U_init_plot, mountain,
        os.path.join(OPT_DIR, "layout_before_after.png"),
    )

    print(f"[done] best U {best_u:+.3f} at epoch {best_epoch}  -> {OPT_DIR}")


if __name__ == "__main__":
    main()
