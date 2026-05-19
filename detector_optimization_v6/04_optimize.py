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
    TRAINING_DATASET_FOLDER, FNN_FOLDER, RECON_FOLDER, OPT_FOLDER,
    LOG_E_MIN, LOG_E_MAX,
)
from modules_v6.reconstruction   import Reconstruction
from modules.utility_functions   import reconstructability, U_E, U_angle, U_PR
from modules.layout_optimization import LearnableXY
from modules_v4.tr_geometry      import load_tr_mountain


# ── Config ───────────────────────────────────────────────────────────────────
# Each entry in INIT_SCHEMES is a separate optimization run; outputs go to
# `<OPT_FOLDER>_{scheme}` (one sibling dir per init) on the run-storage
# filesystem so they sit alongside the v6_run_01..03 artifacts. Set to a
# single-element tuple (e.g. ("center",)) to run only one scheme.
INIT_SCHEMES = ("grid", "center")
OPT_DIR_TEMPLATE = OPT_FOLDER + "_{scheme}"

N_OPT_EPOCHS       = 10000
PRIMARIES_PER_STEP = 256
LR                 = 1              # v3/v4 use lr=10; MLP Jacobian is larger
GRAD_CLIP          = 100.0

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
    E_gev = torch.exp(log_e) - 1.0
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


def _run_optimization(*,
                      init_scheme: str,
                      opt_dir: str,
                      n_epochs: int,
                      primaries_per_step: int,
                      primary_source,           # callable(batch_size) -> (B, 5) tensor on DEVICE
                      mountain,
                      fnn,
                      recon,
                      log_every: int = LOG_EVERY,
                      save_every: int = SAVE_EVERY):
    """Run one optimization cycle and save artifacts to `opt_dir`.

    Returns (best_U, best_epoch). Setup (FNN/recon/mountain/corpus) is done
    once by the caller and shared across runs.
    """
    os.makedirs(opt_dir, exist_ok=True)

    print("-" * 72)
    print(f"[run] init_scheme={init_scheme}  epochs={n_epochs}  "
          f"primaries/step={primaries_per_step}")
    print(f"[run] opt_dir={opt_dir}")

    # Initial layout for this run
    N_init_np, U_init_np = mountain.sample_initial_layout(
        n_units=N_DETECTORS, scheme=init_scheme,
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

    for epoch in range(n_epochs):
        t_epoch = time.time()

        # Primary batch (random sample in normal mode, fixed shower in test mode)
        primary_batch = primary_source(primaries_per_step)
        E_true, theta_true, phi_true = primary_to_physical_labels(primary_batch)

        # Current layout (differentiable)
        x_det, y_det = xy_module()
        xy_per_det = torch.stack([x_det, y_det], dim=-1)                     # (100, 2)
        xy_batch   = xy_per_det.unsqueeze(0).expand(primaries_per_step, -1, -1)  # (B, 100, 2)

        # FNN forward (frozen)
        pred_ET = fnn(primary_batch, xy_batch)                                # (B, 100, 2)
        E_pred_det = pred_ET[..., 0]                                          # (B, 100)
        T_pred_det = pred_ET[..., 1]                                          # (B, 100)

        # Reconstruction forward (frozen): (x, y, E, T) per detector
        recon_feats = torch.stack(
            [xy_batch[..., 0], xy_batch[..., 1], E_pred_det, T_pred_det],
            dim=-1,
        )                                                                    # (B, 100, 4)
        recon_input = recon_feats.reshape(primaries_per_step, -1)             # (B, 400) raw
        pred = recon(recon_input)                                             # (B, 4) raw primary encoding
        E_pred_phys, theta_pred, phi_pred = primary_to_physical_labels(pred)
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

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            print(f"[epoch {epoch+1:4d}/{n_epochs}] "
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
            }, os.path.join(opt_dir, "layout_best.pt"))

        if (epoch + 1) % save_every == 0 or epoch == n_epochs - 1:
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
        "epoch": n_epochs,
        "U_final": log[-1]["U"] if log else None,
    }, os.path.join(opt_dir, "layout_final.pt"))

    if traj:
        snaps = torch.stack([s for _, s in traj], dim=0)                      # (K, 100, 2)
        epochs = torch.tensor([e for e, _ in traj], dtype=torch.long)
        torch.save({"epochs": epochs, "xy": snaps},
                   os.path.join(opt_dir, "xy_trajectory.pt"))

    with open(os.path.join(opt_dir, "optimize_log.json"), "w") as f:
        json.dump({
            "log": log,
            "best_U": best_u,
            "best_epoch": best_epoch,
            "config": dict(
                n_opt_epochs=n_epochs,
                primaries_per_step=primaries_per_step,
                lr=LR, grad_clip=GRAD_CLIP,
                layout_init_scheme=init_scheme,
                w_theta=W_THETA, w_phi=W_PHI, w_e=W_E, w_pr=W_PR, w_div=W_DIV,
                layout_threshold=LAYOUT_THRESHOLD,
                reconstruct_threshold=RECONSTRUCT_THRESHOLD,
                seed=SEED,
            ),
        }, f, indent=2)

    _plot_curves(log, os.path.join(opt_dir, "optimize_curves.png"))
    _plot_layout(
        x_final, y_final, N_init_plot, U_init_plot, mountain,
        os.path.join(opt_dir, "layout_before_after.png"),
    )

    print(f"[done] best U {best_u:+.3f} at epoch {best_epoch}  -> {opt_dir}")
    return best_u, best_epoch


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 72)
    print("v6/04_optimize.py")
    print("=" * 72)
    print(f"geometry         : {GEOMETRY_PATH}")
    print(f"training dataset : {TRAINING_DATASET_FOLDER}")
    print(f"fnn checkpoint   : {FNN_FOLDER}")
    print(f"recon checkpoint : {RECON_FOLDER}")
    print(f"device           : {DEVICE}")
    print(f"grad clip        : {GRAD_CLIP}")
    print(f"init schemes     : {INIT_SCHEMES}")
    print(f"epochs/run       : {N_OPT_EPOCHS}")
    print(f"primaries/step   : {PRIMARIES_PER_STEP}")

    # Load corpus (primaries only — we sample batches each epoch)
    primary_all = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()
    n_total_primaries = int(primary_all.shape[0])
    print(f"[load] {n_total_primaries} primaries")

    # Frozen FNN — read width + dropout from the ckpt config and prefer the
    # ckpt's norm_stats over the disk file (02_train_fnn.py modifies T stats
    # in-memory for log-T training and saves them inside fnn.pt; the on-disk
    # norm_stats.pt still has raw-T values, which would mismatch the trained
    # model's denormalization).
    fnn_ckpt = torch.load(os.path.join(FNN_FOLDER, "fnn.pt"), map_location=DEVICE)
    fnn_cfg  = fnn_ckpt.get("config", {})
    fnn_hidden  = int(fnn_cfg.get("hidden", 512))
    fnn_dropout = float(fnn_cfg.get("dropout", 0.1))
    fnn = FNNSurrogate(n_det=N_DETECTORS, primary_dim=PRIMARY_DIM,
                       hidden=fnn_hidden, dropout=fnn_dropout).to(DEVICE)
    fnn.load_state_dict(fnn_ckpt["state_dict"])
    norm_stats = fnn_ckpt.get(
        "norm_stats",
        torch.load(os.path.join(TRAINING_DATASET_FOLDER, "norm_stats.pt")),
    )
    fnn.set_normalization(norm_stats)
    fnn.eval()
    for p in fnn.parameters():
        p.requires_grad_(False)
    print(f"[load] fnn.pt    epoch={fnn_ckpt.get('epoch','?')}  "
          f"val={fnn_ckpt.get('val_total', fnn_ckpt.get('val','?'))}  "
          f"hidden={fnn_hidden}")

    # Frozen reconstruction
    recon_ckpt = torch.load(os.path.join(RECON_FOLDER, "recon.pt"), map_location=DEVICE)
    recon_feat = int(recon_ckpt.get("input_features", 4))
    recon_nd   = int(recon_ckpt.get("num_detectors", N_DETECTORS))
    cfg = recon_ckpt.get("config", {})
    recon = Reconstruction(
        n_det=recon_nd,
        input_features=recon_feat,
        output_dim=int(cfg.get("output_dim", 4)),
        hidden=int(cfg.get("hidden", 512)),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(DEVICE)
    recon.load_state_dict(recon_ckpt["state_dict"])
    recon.set_normalization(
        in_mean  = recon_ckpt["input_mean" ].to(DEVICE),
        in_std   = recon_ckpt["input_std"  ].to(DEVICE),
        out_mean = recon_ckpt["target_mean"].to(DEVICE),
        out_std  = recon_ckpt["target_std" ].to(DEVICE),
    )
    recon.eval()
    for p in recon.parameters():
        p.requires_grad_(False)
    print(f"[load] recon.pt  epoch={recon_ckpt.get('epoch','?')}  "
          f"val={recon_ckpt.get('val_total', recon_ckpt.get('val','?'))}  "
          f"feats/det={recon_feat}")

    # Mountain (geometry) — built once, shared across runs
    mountain = load_tr_mountain(
        GEOMETRY_PATH, GEOMETRY_GROUP, DET_KEY,
        east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES,
    )

    def random_batch_source(batch_size: int) -> torch.Tensor:
        idx = torch.randint(0, n_total_primaries, (batch_size,))
        return primary_all[idx].to(DEVICE)

    # One full run per entry in INIT_SCHEMES, output dirs disambiguated by
    # scheme name so grid/center trajectories can be compared side-by-side.
    for scheme in INIT_SCHEMES:
        # Re-seed before each scheme so the only difference between runs is
        # the initial layout; sampled primaries and any other RNG draws are
        # otherwise identical across runs.
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        opt_dir = OPT_DIR_TEMPLATE.format(scheme=scheme)
        print()
        print("=" * 72)
        print(f"init scheme: {scheme}")
        print("=" * 72)
        _run_optimization(
            init_scheme=scheme,
            opt_dir=opt_dir,
            n_epochs=N_OPT_EPOCHS,
            primaries_per_step=PRIMARIES_PER_STEP,
            primary_source=random_batch_source,
            mountain=mountain,
            fnn=fnn,
            recon=recon,
        )


if __name__ == "__main__":
    main()
