"""Train a reconstruction NN on frozen-FNN outputs.

Pipeline (step 3 of the v6 plan):

    fnn (frozen)                                          recon targets
       │                                                         │
       ▼                                                         ▼
  primary → FNN(primary, xy) → (E_pred, T_pred)                 (E_GeV, θ, φ)
                                    │                           normalized to (0, 1)
                                    ▼                           via v3 NormalizeLabels
  recon input per detector = (x, y, E_pred, T_pred)               │
                                    │                             │
                                    └─────────────────────────────┘
                                         v3 Reconstruction
                                       (input_features=4)

Permutation augmentation is applied to the recon input on every batch: a
random per-sample permutation of the 100 detectors (with xy, E, T permuted
together; the target is a scalar 3-vector that stays fixed). This teaches
the flat MLP recon to be approximately permutation-invariant in its output.

Shower-level 90/10 split (matches 02_train_fnn.py). Reuses the trained FNN
from `outputs/v6_run_01/fnn.pt`.

Run:

    cd TambOpt/detector_optimization_v6
    python 03_train_recon.py

Artifacts in `outputs/v6_run_01/`:
    recon.pt               best-val model checkpoint
    recon_train_log.json   per-epoch train/val MSE (+ per-axis val)
    recon_train_curves.png
"""
import json
import math
import os
import sys
import time
from typing import Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset

import modules_v6   # sys.path injection for v3 + v4
from modules_v6.fnn_surrogate import FNNSurrogate
from modules.reconstruction   import Reconstruction, NormalizeLabels
from modules_v6.constants import (
    RECON_FOLDER, TRAINING_DATASET_FOLDER, FNN_FOLDER,
    N_DETECTORS, PRIMARY_DIM, LOG_E_MIN, LOG_E_MAX,
    )

# ── Config ───────────────────────────────────────────────────────────────────

RECON_INPUT_FEATURES = 4   # (x, y, E, T) per detector
BATCH_SIZE           = 256
N_EPOCHS             = 400
LR                   = 3e-5
GRAD_CLIP            = 10.0
EARLY_STOP_PATIENCE  = 200 #TODO 20
VAL_FRAC             = 0.10
SEED                 = 1
NUM_WORKERS          = 0
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def shower_level_split(strategy_ids: torch.Tensor,
                       val_frac: float,
                       seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shower-level split so all 5 strategies for one shower share a split."""
    n_pairs   = int(strategy_ids.shape[0])
    n_strat   = int(strategy_ids.max().item() + 1)
    n_showers = n_pairs // n_strat

    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n_showers, generator=g)
    n_val = max(1, int(round(val_frac * n_showers)))

    is_val = torch.zeros(n_showers, dtype=torch.bool)
    is_val[perm[:n_val]] = True

    all_idx = torch.arange(n_pairs, dtype=torch.long)
    shower_of_pair = all_idx - strategy_ids * n_showers
    val_mask   = is_val[shower_of_pair]
    train_mask = ~val_mask
    return (torch.nonzero(train_mask).squeeze(-1),
            torch.nonzero(val_mask).squeeze(-1))


def primary_to_physical_labels(primary: torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert 5-D primary encoding to (E_GeV, theta_rad, phi_rad).

    primary columns: [dir_x, dir_y, dir_z, log_e_norm, pdg]
        dir is (sin θ cos φ, sin θ sin φ, cos θ)
        log_e_norm = (log10(E) - LOG_E_MIN) / (LOG_E_MAX - LOG_E_MIN)
    """
    dir_x = primary[:, 0]
    dir_y = primary[:, 1]
    dir_z = primary[:, 2].clamp(-1.0, 1.0)
    log_e_norm = primary[:, 3]

    log_e = log_e_norm * (LOG_E_MAX - LOG_E_MIN) + LOG_E_MIN
    E_gev = torch.pow(10.0, log_e)
    theta = torch.arccos(dir_z)

    phi = torch.atan2(dir_y, dir_x)
    # atan2 returns [-pi, pi]; wrap to [0, 2pi] to match v3's NormalizeLabels
    two_pi = 2.0 * math.pi
    phi = torch.where(phi < 0, phi + two_pi, phi)
    return E_gev, theta, phi


@torch.no_grad()
def compute_fnn_predictions(model: FNNSurrogate,
                            primary: torch.Tensor,
                            xy:      torch.Tensor,
                            device:  torch.device,
                            batch_size: int = 1024
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run FNN forward on the whole corpus. Returns CPU tensors.

    primary and xy are CPU tensors; only batches are moved to device.
    """
    model.eval()
    N = int(primary.shape[0])
    E_pred = torch.empty((N, N_DETECTORS), dtype=torch.float32)
    T_pred = torch.empty((N, N_DETECTORS), dtype=torch.float32)
    for lo in range(0, N, batch_size):
        hi = min(lo + batch_size, N)
        p_b  = primary[lo:hi].to(device, non_blocking=True)
        xy_b = xy[lo:hi].to(device, non_blocking=True)
        pred = model(p_b, xy_b)                 # (B, 100, 2)
        E_pred[lo:hi] = pred[..., 0].cpu()
        T_pred[lo:hi] = pred[..., 1].cpu()
    return E_pred, T_pred


def permute_detectors_recon(xy: torch.Tensor,
                            E:  torch.Tensor,
                            T:  torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent per-sample random permutation of the 100 detectors.

    The recon target is a scalar 3-vector (E, θ, φ) — invariant — so we do NOT
    permute the target. The network learns to produce the same output for any
    ordering of its per-detector features.
    """
    B, n_det, _ = xy.shape
    rand_key = torch.rand(B, n_det, device=xy.device)
    perms = torch.argsort(rand_key, dim=1)
    idx_xy = perms.unsqueeze(-1).expand(-1, -1, 2)
    xy_p = torch.gather(xy, 1, idx_xy)
    E_p  = torch.gather(E,  1, perms)
    T_p  = torch.gather(T,  1, perms)
    return xy_p, E_p, T_p


def build_recon_input(xy: torch.Tensor,
                      E:  torch.Tensor,
                      T:  torch.Tensor) -> torch.Tensor:
    """(B, 100, 2) xy + (B, 100) E + (B, 100) T -> (B, 400) flat unnormalized."""
    feats = torch.stack([xy[..., 0], xy[..., 1], E, T], dim=-1)  # (B, 100, 4)
    return feats.reshape(feats.shape[0], -1)


def compute_recon_input_stats(xy: torch.Tensor,
                              E:  torch.Tensor,
                              T:  torch.Tensor
                              ) -> "tuple[torch.Tensor, torch.Tensor]":
    """Z-score stats over the (x, y, E_pred, T_pred) flat input.

    v3's `Reconstruction` has no internal normalization — raw mountain-scale
    xy would saturate Tanh. v4's optimization driver solves this by z-scoring
    the per-detector feature vector with FROZEN training-time stats. We do
    the same here: stats are computed once over the recon training corpus
    and saved alongside the checkpoint.
    """
    flat = build_recon_input(xy, E, T)                       # (N, 400)
    mean = flat.mean(dim=0)
    std  = flat.std(dim=0).clamp(min=1e-8)
    return mean, std


def _plot_curves(log, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep = [e["epoch"] for e in log]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
        axes[0].plot(ep, [e["train"] for e in log], label="train")
        axes[0].plot(ep, [e["val"]   for e in log], label="val")
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("MSE (normalized labels)")
        axes[0].set_title("total");  axes[0].grid(alpha=0.3); axes[0].legend()
        axes[1].plot(ep, [e["val_E"]  for e in log], label="val E")
        axes[1].plot(ep, [e["val_th"] for e in log], label="val θ")
        axes[1].plot(ep, [e["val_ph"] for e in log], label="val φ")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("MSE")
        axes[1].set_title("per-axis val"); axes[1].grid(alpha=0.3); axes[1].legend()
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] skipped ({exc!r})")


def main():
    
    print("=" * 72)
    print("v6/03_train_recon.py")
    print("=" * 72)
    os.makedirs(RECON_FOLDER, exist_ok=True)
    print(f"training data   : {TRAINING_DATASET_FOLDER}")
    print(f"fnn checkpoint  : {FNN_FOLDER}")
    print(f"device          : {DEVICE}")
    print(f"batch           : {BATCH_SIZE}")
    print(f"epochs          : {N_EPOCHS}  (early stop patience {EARLY_STOP_PATIENCE})")
    print(f"lr              : {LR}")
    print(f"feats/det       : {RECON_INPUT_FEATURES}  (x, y, E, T)")
    print(f"seed            : {SEED}")

    # Load corpus
    t0 = time.time()
    primary    = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()
    xy         = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "xy.pt")).float()
    strat_ids  = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "strategy_ids.pt")).long()
    norm_stats = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "norm_stats.pt"))
    print(f"[load] corpus in {time.time() - t0:.1f}s  primary={tuple(primary.shape)}")

    # Load frozen FNN
    fnn_ckpt = torch.load(os.path.join(FNN_FOLDER, "fnn.pt"), map_location=DEVICE)
    fnn = FNNSurrogate(n_det=N_DETECTORS, primary_dim=PRIMARY_DIM,
                       hidden=512, dropout=0.1).to(DEVICE)
    fnn.load_state_dict(fnn_ckpt["state_dict"])
    fnn.set_normalization(norm_stats)
    fnn.eval()
    for p in fnn.parameters():
        p.requires_grad_(False)
    print(f"[load] fnn.pt  epoch={fnn_ckpt.get('epoch','?')}  "
          f"val_total={fnn_ckpt.get('val_total', fnn_ckpt.get('val','?'))}")

    # Predict (E, T) on the full corpus once (deterministic, FNN is in eval)
    t0 = time.time()
    E_pred, T_pred = compute_fnn_predictions(fnn, primary, xy, DEVICE, batch_size=1024)
    print(f"[fnn] predictions in {time.time() - t0:.1f}s  "
          f"E mean={E_pred.mean():.3g} std={E_pred.std():.3g}  "
          f"T mean={T_pred.mean():.3g} std={T_pred.std():.3g}")

    # Build normalized recon targets
    E_gev, theta, phi = primary_to_physical_labels(primary)
    E_n, theta_n, phi_n = NormalizeLabels(E_gev, theta, phi)
    target = torch.stack([E_n, theta_n, phi_n], dim=1)   # (N, 3)
    print(f"[target] E_n in [{E_n.min():.3f}, {E_n.max():.3f}]  "
          f"θ_n in [{theta_n.min():.3f}, {theta_n.max():.3f}]  "
          f"φ_n in [{phi_n.min():.3f}, {phi_n.max():.3f}]")

    # Shower-level split
    train_idx, val_idx = shower_level_split(strat_ids, VAL_FRAC, SEED)
    print(f"[split] train pairs={len(train_idx)}  val pairs={len(val_idx)}")

    # Compute recon input stats over the TRAIN subset only — these are frozen
    # and reused by 04_optimize.py so mountain-scale xy doesn't saturate Tanh.
    in_mean, in_std = compute_recon_input_stats(
        xy[train_idx], E_pred[train_idx], T_pred[train_idx],
    )
    print(f"[norm] recon in_mean[:4] = {in_mean[:4].tolist()}  "
          f"in_std[:4] = {in_std[:4].tolist()}")
    in_mean = in_mean.to(DEVICE)
    in_std  = in_std.to(DEVICE)

    full_ds  = TensorDataset(xy, E_pred, T_pred, target)
    train_ds = Subset(full_ds, train_idx.tolist())
    val_ds   = Subset(full_ds, val_idx.tolist())
    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    # Model
    torch.manual_seed(SEED)
    recon = Reconstruction(
        input_features=RECON_INPUT_FEATURES,
        num_detectors=N_DETECTORS,
        hidden_lay1=256, hidden_lay2=128, hidden_lay3=32,
        output_dim=3,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in recon.parameters() if p.requires_grad)
    print(f"[model] recon params={n_params:,}")

    optimizer = torch.optim.Adam(recon.parameters(), lr=LR)

    log = []
    best_val   = float("inf")
    best_epoch = -1
    patience   = 0

    for epoch in range(N_EPOCHS):
        t_epoch = time.time()
        recon.train()
        tr_tot, tr_E, tr_th, tr_ph, n_tr = 0.0, 0.0, 0.0, 0.0, 0
        for xy_b, E_b, T_b, tgt_b in train_loader:
            xy_b  = xy_b.to(DEVICE, non_blocking=True)
            E_b   = E_b.to(DEVICE,  non_blocking=True)
            T_b   = T_b.to(DEVICE,  non_blocking=True)
            tgt_b = tgt_b.to(DEVICE, non_blocking=True)

            # Permutation augmentation (input-only; target stays fixed)
            xy_b, E_b, T_b = permute_detectors_recon(xy_b, E_b, T_b)

            inp  = build_recon_input(xy_b, E_b, T_b)   # (B, 400)
            inp  = (inp - in_mean) / in_std             # frozen z-score
            pred = recon(inp)                           # (B, 3) tanh
            l_E  = F.mse_loss(pred[:, 0], tgt_b[:, 0])
            l_th = F.mse_loss(pred[:, 1], tgt_b[:, 1])
            l_ph = F.mse_loss(pred[:, 2], tgt_b[:, 2])
            loss = l_E + l_th + l_ph

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(recon.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

            B = xy_b.shape[0]
            tr_tot += loss.item() * B
            tr_E   += l_E.item()  * B
            tr_th  += l_th.item() * B
            tr_ph  += l_ph.item() * B
            n_tr   += B
        tr_tot /= max(n_tr, 1)
        tr_E   /= max(n_tr, 1)
        tr_th  /= max(n_tr, 1)
        tr_ph  /= max(n_tr, 1)

        recon.eval()
        va_tot, va_E, va_th, va_ph, n_va = 0.0, 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for xy_b, E_b, T_b, tgt_b in val_loader:
                xy_b  = xy_b.to(DEVICE, non_blocking=True)
                E_b   = E_b.to(DEVICE,  non_blocking=True)
                T_b   = T_b.to(DEVICE,  non_blocking=True)
                tgt_b = tgt_b.to(DEVICE, non_blocking=True)
                inp  = build_recon_input(xy_b, E_b, T_b)
                inp  = (inp - in_mean) / in_std
                pred = recon(inp)
                l_E  = F.mse_loss(pred[:, 0], tgt_b[:, 0])
                l_th = F.mse_loss(pred[:, 1], tgt_b[:, 1])
                l_ph = F.mse_loss(pred[:, 2], tgt_b[:, 2])
                l    = l_E + l_th + l_ph
                B = xy_b.shape[0]
                va_tot += l.item()    * B
                va_E   += l_E.item()  * B
                va_th  += l_th.item() * B
                va_ph  += l_ph.item() * B
                n_va   += B
        va_tot /= max(n_va, 1)
        va_E   /= max(n_va, 1)
        va_th  /= max(n_va, 1)
        va_ph  /= max(n_va, 1)

        dt = time.time() - t_epoch
        print(f"[epoch {epoch+1:3d}/{N_EPOCHS}] "
              f"train={tr_tot:.4f} val={va_tot:.4f} "
              f"(E={va_E:.4f} θ={va_th:.4f} φ={va_ph:.4f})  {dt:.1f}s")
        log.append(dict(
            epoch=epoch + 1,
            train=tr_tot, train_E=tr_E, train_th=tr_th, train_ph=tr_ph,
            val=va_tot,   val_E=va_E,   val_th=va_th,   val_ph=va_ph,
            dt=dt,
        ))

        if va_tot < best_val - 1e-6:
            best_val   = va_tot
            best_epoch = epoch + 1
            patience   = 0
            torch.save({
                "state_dict": recon.state_dict(),
                "epoch": epoch + 1,
                "val_total": va_tot,
                "val_E": va_E, "val_th": va_th, "val_ph": va_ph,
                "input_features": RECON_INPUT_FEATURES,
                "num_detectors": N_DETECTORS,
                "input_mean": in_mean.detach().cpu(),
                "input_std":  in_std.detach().cpu(),
                "config": dict(
                    hidden_lay1=256, hidden_lay2=128, hidden_lay3=32, output_dim=3,
                ),
            }, os.path.join(RECON_FOLDER, "recon.pt"))
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f"[early-stop] val plateau at epoch {epoch+1}  "
                      f"(best {best_val:.4f} at epoch {best_epoch})")
                break

    with open(os.path.join(RECON_FOLDER, "recon_train_log.json"), "w") as f:
        json.dump({
            "log": log,
            "best_val_total": best_val,
            "best_epoch": best_epoch,
            "config": dict(
                batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, lr=LR,
                grad_clip=GRAD_CLIP, early_stop_patience=EARLY_STOP_PATIENCE,
                recon_input_features=RECON_INPUT_FEATURES,
                val_frac=VAL_FRAC, seed=SEED,
            ),
        }, f, indent=2)
    _plot_curves(log, os.path.join(RECON_FOLDER, "recon_train_curves.png"))
    print(f"[done] best recon val {best_val:.4f} at epoch {best_epoch}  -> {RECON_FOLDER}")


if __name__ == "__main__":
    main()
