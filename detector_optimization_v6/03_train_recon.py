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
N_EPOCHS             = 300
LR                   = 3e-5
GRAD_CLIP            = 10.0
VAL_FRAC             = 0.10
SEED                 = 1
NUM_WORKERS          = 0
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── L-BFGS fine-tuning (full-batch, one step, many iterations) ─────────────
LBFGS_LR                 = 1.0
LBFGS_MAX_ITER           = 500
LBFGS_HISTORY_SIZE       = 20


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


def _plot_curves(log, path: str, adam_epochs: int = 0,
                 lbfgs_iter_log=None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Adam entries only (exclude the single lbfgs summary row)
        adam_log = [e for e in log if e.get("phase") != "lbfgs"]
        ep = [e["epoch"] for e in adam_log]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(ep, [e["train"] for e in adam_log], label="train")
        axes[0].plot(ep, [e["val"]   for e in adam_log], label="val")
        axes[1].plot(ep, [e["val_E"]  for e in adam_log], label="val E")
        axes[1].plot(ep, [e["val_th"] for e in adam_log], label="val \u03b8")
        axes[1].plot(ep, [e["val_ph"] for e in adam_log], label="val \u03c6")

        # L-BFGS iterations
        if lbfgs_iter_log:
            lb_ep = [adam_epochs + 1 + e["iter"] for e in lbfgs_iter_log]
            axes[0].plot(lb_ep, [e["loss"]   for e in lbfgs_iter_log],
                         label="L-BFGS train", alpha=0.7)
            axes[0].plot(lb_ep, [e["val"]    for e in lbfgs_iter_log],
                         label="L-BFGS val", alpha=0.7)
            axes[1].plot(lb_ep, [e["val_E"]  for e in lbfgs_iter_log],
                         label="L-BFGS val E", alpha=0.7)
            axes[1].plot(lb_ep, [e["val_th"] for e in lbfgs_iter_log],
                         label="L-BFGS val \u03b8", alpha=0.7)
            axes[1].plot(lb_ep, [e["val_ph"] for e in lbfgs_iter_log],
                         label="L-BFGS val \u03c6", alpha=0.7)
            axes[0].set_yscale("log"); axes[1].set_yscale("log")


        if adam_epochs > 0:
            for ax in axes:
                ax.axvline(adam_epochs, color="gray", linestyle="--", alpha=0.5,
                           label="Adam\u2192L-BFGS")

        axes[0].set_xlabel("epoch / iter"); axes[0].set_ylabel("MSE (normalized labels)")
        axes[0].set_title("total");  axes[0].grid(alpha=0.3); axes[0].legend()
        axes[1].set_xlabel("epoch / iter"); axes[1].set_ylabel("MSE")
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
    print(f"epochs          : {N_EPOCHS}")
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

    with open(os.path.join(RECON_FOLDER, "recon_train_log.json"), "w") as f:
        json.dump({
            "log": log,
            "best_val_total": best_val,
            "best_epoch": best_epoch,
            "config": dict(
                batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, lr=LR,
                grad_clip=GRAD_CLIP, recon_input_features=RECON_INPUT_FEATURES,
                val_frac=VAL_FRAC, seed=SEED,
            ),
        }, f, indent=2)
    _plot_curves(log, os.path.join(RECON_FOLDER, "recon_train_curves.png"))
    print(f"[adam done] best recon val {best_val:.4f} at epoch {best_epoch}")

    # ── Phase 2: L-BFGS fine-tuning (full-batch) ────────────────────────────
    print("\n" + "=" * 72)
    print("Phase 2: L-BFGS fine-tuning (full-batch)")
    print("=" * 72)

    adam_ckpt = torch.load(os.path.join(RECON_FOLDER, "recon.pt"), map_location=DEVICE)
    recon.load_state_dict(adam_ckpt["state_dict"])
    adam_best_val = adam_ckpt["val_total"]
    print(f"[lbfgs] loaded Adam best  epoch={adam_ckpt['epoch']}  "
          f"val={adam_best_val:.6f}")

    # eval() disables dropout; requires_grad stays True
    recon.eval()

    # Move full training set to GPU and pre-compute normalized input
    xy_train = xy[train_idx].to(DEVICE)
    E_train  = E_pred[train_idx].to(DEVICE)
    T_train  = T_pred[train_idx].to(DEVICE)
    tgt_train = target[train_idx].to(DEVICE)
    inp_all  = build_recon_input(xy_train, E_train, T_train)
    inp_all_n = (inp_all - in_mean) / in_std
    del xy_train, E_train, T_train, inp_all  # free intermediates
    print(f"[lbfgs] full train batch on {DEVICE}: {tgt_train.shape[0]} samples")

    lbfgs_optimizer = torch.optim.LBFGS(
        recon.parameters(),
        lr=LBFGS_LR,
        max_iter=LBFGS_MAX_ITER,
        history_size=LBFGS_HISTORY_SIZE,
        line_search_fn="strong_wolfe",
    )

    lbfgs_iter_log = []   # one entry per closure call
    t_lbfgs = time.time()

    def closure():
        lbfgs_optimizer.zero_grad()
        pred = recon(inp_all_n)
        l_E  = F.mse_loss(pred[:, 0], tgt_train[:, 0])
        l_th = F.mse_loss(pred[:, 1], tgt_train[:, 1])
        l_ph = F.mse_loss(pred[:, 2], tgt_train[:, 2])
        loss = l_E + l_th + l_ph
        loss.backward()
        # Validation (no_grad — does not affect L-BFGS gradients)
        with torch.no_grad():
            va_tot, va_E, va_th, va_ph, n_va = 0.0, 0.0, 0.0, 0.0, 0
            for xy_b, E_b, T_b, tgt_b in val_loader:
                xy_b  = xy_b.to(DEVICE, non_blocking=True)
                E_b   = E_b.to(DEVICE,  non_blocking=True)
                T_b   = T_b.to(DEVICE,  non_blocking=True)
                tgt_b = tgt_b.to(DEVICE, non_blocking=True)
                inp  = build_recon_input(xy_b, E_b, T_b)
                inp  = (inp - in_mean) / in_std
                v_pred = recon(inp)
                v_E  = F.mse_loss(v_pred[:, 0], tgt_b[:, 0])
                v_th = F.mse_loss(v_pred[:, 1], tgt_b[:, 1])
                v_ph = F.mse_loss(v_pred[:, 2], tgt_b[:, 2])
                v_l  = v_E + v_th + v_ph
                B = xy_b.shape[0]
                va_tot += v_l.item()   * B
                va_E   += v_E.item()   * B
                va_th  += v_th.item()  * B
                va_ph  += v_ph.item()  * B
                n_va   += B
            va_tot /= max(n_va, 1)
            va_E   /= max(n_va, 1)
            va_th  /= max(n_va, 1)
            va_ph  /= max(n_va, 1)
        it = len(lbfgs_iter_log)
        lbfgs_iter_log.append(dict(
            iter=it, loss=loss.item(),
            mse_E=l_E.item(), mse_th=l_th.item(), mse_ph=l_ph.item(),
            val=va_tot, val_E=va_E, val_th=va_th, val_ph=va_ph,
        ))
        print(f"  [lbfgs iter {it:3d}] loss={loss.item():.6f} "
              f"(E={l_E.item():.6f} \u03b8={l_th.item():.6f} \u03c6={l_ph.item():.6f})  "
              f"val={va_tot:.6f}")
        return loss

    final_loss = lbfgs_optimizer.step(closure)

    # Free GPU memory
    del inp_all_n, tgt_train

    dt_lbfgs = time.time() - t_lbfgs
    n_iters = len(lbfgs_iter_log)
    lbfgs_abort = torch.isnan(final_loss) or torch.isinf(final_loss)

    if lbfgs_abort:
        print(f"[lbfgs] ABORT — NaN/Inf after {n_iters} iters")
        recon.load_state_dict(adam_ckpt["state_dict"])

    print(f"[lbfgs] {n_iters} iterations in {dt_lbfgs:.1f}s")

    # Validation after L-BFGS
    overall_best_val = adam_best_val
    lbfgs_log = []

    if not lbfgs_abort:
        last = lbfgs_iter_log[-1]
        va_tot = last["val"]
        va_E, va_th, va_ph = last["val_E"], last["val_th"], last["val_ph"]

        print(f"[lbfgs val] val={va_tot:.6f} "
              f"(E={va_E:.6f} \u03b8={va_th:.6f} \u03c6={va_ph:.6f})")

        lbfgs_log.append(dict(
            epoch=N_EPOCHS + 1, phase="lbfgs",
            train=last["loss"],
            train_E=last["mse_E"], train_th=last["mse_th"], train_ph=last["mse_ph"],
            val=va_tot, val_E=va_E, val_th=va_th, val_ph=va_ph,
            dt=dt_lbfgs,
        ))

        if va_tot > 2.0 * adam_best_val:
            print(f"[lbfgs] ABORT — val {va_tot:.4f} > 2x Adam best {adam_best_val:.4f}")
            recon.load_state_dict(adam_ckpt["state_dict"])
        elif va_tot < overall_best_val - 1e-6:
            overall_best_val = va_tot
            torch.save({
                "state_dict": recon.state_dict(),
                "epoch": N_EPOCHS + 1,
                "phase": "lbfgs",
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
            print(f"[lbfgs] new best val={va_tot:.6f}  (Adam was {adam_best_val:.6f})")
        else:
            print(f"[lbfgs] no improvement  val={va_tot:.6f} vs Adam {adam_best_val:.6f}")

    # Merge logs and re-save
    full_log = log + lbfgs_log
    with open(os.path.join(RECON_FOLDER, "recon_train_log.json"), "w") as f:
        json.dump({
            "log": full_log,
            "lbfgs_iter_log": lbfgs_iter_log,
            "best_val_total": overall_best_val,
            "best_epoch": best_epoch if overall_best_val == adam_best_val else N_EPOCHS + 1,
            "adam_best_val": adam_best_val,
            "adam_best_epoch": best_epoch,
            "config": dict(
                batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, lr=LR,
                grad_clip=GRAD_CLIP, recon_input_features=RECON_INPUT_FEATURES,
                val_frac=VAL_FRAC, seed=SEED,
                lbfgs_lr=LBFGS_LR, lbfgs_max_iter=LBFGS_MAX_ITER,
                lbfgs_history_size=LBFGS_HISTORY_SIZE,
            ),
        }, f, indent=2)
    _plot_curves(full_log, os.path.join(RECON_FOLDER, "recon_train_curves.png"),
                 adam_epochs=N_EPOCHS, lbfgs_iter_log=lbfgs_iter_log)
    print(f"[done] best recon val {overall_best_val:.4f}  -> {RECON_FOLDER}")


if __name__ == "__main__":
    main()
