"""Train the FNN surrogate on the v6_run_01 corpus.

Loss is MSE in z-score-normalized output space so E and T channels get equal
weight regardless of their physical scales. Every batch applies an independent
random permutation to each sample's detector order (input xy and target
(E, T) permuted the same way) — this teaches the flat MLP to be approximately
permutation-equivariant by augmentation. Train/val split is shower-level, so
the 5 layout variants of the same shower never leak across sets.

Run from the v6 folder:

    cd TambOpt/detector_optimization_v6
    python 02_train_fnn.py

Artifacts land in `outputs/v6_run_01/`:
    fnn.pt              best-val model checkpoint (state_dict + norm_stats + meta)
    fnn_train_log.json  per-epoch train/val MSE (total + per-channel)
    fnn_train_curves.png
"""
import json
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

import modules_v6  # triggers sys.path injection for v3 + v4
from modules_v6.fnn_surrogate import FNNSurrogate
from modules_v6.constants import (
    N_DETECTORS, PRIMARY_DIM, 
    TRAINING_DATASET_FOLDER, FNN_FOLDER
    )


# ── Config ───────────────────────────────────────────────────────────────────
BATCH_SIZE          = 256
N_EPOCHS            = 100
LR                  = 1e-5
LR_MIN              = 1e-7
GRAD_CLIP           = 10.0
VAL_FRAC            = 0.10
SEED                = 0
NUM_WORKERS         = 0   # set >0 if disk I/O becomes the bottleneck
DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── L-BFGS fine-tuning (full-batch, one step, many iterations) ─────────────
LBFGS_LR                 = 1.0
LBFGS_MAX_ITER           = 500
LBFGS_HISTORY_SIZE       = 20


def shower_level_split(strategy_ids: torch.Tensor,
                       val_frac: float,
                       seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shower-level 90/10 split. All 5 strategy entries of a shower share a split.

    The dataset builder lays pairs out in strategy-major order: entry k under
    strategy s is at position `s * n_showers + i` for shower i. So the shower
    index of pair k is `k - strategy_ids[k] * n_showers`.
    """
    n_pairs  = int(strategy_ids.shape[0])
    n_strat  = int(strategy_ids.max().item() + 1)
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


def permute_detectors_batch(xy: torch.Tensor,
                            E:  torch.Tensor,
                            T:  torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply an independent random permutation of the 100 detectors per sample.

    Augmentation for approximate permutation equivariance. Since we permute
    xy and (E, T) with the SAME permutation, the label mapping stays correct.
    """
    B, n_det, _ = xy.shape
    # argsort of uniform noise produces a different permutation per row
    rand_key = torch.rand(B, n_det, device=xy.device)
    perms = torch.argsort(rand_key, dim=1)            # (B, n_det)
    idx_xy = perms.unsqueeze(-1).expand(-1, -1, 2)    # (B, n_det, 2)
    xy_p = torch.gather(xy, 1, idx_xy)
    E_p  = torch.gather(E,  1, perms)
    T_p  = torch.gather(T,  1, perms)
    return xy_p, E_p, T_p


def mse_normalized(pred: torch.Tensor,
                   E_tgt: torch.Tensor,
                   T_tgt: torch.Tensor,
                   out_mean: torch.Tensor,
                   out_std:  torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MSE in the same z-score space the FNN's forward normalizes to.

    Both `pred` (unnormalized) and `target` are passed through the same
    `(x - out_mean) / out_std` before the MSE so E and T get equal weight.
    Returns `(total, mse_E, mse_T)`.
    """
    E_pred = pred[..., 0]
    T_pred = pred[..., 1]
    pred_flat   = torch.cat([E_pred, T_pred], dim=1)      # (B, 200)
    target_flat = torch.cat([E_tgt,  T_tgt],  dim=1)      # (B, 200)

    pred_n   = (pred_flat   - out_mean) / out_std
    target_n = (target_flat - out_mean) / out_std

    n_det = E_tgt.shape[1]
    mse_E = F.mse_loss(pred_n[:, :n_det], target_n[:, :n_det])
    mse_T = F.mse_loss(pred_n[:, n_det:], target_n[:, n_det:])
    total = 0.5 * (mse_E + mse_T)
    return total, mse_E, mse_T


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
        axes[1].plot(ep, [e["train_E"] for e in adam_log], label="train E", linestyle="--")
        axes[1].plot(ep, [e["val_E"]   for e in adam_log], label="val E")
        axes[1].plot(ep, [e["train_T"] for e in adam_log], label="train T", linestyle="--")
        axes[1].plot(ep, [e["val_T"]   for e in adam_log], label="val T")

        # L-BFGS iterations (train only — no per-iter val)
        if lbfgs_iter_log:
            lb_ep = [adam_epochs + 1 + e["iter"] for e in lbfgs_iter_log]
            axes[0].plot(lb_ep, [e["loss"]  for e in lbfgs_iter_log],
                         label="L-BFGS train", alpha=0.7)
            axes[0].plot(lb_ep, [e["val"]   for e in lbfgs_iter_log],
                         label="L-BFGS val", alpha=0.7)
            axes[1].plot(lb_ep, [e["mse_E"] for e in lbfgs_iter_log],
                         label="L-BFGS train E", linestyle="--", alpha=0.7)
            axes[1].plot(lb_ep, [e["mse_T"] for e in lbfgs_iter_log],
                         label="L-BFGS train T", linestyle="--", alpha=0.7)
            axes[1].plot(lb_ep, [e["val_E"] for e in lbfgs_iter_log],
                         label="L-BFGS val E", alpha=0.7)
            axes[1].plot(lb_ep, [e["val_T"] for e in lbfgs_iter_log],
                         label="L-BFGS val T", alpha=0.7)

        if adam_epochs > 0:
            for ax in axes:
                ax.axvline(adam_epochs, color="gray", linestyle="--", alpha=0.5,
                           label="Adam\u2192L-BFGS")

        axes[0].set_xlabel("epoch / iter"); axes[0].set_ylabel("MSE (z-scored)")
        axes[0].set_title("total");  axes[0].grid(alpha=0.3); axes[0].legend()
        axes[1].set_xlabel("epoch / iter"); axes[1].set_ylabel("MSE (z-scored)")
        axes[1].set_title("per-channel"); axes[1].grid(alpha=0.3); axes[1].legend()
        axes[0].set_yscale("log"); axes[1].set_yscale("log")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] skipped ({exc!r})")


def main():
    print("=" * 72)
    print("v6/02_train_fnn.py")
    print("=" * 72)
    os.makedirs(FNN_FOLDER, exist_ok=True)
    
    print(f"data input dir  : {TRAINING_DATASET_FOLDER}")
    print(f"fnn output      : {FNN_FOLDER}")
    print(f"device          : {DEVICE}")
    print(f"batch           : {BATCH_SIZE}")
    print(f"epochs          : {N_EPOCHS}")
    print(f"lr              : {LR} -> {LR_MIN} cosine")
    print(f"val frac        : {VAL_FRAC} (shower-level)")
    print(f"seed            : {SEED}")

    # Load corpus
    t0 = time.time()
    primary    = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()
    xy         = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "xy.pt")).float()
    E_all      = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "E.pt")).float()
    T_all      = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "T.pt")).float()
    strat_ids  = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "strategy_ids.pt")).long()
    norm_stats = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "norm_stats.pt"))
    print(f"[load] corpus in {time.time() - t0:.1f}s  "
          f"primary={tuple(primary.shape)}  xy={tuple(xy.shape)}  "
          f"E={tuple(E_all.shape)}  T={tuple(T_all.shape)}")

    # Shower-level split
    train_idx, val_idx = shower_level_split(strat_ids, VAL_FRAC, SEED)
    print(f"[split] train pairs={len(train_idx)}  val pairs={len(val_idx)}")

    full_ds  = TensorDataset(primary, xy, E_all, T_all)
    train_ds = Subset(full_ds, train_idx.tolist())
    val_ds   = Subset(full_ds, val_idx.tolist())
    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    # Model
    torch.manual_seed(SEED)
    model = FNNSurrogate(
        n_det=N_DETECTORS, primary_dim=PRIMARY_DIM,
        hidden=512, dropout=0.1,
    ).to(DEVICE)
    model.set_normalization(norm_stats)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] params={n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=LR_MIN,
    )

    log = []
    best_val   = float("inf")
    best_epoch = -1

    for epoch in range(N_EPOCHS):
        t_epoch = time.time()
        model.train()
        tr_tot, tr_E, tr_T, n_tr = 0.0, 0.0, 0.0, 0
        for p_b, xy_b, E_b, T_b in train_loader:
            p_b  = p_b.to(DEVICE, non_blocking=True)
            xy_b = xy_b.to(DEVICE, non_blocking=True)
            E_b  = E_b.to(DEVICE, non_blocking=True)
            T_b  = T_b.to(DEVICE, non_blocking=True)

            # Permutation augmentation: independent perm per sample in the batch
            xy_b, E_b, T_b = permute_detectors_batch(xy_b, E_b, T_b)

            pred = model(p_b, xy_b)                # (B, 100, 2) unnormalized
            loss, mE, mT = mse_normalized(
                pred, E_b, T_b, model.out_mean, model.out_std,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

            B = p_b.shape[0]
            tr_tot += loss.item() * B
            tr_E   += mE.item()   * B
            tr_T   += mT.item()   * B
            n_tr   += B
        scheduler.step()
        tr_tot /= max(n_tr, 1)
        tr_E   /= max(n_tr, 1)
        tr_T   /= max(n_tr, 1)

        model.eval()
        va_tot, va_E, va_T, n_va = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for p_b, xy_b, E_b, T_b in val_loader:
                p_b  = p_b.to(DEVICE, non_blocking=True)
                xy_b = xy_b.to(DEVICE, non_blocking=True)
                E_b  = E_b.to(DEVICE, non_blocking=True)
                T_b  = T_b.to(DEVICE, non_blocking=True)
                pred = model(p_b, xy_b)
                loss, mE, mT = mse_normalized(
                    pred, E_b, T_b, model.out_mean, model.out_std,
                )
                B = p_b.shape[0]
                va_tot += loss.item() * B
                va_E   += mE.item()   * B
                va_T   += mT.item()   * B
                n_va   += B
        va_tot /= max(n_va, 1)
        va_E   /= max(n_va, 1)
        va_T   /= max(n_va, 1)

        dt = time.time() - t_epoch
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch+1:3d}/{N_EPOCHS}] "
              f"train={tr_tot:.4f} (E={tr_E:.4f} T={tr_T:.4f})  "
              f"val={va_tot:.4f} (E={va_E:.4f} T={va_T:.4f})  "
              f"lr={lr_now:.1e}  {dt:.1f}s")
        log.append(dict(
            epoch=epoch + 1,
            train=tr_tot, train_E=tr_E, train_T=tr_T,
            val=va_tot,   val_E=va_E,   val_T=va_T,
            lr=lr_now, dt=dt,
        ))

        if va_tot < best_val - 1e-5:
            best_val   = va_tot
            best_epoch = epoch + 1
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_total": va_tot,
                "val_E": va_E,
                "val_T": va_T,
                "norm_stats": norm_stats,
                "config": dict(
                    n_det=N_DETECTORS, primary_dim=PRIMARY_DIM,
                    hidden=512, dropout=0.1,
                ),
            }, os.path.join(FNN_FOLDER, "fnn.pt"))

    with open(os.path.join(FNN_FOLDER, "fnn_train_log.json"), "w") as f:
        json.dump({
            "log": log,
            "best_val_total": best_val,
            "best_epoch": best_epoch,
            "config": dict(
                batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, lr=LR, lr_min=LR_MIN,
                grad_clip=GRAD_CLIP, val_frac=VAL_FRAC, seed=SEED,
            ),
        }, f, indent=2)
    _plot_curves(log, os.path.join(FNN_FOLDER, "fnn_train_curves.png"))
    print(f"[adam done] best val {best_val:.4f} at epoch {best_epoch}")

    # ── Phase 2: L-BFGS fine-tuning (full-batch) ────────────────────────────
    print("\n" + "=" * 72)
    print("Phase 2: L-BFGS fine-tuning (full-batch)")
    print("=" * 72)

    adam_ckpt = torch.load(os.path.join(FNN_FOLDER, "fnn.pt"), map_location=DEVICE)
    model.load_state_dict(adam_ckpt["state_dict"])
    adam_best_val = adam_ckpt["val_total"]
    print(f"[lbfgs] loaded Adam best  epoch={adam_ckpt['epoch']}  "
          f"val={adam_best_val:.6f}")

    # eval() disables dropout; requires_grad stays True
    model.eval()

    # Move full training set to GPU
    p_all  = primary[train_idx].to(DEVICE)
    xy_all = xy[train_idx].to(DEVICE)
    E_all_train = E_all[train_idx].to(DEVICE)
    T_all_train = T_all[train_idx].to(DEVICE)
    print(f"[lbfgs] full train batch on {DEVICE}: {p_all.shape[0]} samples")

    lbfgs_optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=LBFGS_LR,
        max_iter=LBFGS_MAX_ITER,
        history_size=LBFGS_HISTORY_SIZE,
        line_search_fn="strong_wolfe",
    )

    lbfgs_iter_log = []   # one entry per closure call
    t_lbfgs = time.time()

    def closure():
        lbfgs_optimizer.zero_grad()
        pred = model(p_all, xy_all)
        loss, mE, mT = mse_normalized(
            pred, E_all_train, T_all_train, model.out_mean, model.out_std,
        )
        loss.backward()
        # Validation (no_grad — does not affect L-BFGS gradients)
        with torch.no_grad():
            va_tot, va_E, va_T, n_va = 0.0, 0.0, 0.0, 0
            for p_b, xy_b, E_b, T_b in val_loader:
                p_b  = p_b.to(DEVICE, non_blocking=True)
                xy_b = xy_b.to(DEVICE, non_blocking=True)
                E_b  = E_b.to(DEVICE, non_blocking=True)
                T_b  = T_b.to(DEVICE, non_blocking=True)
                v_pred = model(p_b, xy_b)
                v_loss, v_mE, v_mT = mse_normalized(
                    v_pred, E_b, T_b, model.out_mean, model.out_std,
                )
                B = p_b.shape[0]
                va_tot += v_loss.item() * B
                va_E   += v_mE.item()   * B
                va_T   += v_mT.item()   * B
                n_va   += B
            va_tot /= max(n_va, 1)
            va_E   /= max(n_va, 1)
            va_T   /= max(n_va, 1)
        it = len(lbfgs_iter_log)
        lbfgs_iter_log.append(dict(
            iter=it, loss=loss.item(), mse_E=mE.item(), mse_T=mT.item(),
            val=va_tot, val_E=va_E, val_T=va_T,
        ))
        print(f"  [lbfgs iter {it:3d}] loss={loss.item():.6f} "
              f"(E={mE.item():.6f} T={mT.item():.6f})  "
              f"val={va_tot:.6f} (E={va_E:.6f} T={va_T:.6f})")
        return loss

    final_loss = lbfgs_optimizer.step(closure)

    # Free GPU memory
    del p_all, xy_all, E_all_train, T_all_train

    dt_lbfgs = time.time() - t_lbfgs
    n_iters = len(lbfgs_iter_log)
    lbfgs_abort = torch.isnan(final_loss) or torch.isinf(final_loss)

    if lbfgs_abort:
        print(f"[lbfgs] ABORT — NaN/Inf after {n_iters} iters")
        model.load_state_dict(adam_ckpt["state_dict"])

    print(f"[lbfgs] {n_iters} iterations in {dt_lbfgs:.1f}s")

    # Validation after L-BFGS
    overall_best_val = adam_best_val
    lbfgs_log = []

    if not lbfgs_abort:
        va_tot, va_E, va_T, n_va = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for p_b, xy_b, E_b, T_b in val_loader:
                p_b  = p_b.to(DEVICE, non_blocking=True)
                xy_b = xy_b.to(DEVICE, non_blocking=True)
                E_b  = E_b.to(DEVICE, non_blocking=True)
                T_b  = T_b.to(DEVICE, non_blocking=True)
                pred = model(p_b, xy_b)
                loss, mE, mT = mse_normalized(
                    pred, E_b, T_b, model.out_mean, model.out_std,
                )
                B = p_b.shape[0]
                va_tot += loss.item() * B
                va_E   += mE.item()   * B
                va_T   += mT.item()   * B
                n_va   += B
        va_tot /= max(n_va, 1)
        va_E   /= max(n_va, 1)
        va_T   /= max(n_va, 1)

        print(f"[lbfgs val] val={va_tot:.6f} (E={va_E:.6f} T={va_T:.6f})")

        lbfgs_log.append(dict(
            epoch=N_EPOCHS + 1, phase="lbfgs",
            train=lbfgs_iter_log[-1]["loss"],
            train_E=lbfgs_iter_log[-1]["mse_E"],
            train_T=lbfgs_iter_log[-1]["mse_T"],
            val=va_tot, val_E=va_E, val_T=va_T, dt=dt_lbfgs,
        ))

        if va_tot > 2.0 * adam_best_val:
            print(f"[lbfgs] ABORT — val {va_tot:.4f} > 2x Adam best {adam_best_val:.4f}")
            model.load_state_dict(adam_ckpt["state_dict"])
        elif va_tot < overall_best_val - 1e-5:
            overall_best_val = va_tot
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": N_EPOCHS + 1,
                "phase": "lbfgs",
                "val_total": va_tot,
                "val_E": va_E,
                "val_T": va_T,
                "norm_stats": norm_stats,
                "config": dict(
                    n_det=N_DETECTORS, primary_dim=PRIMARY_DIM,
                    hidden=512, dropout=0.1,
                ),
            }, os.path.join(FNN_FOLDER, "fnn.pt"))
            print(f"[lbfgs] new best val={va_tot:.6f}  (Adam was {adam_best_val:.6f})")
        else:
            print(f"[lbfgs] no improvement  val={va_tot:.6f} vs Adam {adam_best_val:.6f}")

    # Merge logs and re-save
    full_log = log + lbfgs_log
    with open(os.path.join(FNN_FOLDER, "fnn_train_log.json"), "w") as f:
        json.dump({
            "log": full_log,
            "lbfgs_iter_log": lbfgs_iter_log,
            "best_val_total": overall_best_val,
            "best_epoch": best_epoch if overall_best_val == adam_best_val else N_EPOCHS + 1,
            "adam_best_val": adam_best_val,
            "adam_best_epoch": best_epoch,
            "config": dict(
                batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, lr=LR, lr_min=LR_MIN,
                grad_clip=GRAD_CLIP, val_frac=VAL_FRAC, seed=SEED,
                lbfgs_lr=LBFGS_LR, lbfgs_max_iter=LBFGS_MAX_ITER,
                lbfgs_history_size=LBFGS_HISTORY_SIZE,
            ),
        }, f, indent=2)
    _plot_curves(full_log, os.path.join(FNN_FOLDER, "fnn_train_curves.png"),
                 adam_epochs=N_EPOCHS, lbfgs_iter_log=lbfgs_iter_log)
    print(f"[done] best val {overall_best_val:.4f}  -> {FNN_FOLDER}")


if __name__ == "__main__":
    main()
