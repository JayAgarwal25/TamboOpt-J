"""Train TWO parallel per-species DeepSets surrogates (electron + muon).

The paired dual-species corpus (00_generate_data_dual_species.py) holds two
COMPONENTS of each physical event — electron rows (pdg feature 0) and muon
rows (pdg feature 1) sharing the same primaries. This trainer splits the
dataset rows by that species id and trains one DeepSets surrogate per species:
each model learns its component's response f_s(primary, layout) -> (E_s, T_s).
Stages 3-4 evaluate BOTH models per event and combine the outputs physically
(modules_v6/dual_surrogate.py: counts add, times average count-weighted).

Per species, everything matches the original single-model trainer: shower-level
split, log-T target treatment, z-scored MSE loss, two-phase Adam(OneCycle) →
chunked-L-BFGS recipe with per-iter best-val save. Norm stats are computed on
each species SUBSET (electron and muon count scales differ by ~an order of
magnitude, so shared stats would mis-weight the smaller component's loss); the
per-model stats ship inside each checkpoint.

Architecture note (THEORY.md §10): DeepSets — a shared per-detector
encoder/decoder with a pooled context, permutation-equivariant BY CONSTRUCTION,
so no permutation augmentation is used.

Checkpoints land DIRECTLY in FNN_FOLDER as `fnn_electron.pt` / `fnn_muon.pt`
(species-tagged names cannot clobber a legacy single-model fnn.pt); stages 3-4
load them via `modules_v6.dual_surrogate.load_dual_surrogate(FNN_FOLDER, ...)`.

Run:

    cd TambOpt/detector_optimization_v6
    python 02_train_fnn_deepsets.py                         # both species
    python 02_train_fnn_deepsets.py --species muon          # retrain one
    python 02_train_fnn_deepsets.py --epochs 2 --lbfgs-iters 3   # smoke run
"""
import argparse
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
from modules_v6.deepsets_surrogate import DeepSetsSurrogate
from modules_v6.fnn_surrogate import compute_normalization
from modules_v6.constants import (
    N_DETECTORS, PRIMARY_DIM, T_LOG_SCALE,
    TRAINING_DATASET_FOLDER, FNN_FOLDER, TRAIN_FRACTION,
)

# ── Config ───────────────────────────────────────────────────────────────────
# Species-tagged checkpoints (fnn_electron.pt / fnn_muon.pt) go straight into
# FNN_FOLDER — they cannot clobber a legacy single-model fnn.pt, and stages 3-4
# load the pair from there via dual_surrogate.load_dual_surrogate.
OUTPUT_FOLDER = FNN_FOLDER
SPECIES_TAGS  = (("electron", 0), ("muon", 1))   # (tag, species id: 0=electron, 1=muon)

BATCH_SIZE          = 256
N_EPOCHS            = 100
LR                  = 1e-5     # OneCycle initial
LR_MAX              = 3e-4     # OneCycle peak (DeepSets tolerates a higher peak than the flat MLP)
LR_MIN              = 1e-6     # OneCycle floor (kept off the dead ~1e-8 zone)
ONECYCLE_PCT_START  = 0.10
GRAD_CLIP           = 10.0
VAL_FRAC            = 0.10
SEED                = 0
NUM_WORKERS         = 0
DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DeepSets shape (parameter-light vs the 6.7M flat MLP).
DS_HIDDEN   = 256
DS_CONTEXT  = 64
DS_N_ENC    = 3
DS_N_DEC    = 3
DS_DROPOUT  = 0.0

# ── L-BFGS fine-tuning (full-batch, chunked closure) ─────────────
LBFGS_LR            = 1.0
LBFGS_MAX_ITER      = 1000
LBFGS_HISTORY_SIZE  = 10
LBFGS_CHUNK_SIZE    = 8192   # DeepSets is light → larger chunks fit
# Early stop: abort L-BFGS after this many closure calls with no val improvement.
# DeepSets is well-fit by Adam, so L-BFGS tends to overfit train from iter 0;
# this stops it burning the full budget while best-save still keeps the optimum.
LBFGS_PATIENCE      = 150


def shower_level_split(strategy_ids: torch.Tensor,
                       val_frac: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shower-level split. All strategy entries of a shower share a split.
    Pairs are strategy-major: pair k's shower index is k - strat[k]*n_showers."""
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
    val_mask = is_val[shower_of_pair]
    return (torch.nonzero(~val_mask).squeeze(-1),
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
    

def mse_normalized(pred, E_tgt, T_tgt, out_mean, out_std):
    """MSE in the z-score space the model normalizes to. Returns (total, E, T)."""
    pred_flat   = torch.cat([pred[..., 0], pred[..., 1]], dim=1)   # (B, 200)
    target_flat = torch.cat([E_tgt, T_tgt], dim=1)
    pred_n   = (pred_flat   - out_mean) / out_std
    target_n = (target_flat - out_mean) / out_std
    n = E_tgt.shape[1]
    mse_E = F.mse_loss(pred_n[:, :n], target_n[:, :n])
    mse_T = F.mse_loss(pred_n[:, n:], target_n[:, n:])
    return 0.5 * (mse_E + mse_T), mse_E, mse_T


def _plot_curves(log, path, adam_epochs=0, lbfgs_iter_log=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        adam_log = [e for e in log if e.get("phase") != "lbfgs"]
        ep = [e["epoch"] for e in adam_log]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(ep, [e["train"] for e in adam_log], color="C0", label="train")
        axes[0].plot(ep, [e["val"]   for e in adam_log], color="C1", label="val")
        axes[1].plot(ep, [e["train_E"] for e in adam_log], color="C0")
        axes[1].plot(ep, [e["val_E"]   for e in adam_log], color="C1")
        axes[1].plot(ep, [e["train_T"] for e in adam_log], color="C0", linestyle="--")
        axes[1].plot(ep, [e["val_T"]   for e in adam_log], color="C1", linestyle="--")

        if lbfgs_iter_log:
            lb_ep = [adam_epochs + 1 + e["iter"] for e in lbfgs_iter_log]
            axes[0].plot(lb_ep, [e["loss"]  for e in lbfgs_iter_log], color="C0")
            axes[0].plot(lb_ep, [e["val"]   for e in lbfgs_iter_log], color="C1")
            axes[1].plot(lb_ep, [e["mse_E"] for e in lbfgs_iter_log], color="C0")
            axes[1].plot(lb_ep, [e["val_E"] for e in lbfgs_iter_log], color="C1")
            axes[1].plot(lb_ep, [e["mse_T"] for e in lbfgs_iter_log], color="C0", linestyle="--")
            axes[1].plot(lb_ep, [e["val_T"] for e in lbfgs_iter_log], color="C1", linestyle="--")

        if adam_epochs > 0:
            for ax in axes:
                ax.axvline(adam_epochs, color="gray", linestyle="--", alpha=0.5,
                           label="Adam\u2192L-BFGS")

        axes[0].set_xlabel("epoch / iter"); axes[0].set_ylabel("MSE (z-scored)")
        axes[0].set_title("total");  axes[0].grid(alpha=0.3); axes[0].legend(fontsize=9)
        axes[1].set_xlabel("epoch / iter"); axes[1].set_ylabel("MSE (z-scored)")
        axes[1].set_title("per-channel"); axes[1].grid(alpha=0.3)
        # Proxy legend: 2 color entries (train/val) + 2 style entries (E/T).
        axes[1].legend(handles=[
            Line2D([], [], color="C0",                 label="train"),
            Line2D([], [], color="C1",                 label="val"),
            Line2D([], [], color="black",              label="E"),
            Line2D([], [], color="black", linestyle="--", label="T"),
        ], fontsize=9, loc="best")
        axes[0].set_yscale("log"); axes[1].set_yscale("log")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] skipped ({exc!r})")


def _ckpt_config(tag: str):
    return dict(
        model_type="deepsets", species=tag,
        n_det=N_DETECTORS, primary_dim=PRIMARY_DIM,
        hidden=DS_HIDDEN, context=DS_CONTEXT, n_enc=DS_N_ENC, n_dec=DS_N_DEC,
        dropout=DS_DROPOUT,
    )


def train_species(tag:        str,
                  primary:    torch.Tensor,
                  xy:         torch.Tensor,
                  E_all:      torch.Tensor,
                  T_all:      torch.Tensor,
                  strat_ids:  torch.Tensor,
                  n_epochs:   int,
                  lbfgs_max_iter: int) -> None:
    """Train one per-species DeepSets surrogate on its (already filtered) rows.

    Norm stats are computed on the SUBSET — per-species E/T scales differ —
    and the log-T transform mutates the T slots exactly as the single-model
    trainer did; the per-model stats ship inside the checkpoint.
    """
    ckpt_name   = f"fnn_{tag}.pt"
    log_name    = f"fnn_{tag}_train_log.json"
    curves_name = f"fnn_{tag}_train_curves.png"
    tvp_name    = f"fnn_{tag}_target_vs_pred.png"

    print("=" * 72)
    print(f"[{tag}] DeepSets surrogate on {primary.shape[0]} rows")
    print("=" * 72)

    # Per-species z-score stats (NOT the corpus-wide norm_stats.pt).
    norm_stats = compute_normalization(primary, xy, E_all, T_all)

    # log-T canonical target (mirrors 02_train_fnn.py); ship modified stats in ckpt.
    T_all = torch.log1p(T_all * T_LOG_SCALE)
    _n = T_all.shape[1]
    norm_stats["out_mean"][_n:] = float(T_all.mean().item())
    norm_stats["out_std"][_n:]  = max(float(T_all.std().item()), 1e-6)
    print(f"[log1p-T] applied log1p(T*{T_LOG_SCALE:.0e}); "
          f"T mean={norm_stats['out_mean'][_n]:.4f} std={norm_stats['out_std'][_n]:.4f}")

    train_idx, val_idx = shower_level_split(strat_ids, VAL_FRAC, SEED)
    print(f"[split] train pairs={len(train_idx)}  val pairs={len(val_idx)}")
    if 0.0 < TRAIN_FRACTION < 1.0:
        _n_orig = int(train_idx.shape[0])
        _n_keep = max(1, int(round(TRAIN_FRACTION * _n_orig)))
        _g_sub = torch.Generator().manual_seed(SEED)
        _perm = torch.randperm(_n_orig, generator=_g_sub)
        train_idx = train_idx[_perm[:_n_keep]]
        print(f"[subsample] kept {_n_keep} of {_n_orig} train pairs (TRAIN_FRACTION={TRAIN_FRACTION})")

    full_ds  = TensorDataset(primary, xy, E_all, T_all)
    train_ds = Subset(full_ds, train_idx.tolist())
    val_ds   = Subset(full_ds, val_idx.tolist())
    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    # ── Model ────────────────────────────────────────────────────────────
    torch.manual_seed(SEED)
    model = DeepSetsSurrogate(
        n_det=N_DETECTORS, primary_dim=PRIMARY_DIM,
        hidden=DS_HIDDEN, context=DS_CONTEXT,
        n_enc=DS_N_ENC, n_dec=DS_N_DEC, dropout=DS_DROPOUT,
    ).to(DEVICE)
    model.set_normalization(norm_stats)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] DeepSets params={n_params:,}  (flat MLP was ~6.7M)")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    steps_per_epoch = len(train_loader)
    total_steps = n_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LR_MAX, 
        total_steps=total_steps,
        pct_start=ONECYCLE_PCT_START,
        anneal_strategy="cos",
        div_factor=LR_MAX / LR, # OneCycle: min_lr = initial_lr / final_div_factor (relative to INITIAL lr).
        final_div_factor=LR / LR_MIN,
    )

    log = []
    best_val   = float("inf")
    best_epoch = -1

    # ── Phase 1: Adam (no permutation augmentation) ──────────────────────
    for epoch in range(n_epochs):
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
            scheduler.step()   # OneCycleLR steps per-batch

            B = p_b.shape[0]
            tr_tot += loss.item() * B
            tr_E   += mE.item()   * B
            tr_T   += mT.item()   * B
            n_tr   += B
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
        print(f"[{tag} epoch {epoch+1:3d}/{n_epochs}] "
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
                "config": _ckpt_config(tag)
                }, os.path.join(OUTPUT_FOLDER, ckpt_name))

    with open(os.path.join(OUTPUT_FOLDER, log_name), "w") as f:
        json.dump({
            "log": log,
            "best_val_total": best_val,
            "best_epoch": best_epoch,
            "config": dict(
                batch_size=BATCH_SIZE, n_epochs=n_epochs, lr=LR,lr_max=LR_MAX, lr_min=LR_MIN,
                val_frac=VAL_FRAC, seed=SEED, **_ckpt_config(tag)
                ),
                }, f, indent=2)
    _plot_curves(log, os.path.join(OUTPUT_FOLDER, curves_name))
    print(f"[{tag} adam done] best val {best_val:.4f} at epoch {best_epoch}")

    # ── Phase 2: L-BFGS fine-tuning (full-batch) ────────────────────────────
    print("\n" + "=" * 72)
    print(f"[{tag}] Phase 2: L-BFGS fine-tuning (full-batch)")
    print("=" * 72)
    adam_ckpt = torch.load(os.path.join(OUTPUT_FOLDER, ckpt_name), map_location=DEVICE)
    model.load_state_dict(adam_ckpt["state_dict"])
    adam_best_val = adam_ckpt["val_total"]
    print(f"[lbfgs] loaded Adam best epoch={adam_ckpt['epoch']} val={adam_best_val:.6f}")
    model.eval()

    p_all  = primary[train_idx].to(DEVICE)
    xy_all = xy[train_idx].to(DEVICE)
    E_all_train = E_all[train_idx].to(DEVICE)
    T_all_train = T_all[train_idx].to(DEVICE)
    print(f"[lbfgs] full train batch on {DEVICE}: {p_all.shape[0]} samples")

    lbfgs_optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=LBFGS_LR,
        max_iter=lbfgs_max_iter,
        history_size=LBFGS_HISTORY_SIZE,
        line_search_fn="strong_wolfe",
    )

    lbfgs_iter_log = []   # one entry per closure call
    t_lbfgs = time.time()
    # 0030 (matches new baseline): track best L-BFGS val and save fnn.pt
    # whenever it improves. Previously only the last iter was checked, so
    # we threw away the L-BFGS minimum (siblings 0010/0023/0024 logged
    # min < last by 0.003-0.006 absolute).
    lbfgs_best_val = adam_best_val
    lbfgs_best_iter = -1
    lbfgs_no_improve = 0
    n_total = int(p_all.shape[0])

    class _LBFGSStop(Exception):
        """Raised from the closure to early-stop L-BFGS once val stalls."""

    def closure():
        # 0030: chunked forward+backward to avoid full-batch OOM on wider
        # Deep Sets. zero grads once, then for each chunk run forward,
        # weight loss by chunk_size/n_total, backward (accumulates grad),
        # detach the loss scalar. Final scalar = mean over all samples,
        # final grad = exact full-batch gradient (verified in 0010).
        nonlocal lbfgs_best_val, lbfgs_best_iter, lbfgs_no_improve
        lbfgs_optimizer.zero_grad()
        sum_loss = 0.0
        sum_E    = 0.0
        sum_T    = 0.0
        for start in range(0, n_total, LBFGS_CHUNK_SIZE):
            end = min(start + LBFGS_CHUNK_SIZE, n_total)
            chunk_size = end - start
            p_c  = p_all[start:end]
            xy_c = xy_all[start:end]
            E_c  = E_all_train[start:end]
            T_c  = T_all_train[start:end]
            pred_c = model(p_c, xy_c)
            chunk_loss, chunk_mE, chunk_mT = mse_normalized(
                pred_c, E_c, T_c, model.out_mean, model.out_std,
            )
            weight = chunk_size / n_total
            (chunk_loss * weight).backward()
            sum_loss += chunk_loss.detach() * chunk_size
            sum_E    += chunk_mE.detach()   * chunk_size
            sum_T    += chunk_mT.detach()   * chunk_size
        loss = sum_loss / n_total
        mE   = sum_E    / n_total
        mT   = sum_T    / n_total
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
        if va_tot < lbfgs_best_val - 1e-5:
            lbfgs_best_val = va_tot
            lbfgs_best_iter = it
            lbfgs_no_improve = 0
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": n_epochs + 1,
                "phase": "lbfgs",
                "lbfgs_iter": it,
                "val_total": va_tot,
                "val_E": va_E,
                "val_T": va_T,
                "norm_stats": norm_stats,
                "config": _ckpt_config(tag)
            }, os.path.join(OUTPUT_FOLDER, ckpt_name))
            marker = "  <- NEW BEST (saved)"
        else:
            lbfgs_no_improve += 1
            marker = ""
        print(f"  [lbfgs iter {it:3d}] loss={loss.item():.6f} "
              f"(E={mE.item():.6f} T={mT.item():.6f})  "
              f"val={va_tot:.6f} (E={va_E:.6f} T={va_T:.6f}){marker}")
        if lbfgs_no_improve >= LBFGS_PATIENCE:
            raise _LBFGSStop
        return loss

    try:
        final_loss = lbfgs_optimizer.step(closure)
    except _LBFGSStop:
        print(f"[lbfgs] early stop: {LBFGS_PATIENCE} closure calls with no val "
              f"improvement (best val={lbfgs_best_val:.6f} at iter {lbfgs_best_iter})")
        final_loss = torch.tensor(lbfgs_best_val, device=DEVICE)

    # Free GPU memory
    del p_all, xy_all, E_all_train, T_all_train

    dt_lbfgs = time.time() - t_lbfgs
    n_iters = len(lbfgs_iter_log)
    lbfgs_abort = torch.isnan(final_loss) or torch.isinf(final_loss)

    if lbfgs_abort:
        print(f"[lbfgs] ABORT — NaN/Inf after {n_iters} iters")
        model.load_state_dict(adam_ckpt["state_dict"])

    print(f"[lbfgs] {n_iters} iterations in {dt_lbfgs:.1f}s")

    # Validation after L-BFGS: per-iter best save already happened inside
    # the closure. fnn.pt holds the best-ever weights.
    overall_best_val = lbfgs_best_val
    lbfgs_log = []

    if not lbfgs_abort and lbfgs_iter_log:
        last = lbfgs_iter_log[-1]
        lbfgs_log.append(dict(
            epoch=n_epochs + 1, phase="lbfgs",
            train=last["loss"], train_E=last["mse_E"], train_T=last["mse_T"],
            val=last["val"], val_E=last["val_E"], val_T=last["val_T"], dt=dt_lbfgs,
        ))

    if lbfgs_best_iter >= 0:
        print(f"[lbfgs] best val={lbfgs_best_val:.6f} at iter {lbfgs_best_iter} "
              f"(Adam was {adam_best_val:.6f}, gain={adam_best_val-lbfgs_best_val:.6f})")
    else:
        print(f"[lbfgs] no improvement over Adam best {adam_best_val:.6f}  "
              f"(fnn.pt unchanged)")

    # Merge logs and re-save
    full_log = log + lbfgs_log
    with open(os.path.join(OUTPUT_FOLDER, log_name), "w") as f:
        json.dump({
            "log": full_log,
            "lbfgs_iter_log": lbfgs_iter_log,
            "best_val_total": overall_best_val,
            "best_epoch": best_epoch if overall_best_val == adam_best_val else n_epochs + 1,
            "adam_best_val": adam_best_val,
            "adam_best_epoch": best_epoch,
            "config": dict(
                batch_size=BATCH_SIZE, n_epochs=n_epochs, lr=LR, lr_min=LR_MIN,
                grad_clip=GRAD_CLIP, val_frac=VAL_FRAC, seed=SEED,
                lbfgs_lr=LBFGS_LR, lbfgs_max_iter=lbfgs_max_iter,
                lbfgs_history_size=LBFGS_HISTORY_SIZE,
                **_ckpt_config(tag),
            ),
        }, f, indent=2)
    _plot_curves(full_log, os.path.join(OUTPUT_FOLDER, curves_name),
                 adam_epochs=n_epochs, lbfgs_iter_log=lbfgs_iter_log)
    print(f"[{tag} done] best val {overall_best_val:.4f}  -> "
          f"{os.path.join(OUTPUT_FOLDER, ckpt_name)}")

    # ── Auto-render target-vs-pred from the best checkpoint ──────────────
    try:
        best = torch.load(os.path.join(OUTPUT_FOLDER, ckpt_name), map_location=DEVICE)
        model.load_state_dict(best["state_dict"]); model.eval()
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "_plot_tvp",
            os.path.join(_HERE, "plots", "02_plot_nn_target_vs_pred.py"),
            )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _mod.plot_fnn_only(
            fnn=model,
            primary=primary, xy=xy,
            E_true=E_all, T_true=T_all,
            val_idx=val_idx,
            output_path=os.path.join(OUTPUT_FOLDER, tvp_name))
    except Exception as exc:
        print(f"[plot-tvp] skipped ({exc!r})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=N_EPOCHS,
                    help="Adam epochs per species (default from config)")
    ap.add_argument("--lbfgs-iters", type=int, default=LBFGS_MAX_ITER,
                    help="L-BFGS max iterations per species (default from config)")
    ap.add_argument("--species", type=str, default="electron,muon",
                    help="comma-separated subset of {electron,muon} to (re)train")
    args = ap.parse_args()
    wanted = {s.strip() for s in args.species.split(",") if s.strip()}
    unknown = wanted - {tag for tag, _ in SPECIES_TAGS}
    if unknown:
        raise SystemExit(f"unknown species {sorted(unknown)}; valid: electron, muon")

    print("=" * 72)
    print("v6/02_train_fnn_deepsets.py — two parallel per-species DeepSets surrogates")
    print("=" * 72)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"data input dir  : {TRAINING_DATASET_FOLDER}")
    print(f"output dir      : {OUTPUT_FOLDER}")
    print(f"device          : {DEVICE}")
    print(f"species         : {sorted(wanted)}")
    print(f"batch           : {BATCH_SIZE}   epochs: {args.epochs}   "
          f"lbfgs iters: {args.lbfgs_iters}")
    print(f"lr              : {LR} -> {LR_MAX} -> {LR_MIN} OneCycleLR (pct_start={ONECYCLE_PCT_START})")
    print(f"deepsets        : hidden={DS_HIDDEN} context={DS_CONTEXT} "
          f"enc={DS_N_ENC} dec={DS_N_DEC} dropout={DS_DROPOUT}")
    print(f"augmentation    : NONE (equivariant by construction)")

    t0 = time.time()
    primary   = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()
    xy        = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "xy.pt")).float()
    E_all     = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "E.pt")).float()
    T_all     = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "T.pt")).float()
    strat_ids = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "strategy_ids.pt")).long()
    species_ids = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "species_ids.pt")).long()
    print(f"[load] corpus in {time.time()-t0:.1f}s  primary={tuple(primary.shape)}")

    n_strat = int(strat_ids.max().item() + 1)
    for tag, species_val in SPECIES_TAGS:
        if tag not in wanted:
            continue
        # Split on the Step-1 species sidecar (0=electron, 1=muon). The primary's
        # 5th feature is now the EM/hadronic class (a real input the model learns),
        # not the species, so it must NOT be used to route rows here.
        idx = torch.nonzero(species_ids == species_val).squeeze(-1)
        if idx.numel() == 0:
            raise SystemExit(
                f"no rows with species id {species_val} ({tag}) in the dataset — "
                f"was 01_build_dataset.py run on the paired dual-species corpus "
                f"(and did it write species_ids.pt)?")
        # shower_level_split relies on strategy-major contiguous blocks; the
        # species filter preserves that as long as every strategy block holds
        # the same per-species rows (true for the paired corpus).
        assert idx.numel() % n_strat == 0, (idx.numel(), n_strat)
        train_species(
            tag,
            primary[idx], xy[idx], E_all[idx], T_all[idx], strat_ids[idx],
            n_epochs=args.epochs, lbfgs_max_iter=args.lbfgs_iters,
        )


if __name__ == "__main__":
    main()
