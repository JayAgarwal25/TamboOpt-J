"""LR-range test for the FNN surrogate (de-risk the OneCycle peak LR).

Fast-ai-style LR finder: exponentially ramp the learning rate from LR_START to
LR_END over NUM_STEPS minibatches while recording an EMA-smoothed training loss,
then read off the steepest-descent LR. Use ~1/1 of that (or min-loss-LR / 10) as
`LR_MAX` in `02_train_fnn.py`'s OneCycleLR.

This is a READ-ONLY probe: it trains a throwaway model in-memory and writes only
two diagnostics into FNN_FOLDER (`lr_range_test.png`, `lr_range_test.json`). It
does NOT touch `fnn.pt`.

It reuses `02_train_fnn.py`'s exact corpus loading + log-T transform + loss so the
measured optimum transfers directly to the real trainer.

Run:

    cd TambOpt/detector_optimization_v6
    python lr_range_test.py
"""
import importlib.util
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
from torch.utils.data import TensorDataset, DataLoader, Subset

import modules_v6  # noqa: F401  (side-effect: injects v3+v4 onto sys.path)
from modules_v6.fnn_surrogate import FNNSurrogate
from modules_v6.constants import (
    N_DETECTORS, PRIMARY_DIM,
    TRAINING_DATASET_FOLDER, FNN_FOLDER, TRAIN_FRACTION,
)

# Reuse the trainer's helpers verbatim (filename starts with a digit → importlib).
_spec = importlib.util.spec_from_file_location(
    "_t2", os.path.join(_HERE, "02_train_fnn.py"))
_t2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_t2)
shower_level_split      = _t2.shower_level_split
permute_detectors_batch = _t2.permute_detectors_batch
mse_normalized          = _t2.mse_normalized


# ── Config ───────────────────────────────────────────────────────────────────
BATCH_SIZE   = 256        # match the trainer; the recommended LR is batch-specific
HIDDEN       = 1024       # match the trainer's model width
DROPOUT      = 0.0        # OFF for a clean signal (also previews the dropout-removal fix)
WEIGHT_DECAY = 1e-5       # match the AdamW the trainer will use

LR_START     = 1e-6
LR_END       = 1e-1
NUM_STEPS    = 2000       # exponential-ramp resolution
EMA_BETA     = 0.98       # loss smoothing
DIVERGE_MULT = 4.0        # stop once smoothed loss > DIVERGE_MULT × best
VAL_FRAC     = 0.10
SEED         = 0
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_PNG  = os.path.join(FNN_FOLDER, "lr_range_test.png")
OUT_JSON = os.path.join(FNN_FOLDER, "lr_range_test.json")


def main():
    print("=" * 72)
    print("v6/lr_range_test.py — LR finder for 02_train_fnn.py")
    print("=" * 72)
    os.makedirs(FNN_FOLDER, exist_ok=True)
    print(f"device   : {DEVICE}")
    print(f"batch    : {BATCH_SIZE}   hidden: {HIDDEN}   dropout: {DROPOUT}   wd: {WEIGHT_DECAY}")
    print(f"ramp     : {LR_START:.1e} -> {LR_END:.1e} over {NUM_STEPS} steps (exponential)")

    # ── Corpus (identical to 02_train_fnn.py, including the log-T transform) ──
    primary    = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()
    xy         = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "xy.pt")).float()
    E_all      = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "E.pt")).float()
    T_all      = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "T.pt")).float()
    strat_ids  = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "strategy_ids.pt")).long()
    norm_stats = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "norm_stats.pt"))
    print(f"[load] primary={tuple(primary.shape)}  xy={tuple(xy.shape)}")

    # log-T (mirrors 02_train_fnn.py) so the loss scale matches the real trainer.
    T_LOG_SCALE = 1.0e8
    T_all = torch.log1p(T_all * T_LOG_SCALE)
    _n = T_all.shape[1]
    norm_stats["out_mean"][_n:] = float(T_all.mean().item())
    norm_stats["out_std"][_n:]  = max(float(T_all.std().item()), 1e-6)

    train_idx, _val_idx = shower_level_split(strat_ids, VAL_FRAC, SEED)
    if 0.0 < TRAIN_FRACTION < 1.0:
        g = torch.Generator().manual_seed(SEED)
        keep = max(1, int(round(TRAIN_FRACTION * train_idx.shape[0])))
        train_idx = train_idx[torch.randperm(train_idx.shape[0], generator=g)[:keep]]

    full_ds  = TensorDataset(primary, xy, E_all, T_all)
    train_ds = Subset(full_ds, train_idx.tolist())
    pin = (DEVICE.type == "cuda")
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=pin)

    # ── Fresh throwaway model + AdamW (matches the planned trainer change) ────
    torch.manual_seed(SEED)
    model = FNNSurrogate(n_det=N_DETECTORS, primary_dim=PRIMARY_DIM,
                         hidden=HIDDEN, dropout=DROPOUT).to(DEVICE)
    model.set_normalization(norm_stats)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR_START, weight_decay=WEIGHT_DECAY)

    mult = (LR_END / LR_START) ** (1.0 / max(NUM_STEPS - 1, 1))
    lr = LR_START

    lrs, raw, smooth = [], [], []
    avg = 0.0
    best = float("inf")
    step = 0

    print("[run] ramping…")
    done = False
    while not done:
        for p_b, xy_b, E_b, T_b in loader:
            p_b  = p_b.to(DEVICE, non_blocking=True)
            xy_b = xy_b.to(DEVICE, non_blocking=True)
            E_b  = E_b.to(DEVICE, non_blocking=True)
            T_b  = T_b.to(DEVICE, non_blocking=True)
            xy_b, E_b, T_b = permute_detectors_batch(xy_b, E_b, T_b)

            for grp in opt.param_groups:
                grp["lr"] = lr
            pred = model(p_b, xy_b)
            loss, _, _ = mse_normalized(pred, E_b, T_b, model.out_mean, model.out_std)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            opt.step()

            lv = float(loss.item())
            avg = EMA_BETA * avg + (1.0 - EMA_BETA) * lv
            sm  = avg / (1.0 - EMA_BETA ** (step + 1))   # bias-corrected EMA
            lrs.append(lr); raw.append(lv); smooth.append(sm)
            best = min(best, sm)

            if step % 100 == 0:
                print(f"  step {step:4d}  lr={lr:.2e}  loss={lv:.4f}  smooth={sm:.4f}")

            step += 1
            lr *= mult
            if (not math.isfinite(lv)) or (step > 10 and sm > DIVERGE_MULT * best) \
                    or step >= NUM_STEPS:
                done = True
                break

    # ── Analyse: steepest descent (min gradient of smoothed loss vs log-lr) ──
    import numpy as np
    lrs_a = np.array(lrs); sm_a = np.array(smooth)
    log_lr = np.log10(lrs_a)
    # only the descending region up to the min-loss point matters
    i_min = int(np.argmin(sm_a))
    grad = np.gradient(sm_a, log_lr)
    # steepest (most negative) slope before the min-loss point
    region = grad[: max(i_min, 1)]
    i_steep = int(np.argmin(region)) if region.size else i_min
    lr_steepest = float(lrs_a[i_steep])
    lr_min_loss = float(lrs_a[i_min])
    # Recommendation: steepest-descent LR, never above min_loss/2 (stay out of the
    # divergence elbow). fast.ai uses the min-gradient point directly.
    rec = min(lr_steepest, lr_min_loss / 2.0)

    print("-" * 72)
    print(f"[result] min-loss LR        : {lr_min_loss:.2e}  (loss {sm_a[i_min]:.4f})")
    print(f"[result] steepest-descent LR: {lr_steepest:.2e}")
    print(f"[RECOMMEND] set LR_MAX ≈ {rec:.2e} in 02_train_fnn.py  "
          f"(and LR = LR_MAX/25, LR_MIN = LR_MAX/100)")
    print("-" * 72)

    with open(OUT_JSON, "w") as f:
        json.dump({
            "recommended_lr_max": rec,
            "lr_steepest_descent": lr_steepest,
            "lr_min_loss": lr_min_loss,
            "min_smoothed_loss": float(sm_a[i_min]),
            "config": dict(batch_size=BATCH_SIZE, hidden=HIDDEN, dropout=DROPOUT,
                           weight_decay=WEIGHT_DECAY, lr_start=LR_START, lr_end=LR_END,
                           num_steps=NUM_STEPS, steps_run=step),
            "curve": {"lr": lrs, "loss_smoothed": smooth, "loss_raw": raw},
        }, f, indent=2)
    print(f"[save] {OUT_JSON}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(lrs_a, sm_a, color="C0", label="smoothed loss")
        ax.plot(lrs_a, raw, color="C0", alpha=0.2, linewidth=0.6, label="raw loss")
        ax.axvline(lr_steepest, color="C2", linestyle="--",
                   label=f"steepest {lr_steepest:.1e}")
        ax.axvline(lr_min_loss, color="C3", linestyle="--",
                   label=f"min-loss {lr_min_loss:.1e}")
        ax.axvline(rec, color="black", linestyle="-", alpha=0.7,
                   label=f"RECOMMEND LR_MAX {rec:.1e}")
        ax.set_xscale("log")
        ax.set_xlabel("learning rate"); ax.set_ylabel("training loss (z-MSE)")
        ax.set_title(f"LR range test (batch={BATCH_SIZE}, hidden={HIDDEN}, dropout={DROPOUT})")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(OUT_PNG, dpi=110); plt.close(fig)
        print(f"[save] {OUT_PNG}")
    except Exception as exc:
        print(f"[plot] skipped ({exc!r})")


if __name__ == "__main__":
    main()
