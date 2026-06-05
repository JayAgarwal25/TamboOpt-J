"""Train the FNN surrogate as a permutation-equivariant DeepSets model.

Path-(a) rewrite of `02_train_fnn.py` (see THEORY.md §10): the flat MLP plateaued
because it must learn a per-detector-local map across 100 flat slots and fake
permutation equivariance via augmentation. This trainer swaps in
`DeepSetsSurrogate` — a shared per-detector encoder/decoder with a pooled
context, equivariant BY CONSTRUCTION — so the permutation augmentation is
DELETED (it was approximating exactly this symmetry).

Everything else matches `02_train_fnn.py`: same corpus, same shower-level split,
same log-T target treatment, same z-scored MSE loss, same two-phase
Adam(OneCycle) → chunked-L-BFGS recipe with per-iter best-val save. The model is
a literal drop-in for the FNN contract (`forward(primary, xy)→(B,100,2)`,
`set_normalization(stats)`), so Steps 3–4 can load this checkpoint unchanged.

Artifacts go to a DEDICATED folder (`FNN_FOLDER + "_deepsets"`) so they never
clobber the production flat-MLP `fnn.pt`. To promote this model to production,
point `FNN_FOLDER` in modules_v6/constants.py at this folder (or copy the .pt).

Run:

    cd TambOpt/detector_optimization_v6
    python 02_train_fnn_deepsets.py
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

import modules_v6  # noqa: F401  (side-effect: injects v3+v4 onto sys.path)
from modules_v6.deepsets_surrogate import DeepSetsSurrogate
from modules_v6.constants import (
    N_DETECTORS, PRIMARY_DIM,
    TRAINING_DATASET_FOLDER, FNN_FOLDER, TRAIN_FRACTION,
)

# ── Config ───────────────────────────────────────────────────────────────────
# Dedicated output dir — never overwrite the production flat-MLP fnn.pt.
OUTPUT_FOLDER = FNN_FOLDER + "_deepsets"
CKPT_NAME     = "fnn.pt"   # same name → drop-in if FNN_FOLDER is repointed here
LOG_NAME      = "fnn_train_log.json"
CURVES_NAME   = "fnn_train_curves.png"
TVP_NAME      = "fnn_target_vs_pred.png"

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
LBFGS_MAX_ITER      = 800
LBFGS_HISTORY_SIZE  = 10
LBFGS_CHUNK_SIZE    = 8192   # DeepSets is light → larger chunks fit


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
            lb = [adam_epochs + 1 + e["iter"] for e in lbfgs_iter_log]
            axes[0].plot(lb, [e["loss"]  for e in lbfgs_iter_log], color="C0")
            axes[0].plot(lb, [e["val"]   for e in lbfgs_iter_log], color="C1")
            axes[1].plot(lb, [e["mse_E"] for e in lbfgs_iter_log], color="C0")
            axes[1].plot(lb, [e["val_E"] for e in lbfgs_iter_log], color="C1")
            axes[1].plot(lb, [e["mse_T"] for e in lbfgs_iter_log], color="C0", linestyle="--")
            axes[1].plot(lb, [e["val_T"] for e in lbfgs_iter_log], color="C1", linestyle="--")
        if adam_epochs > 0:
            for ax in axes:
                ax.axvline(adam_epochs, color="gray", linestyle="--", alpha=0.5,
                           label="Adam→L-BFGS")
        axes[0].set_xlabel("epoch / iter"); axes[0].set_ylabel("MSE (z-scored)")
        axes[0].set_title("total"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=9)
        axes[1].set_xlabel("epoch / iter"); axes[1].set_ylabel("MSE (z-scored)")
        axes[1].set_title("per-channel"); axes[1].grid(alpha=0.3)
        axes[1].legend(handles=[
            Line2D([], [], color="C0", label="train"),
            Line2D([], [], color="C1", label="val"),
            Line2D([], [], color="black", label="E"),
            Line2D([], [], color="black", linestyle="--", label="T"),
        ], fontsize=9, loc="best")
        axes[0].set_yscale("log"); axes[1].set_yscale("log")
        fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] skipped ({exc!r})")


def _ckpt_config():
    return dict(
        model_type="deepsets", n_det=N_DETECTORS, primary_dim=PRIMARY_DIM,
        hidden=DS_HIDDEN, context=DS_CONTEXT, n_enc=DS_N_ENC, n_dec=DS_N_DEC,
        dropout=DS_DROPOUT,
    )


def main():
    print("=" * 72)
    print("v6/02_train_fnn_deepsets.py — permutation-equivariant DeepSets surrogate")
    print("=" * 72)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"data input dir  : {TRAINING_DATASET_FOLDER}")
    print(f"output dir      : {OUTPUT_FOLDER}")
    print(f"device          : {DEVICE}")
    print(f"batch           : {BATCH_SIZE}   epochs: {N_EPOCHS}")
    print(f"lr              : {LR} -> {LR_MAX} -> {LR_MIN} OneCycleLR (pct_start={ONECYCLE_PCT_START})")
    print(f"deepsets        : hidden={DS_HIDDEN} context={DS_CONTEXT} "
          f"enc={DS_N_ENC} dec={DS_N_DEC} dropout={DS_DROPOUT}")
    print(f"augmentation    : NONE (equivariant by construction)")

    # ── Corpus (identical to 02_train_fnn.py, incl. the log-T transform) ──
    t0 = time.time()
    primary    = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()
    xy         = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "xy.pt")).float()
    E_all      = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "E.pt")).float()
    T_all      = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "T.pt")).float()
    strat_ids  = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "strategy_ids.pt")).long()
    norm_stats = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "norm_stats.pt"))
    print(f"[load] corpus in {time.time()-t0:.1f}s  primary={tuple(primary.shape)}")

    # log-T canonical target (mirrors 02_train_fnn.py); ship modified stats in ckpt.
    T_LOG_SCALE = 1.0e8
    T_all = torch.log1p(T_all * T_LOG_SCALE)
    _n = T_all.shape[1]
    norm_stats["out_mean"][_n:] = float(T_all.mean().item())
    norm_stats["out_std"][_n:]  = max(float(T_all.std().item()), 1e-6)
    print(f"[log1p-T] applied log1p(T*{T_LOG_SCALE:.0e}); "
          f"T mean={norm_stats['out_mean'][_n]:.4f} std={norm_stats['out_std'][_n]:.4f}")

    train_idx, val_idx = shower_level_split(strat_ids, VAL_FRAC, SEED)
    print(f"[split] train pairs={len(train_idx)}  val pairs={len(val_idx)}")
    if 0.0 < TRAIN_FRACTION < 1.0:
        g = torch.Generator().manual_seed(SEED)
        keep = max(1, int(round(TRAIN_FRACTION * train_idx.shape[0])))
        train_idx = train_idx[torch.randperm(train_idx.shape[0], generator=g)[:keep]]
        print(f"[subsample] kept {keep} train pairs (TRAIN_FRACTION={TRAIN_FRACTION})")

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
    total_steps = N_EPOCHS * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR_MAX, total_steps=total_steps,
        pct_start=ONECYCLE_PCT_START, anneal_strategy="cos",
        div_factor=LR_MAX / LR,
        # OneCycle: min_lr = initial_lr / final_div_factor (relative to INITIAL lr).
        final_div_factor=LR / LR_MIN,
    )

    log = []
    best_val, best_epoch = float("inf"), -1

    # ── Phase 1: Adam (no permutation augmentation) ──────────────────────
    for epoch in range(N_EPOCHS):
        t_epoch = time.time()
        model.train()
        tr_tot = tr_E = tr_T = 0.0; n_tr = 0
        for p_b, xy_b, E_b, T_b in train_loader:
            p_b  = p_b.to(DEVICE, non_blocking=True)
            xy_b = xy_b.to(DEVICE, non_blocking=True)
            E_b  = E_b.to(DEVICE, non_blocking=True)
            T_b  = T_b.to(DEVICE, non_blocking=True)
            # NOTE: no permute_detectors_batch — DeepSets is equivariant.
            pred = model(p_b, xy_b)
            loss, mE, mT = mse_normalized(pred, E_b, T_b, model.out_mean, model.out_std)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step(); scheduler.step()
            B = p_b.shape[0]
            tr_tot += loss.item()*B; tr_E += mE.item()*B; tr_T += mT.item()*B; n_tr += B
        tr_tot /= max(n_tr,1); tr_E /= max(n_tr,1); tr_T /= max(n_tr,1)

        model.eval()
        va_tot = va_E = va_T = 0.0; n_va = 0
        with torch.no_grad():
            for p_b, xy_b, E_b, T_b in val_loader:
                p_b  = p_b.to(DEVICE, non_blocking=True)
                xy_b = xy_b.to(DEVICE, non_blocking=True)
                E_b  = E_b.to(DEVICE, non_blocking=True)
                T_b  = T_b.to(DEVICE, non_blocking=True)
                pred = model(p_b, xy_b)
                loss, mE, mT = mse_normalized(pred, E_b, T_b, model.out_mean, model.out_std)
                B = p_b.shape[0]
                va_tot += loss.item()*B; va_E += mE.item()*B; va_T += mT.item()*B; n_va += B
        va_tot /= max(n_va,1); va_E /= max(n_va,1); va_T /= max(n_va,1)

        dt = time.time() - t_epoch
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch+1:3d}/{N_EPOCHS}] "
              f"train={tr_tot:.4f} (E={tr_E:.4f} T={tr_T:.4f})  "
              f"val={va_tot:.4f} (E={va_E:.4f} T={va_T:.4f})  lr={lr_now:.1e}  {dt:.1f}s")
        log.append(dict(epoch=epoch+1, train=tr_tot, train_E=tr_E, train_T=tr_T,
                        val=va_tot, val_E=va_E, val_T=va_T, lr=lr_now, dt=dt))
        if va_tot < best_val - 1e-5:
            best_val, best_epoch = va_tot, epoch+1
            torch.save({"state_dict": model.state_dict(), "epoch": epoch+1,
                        "val_total": va_tot, "val_E": va_E, "val_T": va_T,
                        "norm_stats": norm_stats, "config": _ckpt_config()},
                       os.path.join(OUTPUT_FOLDER, CKPT_NAME))

    with open(os.path.join(OUTPUT_FOLDER, LOG_NAME), "w") as f:
        json.dump({"log": log, "best_val_total": best_val, "best_epoch": best_epoch,
                   "config": dict(batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, lr=LR,
                                  lr_max=LR_MAX, lr_min=LR_MIN, val_frac=VAL_FRAC,
                                  seed=SEED, **_ckpt_config())}, f, indent=2)
    _plot_curves(log, os.path.join(OUTPUT_FOLDER, CURVES_NAME))
    print(f"[adam done] best val {best_val:.4f} at epoch {best_epoch}")

    # ── Phase 2: L-BFGS fine-tuning (chunked, per-iter best-val save) ────
    print("\n" + "=" * 72)
    print("Phase 2: L-BFGS fine-tuning (chunked closure)")
    print("=" * 72)
    adam_ckpt = torch.load(os.path.join(OUTPUT_FOLDER, CKPT_NAME), map_location=DEVICE)
    model.load_state_dict(adam_ckpt["state_dict"])
    adam_best_val = adam_ckpt["val_total"]
    print(f"[lbfgs] loaded Adam best epoch={adam_ckpt['epoch']} val={adam_best_val:.6f}")
    model.eval()

    p_all  = primary[train_idx].to(DEVICE)
    xy_all = xy[train_idx].to(DEVICE)
    E_tr   = E_all[train_idx].to(DEVICE)
    T_tr   = T_all[train_idx].to(DEVICE)
    n_total = int(p_all.shape[0])
    print(f"[lbfgs] full train batch on {DEVICE}: {n_total} samples")

    lbfgs = torch.optim.LBFGS(model.parameters(), lr=LBFGS_LR, max_iter=LBFGS_MAX_ITER,
                              history_size=LBFGS_HISTORY_SIZE, line_search_fn="strong_wolfe")
    lbfgs_iter_log = []
    lbfgs_best_val = adam_best_val
    lbfgs_best_iter = -1
    t_lbfgs = time.time()

    def closure():
        nonlocal lbfgs_best_val, lbfgs_best_iter
        lbfgs.zero_grad()
        z = torch.zeros((), device=DEVICE)
        s_loss = z.clone(); s_E = z.clone(); s_T = z.clone()
        for start in range(0, n_total, LBFGS_CHUNK_SIZE):
            end = min(start + LBFGS_CHUNK_SIZE, n_total)
            cs = end - start
            pred_c = model(p_all[start:end], xy_all[start:end])
            cl, cE, cT = mse_normalized(pred_c, E_tr[start:end], T_tr[start:end],
                                        model.out_mean, model.out_std)
            (cl * (cs / n_total)).backward()
            s_loss += cl.detach()*cs; s_E += cE.detach()*cs; s_T += cT.detach()*cs
        loss = s_loss / n_total; mE = s_E / n_total; mT = s_T / n_total
        with torch.no_grad():
            va_tot = va_E = va_T = 0.0; n_va = 0
            for p_b, xy_b, E_b, T_b in val_loader:
                p_b=p_b.to(DEVICE); xy_b=xy_b.to(DEVICE); E_b=E_b.to(DEVICE); T_b=T_b.to(DEVICE)
                vl, vE, vT = mse_normalized(model(p_b, xy_b), E_b, T_b,
                                            model.out_mean, model.out_std)
                B = p_b.shape[0]; va_tot+=vl.item()*B; va_E+=vE.item()*B; va_T+=vT.item()*B; n_va+=B
            va_tot/=max(n_va,1); va_E/=max(n_va,1); va_T/=max(n_va,1)
        it = len(lbfgs_iter_log)
        lbfgs_iter_log.append(dict(iter=it, loss=loss.item(), mse_E=mE.item(), mse_T=mT.item(),
                                   val=va_tot, val_E=va_E, val_T=va_T))
        if va_tot < lbfgs_best_val - 1e-5:
            lbfgs_best_val, lbfgs_best_iter = va_tot, it
            torch.save({"state_dict": model.state_dict(), "epoch": N_EPOCHS+1,
                        "phase": "lbfgs", "lbfgs_iter": it, "val_total": va_tot,
                        "val_E": va_E, "val_T": va_T, "norm_stats": norm_stats,
                        "config": _ckpt_config()},
                       os.path.join(OUTPUT_FOLDER, CKPT_NAME))
            marker = "  <- NEW BEST (saved)"
        else:
            marker = ""
        print(f"  [lbfgs iter {it:3d}] loss={loss.item():.6f} "
              f"(E={mE.item():.6f} T={mT.item():.6f})  val={va_tot:.6f}{marker}")
        return loss

    final_loss = lbfgs.step(closure)
    del p_all, xy_all, E_tr, T_tr
    dt_lbfgs = time.time() - t_lbfgs
    abort = bool(torch.isnan(final_loss) or torch.isinf(final_loss))
    if abort:
        print(f"[lbfgs] ABORT — NaN/Inf; restoring Adam best")
        model.load_state_dict(adam_ckpt["state_dict"])
    print(f"[lbfgs] {len(lbfgs_iter_log)} iterations in {dt_lbfgs:.1f}s")

    overall_best = lbfgs_best_val
    lbfgs_log = []
    if not abort and lbfgs_iter_log:
        last = lbfgs_iter_log[-1]
        lbfgs_log.append(dict(epoch=N_EPOCHS+1, phase="lbfgs",
                              train=last["loss"], train_E=last["mse_E"], train_T=last["mse_T"],
                              val=last["val"], val_E=last["val_E"], val_T=last["val_T"], dt=dt_lbfgs))
    if lbfgs_best_iter >= 0:
        print(f"[lbfgs] best val={lbfgs_best_val:.6f} at iter {lbfgs_best_iter} "
              f"(Adam was {adam_best_val:.6f}, gain={adam_best_val-lbfgs_best_val:.6f})")
    else:
        print(f"[lbfgs] no improvement over Adam best {adam_best_val:.6f}")

    full_log = log + lbfgs_log
    with open(os.path.join(OUTPUT_FOLDER, LOG_NAME), "w") as f:
        json.dump({"log": full_log, "lbfgs_iter_log": lbfgs_iter_log,
                   "best_val_total": overall_best,
                   "best_epoch": best_epoch if overall_best == adam_best_val else N_EPOCHS+1,
                   "adam_best_val": adam_best_val, "adam_best_epoch": best_epoch,
                   "config": dict(batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, lr=LR,
                                  lr_max=LR_MAX, lr_min=LR_MIN, val_frac=VAL_FRAC, seed=SEED,
                                  lbfgs_lr=LBFGS_LR, lbfgs_max_iter=LBFGS_MAX_ITER,
                                  lbfgs_history_size=LBFGS_HISTORY_SIZE, **_ckpt_config())},
                  f, indent=2)
    _plot_curves(full_log, os.path.join(OUTPUT_FOLDER, CURVES_NAME),
                 adam_epochs=N_EPOCHS, lbfgs_iter_log=lbfgs_iter_log)
    print(f"[done] best val {overall_best:.4f}  -> {OUTPUT_FOLDER}")

    # ── Auto-render target-vs-pred from the best checkpoint ──────────────
    try:
        best = torch.load(os.path.join(OUTPUT_FOLDER, CKPT_NAME), map_location=DEVICE)
        model.load_state_dict(best["state_dict"]); model.eval()
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_plot_tvp", os.path.join(_HERE, "plots", "02_plot_nn_target_vs_pred.py"))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        mod.plot_fnn_only(fnn=model, primary=primary, xy=xy,
                          E_true=E_all, T_true=T_all, val_idx=val_idx,
                          output_path=os.path.join(OUTPUT_FOLDER, TVP_NAME))
    except Exception as exc:
        print(f"[plot-tvp] skipped ({exc!r})")


if __name__ == "__main__":
    main()
