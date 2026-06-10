"""Train a reconstruction NN on frozen dual-surrogate outputs.

Pipeline (step 3 of the v6 plan, dual-species):

    fnn_electron + fnn_muon (frozen,                      recon targets
    combined per event by DualSpeciesSurrogate)                  │
       │                                                         ▼
       ▼                                                  primary[:, :4] =
  primary → dual(primary, xy) → (E_comb, T_comb)         [dir_x, dir_y, dir_z, log_e_norm]
                                    │                    z-scored from the data
  recon input per detector = (x, y, E_comb, T_comb)              │
                                    │                            │
                                    └────────────────────────────┘
                                         v3 Reconstruction
                                      (input_features=4, output_dim=4)

The combined response is the COMPLETE physical event: both per-species
surrogates evaluated with the same primary and layout, counts summed and
times count-weight averaged (modules_v6/dual_surrogate.py). The recon
therefore learns to invert the full detector response, matching how stage 4
evaluates layouts.

Permutation augmentation is applied to the recon input on every batch: a
random per-sample permutation of the 100 detectors (with xy, E, T permuted
together; the target is a scalar 4-vector that stays fixed). This teaches
the flat MLP recon to be approximately permutation-invariant in its output.

Shower-level 90/10 split (matches 02). Loads fnn_electron.pt + fnn_muon.pt
from FNN_FOLDER.

Run:

    cd TambOpt/detector_optimization_v6
    python 03_train_recon.py
    python 03_train_recon.py --epochs 2 --lbfgs-iters 3    # smoke run

Artifacts in RECON_FOLDER:
    recon.pt               best-val model checkpoint
    recon_train_log.json   per-epoch train/val MSE (+ per-axis val)
    recon_train_curves.png
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

import modules_v6   # noqa: F401  (side-effect import: injects v3 + v4 into sys.path)
from modules_v6.dual_surrogate import load_dual_surrogate
from modules_v6.reconstruction import Reconstruction
from modules_v6.constants import (
    RECON_FOLDER, TRAINING_DATASET_FOLDER, FNN_FOLDER,
    N_DETECTORS,
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
# Gradient accumulation chunk for the L-BFGS closure. The full train set is
# 3.15M rows × 400 features; running forward+backward on it all at once OOMs
# the GPU (the activations of the 512-hidden MLP are the bottleneck, ~6+ GiB
# per layer). Chunking the closure and accumulating gradients gives the same
# full-batch gradient L-BFGS needs but caps peak memory at O(chunk_size).
LBFGS_CHUNK              = 32_768


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


@torch.no_grad()
def compute_fnn_predictions(model: torch.nn.Module,
                            primary: torch.Tensor,
                            xy:      torch.Tensor,
                            device:  torch.device,
                            batch_size: int = 1024
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the (dual) surrogate forward on the whole corpus. Returns CPU tensors.

    primary and xy are CPU tensors; only batches are moved to device. With the
    DualSpeciesSurrogate, the returned (E, T) are the COMBINED event response —
    the row's own pdg feature is ignored by the wrapper.
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

    The recon target is a scalar 4-vector (dir_x, dir_y, dir_z, log_e_norm) —
    invariant under detector permutations — so we do NOT
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


# ── Loss / eval / checkpoint helpers ────────────────────────────────────────
# All three reused by Adam train, Adam val, L-BFGS closure, and L-BFGS val.

_AXIS_KEYS = ("tot", "dx", "dy", "dz", "logE")


def _per_axis_loss(pred: torch.Tensor, tgt: torch.Tensor, reduction: str = "mean"):
    """Per-axis MSE on the 4 output dims. Returns (total, dx, dy, dz, logE)."""
    l_dx = F.mse_loss(pred[:, 0], tgt[:, 0], reduction=reduction)
    l_dy = F.mse_loss(pred[:, 1], tgt[:, 1], reduction=reduction)
    l_dz = F.mse_loss(pred[:, 2], tgt[:, 2], reduction=reduction)
    l_lE = F.mse_loss(pred[:, 3], tgt[:, 3], reduction=reduction)
    return l_dx + l_dy + l_dz + l_lE, l_dx, l_dy, l_dz, l_lE


@torch.no_grad()
def _validate(recon: Reconstruction,
              loader: DataLoader,
              device: torch.device) -> dict:
    """Mean-MSE on `loader`. Returns dict keyed by _AXIS_KEYS."""
    sums = [0.0] * 5
    n = 0
    for xy_b, E_b, T_b, tgt_b in loader:
        xy_b  = xy_b .to(device, non_blocking=True)
        E_b   = E_b  .to(device, non_blocking=True)
        T_b   = T_b  .to(device, non_blocking=True)
        tgt_b = tgt_b.to(device, non_blocking=True)
        pred  = recon(build_recon_input(xy_b, E_b, T_b))
        losses = _per_axis_loss(pred, tgt_b)
        B = xy_b.shape[0]
        for i, v in enumerate(losses):
            sums[i] += v.item() * B
        n += B
    n = max(n, 1)
    return {k: sums[i] / n for i, k in enumerate(_AXIS_KEYS)}


def _save_ckpt(path: str,
               recon: Reconstruction,
               epoch: int,
               val: dict,
               in_mean: torch.Tensor, in_std: torch.Tensor,
               tgt_mean: torch.Tensor, tgt_std: torch.Tensor,
               **extra) -> None:
    """Standard recon.pt payload — used by both Adam-best and L-BFGS-best saves."""
    torch.save({
        "state_dict": recon.state_dict(),
        "epoch": epoch,
        "val_total": val["tot"],
        "val_dx":    val["dx"],   "val_dy":    val["dy"],
        "val_dz":    val["dz"],   "val_logE":  val["logE"],
        "input_features": RECON_INPUT_FEATURES,
        "num_detectors":  N_DETECTORS,
        "input_mean":  in_mean.detach().cpu(),
        "input_std":   in_std .detach().cpu(),
        "target_mean": tgt_mean.detach().cpu(),
        "target_std":  tgt_std .detach().cpu(),
        "config": dict(hidden=512, n_hidden_layers=3, dropout=0.1, output_dim=4),
        **extra,
    }, path)


def _plot_curves(log, path: str, adam_epochs: int = 0,
                 lbfgs_iter_log=None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Adam entries only (exclude the single lbfgs summary row)
        adam_log = [e for e in log if e.get("phase") != "lbfgs"]
        ep = [e["epoch"] for e in adam_log]

        # Colors fixed: train -> C0, val -> C1. Per-axis plot uses 4 line
        # styles to distinguish the four axes (dx -, dy --, dz :, logE -.).
        # L-BFGS reuses the same colors+styles. Per-curve labels are dropped
        # — the legend is built from proxy handles so it shows SEMANTICS only
        # (color = split, style = axis), not 8 redundant entries.
        from matplotlib.lines import Line2D
        AXIS_STYLES = [("dx", "-"), ("dy", "--"), ("dz", ":"), ("logE", "-.")]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(ep, [e["train"] for e in adam_log], color="C0", label="train")
        axes[0].plot(ep, [e["val"]   for e in adam_log], color="C1", label="val")
        for name, ls in AXIS_STYLES:
            axes[1].plot(ep, [e[f"train_{name}"] for e in adam_log],
                         color="C0", linestyle=ls)
            axes[1].plot(ep, [e[f"val_{name}"]   for e in adam_log],
                         color="C1", linestyle=ls)

        if lbfgs_iter_log:
            lb_ep = [adam_epochs + 1 + e["iter"] for e in lbfgs_iter_log]
            axes[0].plot(lb_ep, [e["loss"] for e in lbfgs_iter_log], color="C0")
            axes[0].plot(lb_ep, [e["val"]  for e in lbfgs_iter_log], color="C1")
            for name, ls in AXIS_STYLES:
                # L-BFGS train per-axis = mse_{name}, val per-axis = val_{name}.
                axes[1].plot(lb_ep, [e[f"mse_{name}"] for e in lbfgs_iter_log],
                             color="C0", linestyle=ls)
                axes[1].plot(lb_ep, [e[f"val_{name}"] for e in lbfgs_iter_log],
                             color="C1", linestyle=ls)
            axes[0].set_yscale("log"); axes[1].set_yscale("log")


        if adam_epochs > 0:
            for ax in axes:
                ax.axvline(adam_epochs, color="gray", linestyle="--", alpha=0.5,
                           label="Adam\u2192L-BFGS")

        axes[0].set_xlabel("epoch / iter"); axes[0].set_ylabel("MSE (normalized labels)")
        axes[0].set_title("total");  axes[0].grid(alpha=0.3); axes[0].legend(fontsize=9)
        axes[1].set_xlabel("epoch / iter"); axes[1].set_ylabel("MSE")
        axes[1].set_title("per-axis"); axes[1].grid(alpha=0.3)
        # Proxy legend: 2 color entries (train/val) + 4 style entries (axes).
        axes[1].legend(handles=[
            Line2D([], [], color="C0",                          label="train"),
            Line2D([], [], color="C1",                          label="val"),
            Line2D([], [], color="black", linestyle="-",        label="dx"),
            Line2D([], [], color="black", linestyle="--",       label="dy"),
            Line2D([], [], color="black", linestyle=":",        label="dz"),
            Line2D([], [], color="black", linestyle="-.",       label="logE"),
        ], ncol=2, fontsize=9, loc="best")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] skipped ({exc!r})")


def main():
    global N_EPOCHS, LBFGS_MAX_ITER
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=N_EPOCHS)
    ap.add_argument("--lbfgs-iters", type=int, default=LBFGS_MAX_ITER)
    args = ap.parse_args()
    N_EPOCHS, LBFGS_MAX_ITER = int(args.epochs), int(args.lbfgs_iters)

    print("=" * 72)
    print("v6/03_train_recon.py — recon on combined dual-species predictions")
    print("=" * 72)
    os.makedirs(RECON_FOLDER, exist_ok=True)
    print(f"training data   : {TRAINING_DATASET_FOLDER}")
    print(f"fnn checkpoints : {FNN_FOLDER}  (fnn_electron.pt + fnn_muon.pt)")
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
    print(f"[load] corpus in {time.time() - t0:.1f}s  primary={tuple(primary.shape)}")

    # Load the frozen dual surrogate: fnn_electron.pt + fnn_muon.pt, each built
    # from its own saved config + norm stats, combined per event by the wrapper
    # (counts add, times average count-weighted).
    dual = load_dual_surrogate(FNN_FOLDER, DEVICE)

    # Predict combined (E, T) on the full corpus once (deterministic, eval mode)
    t0 = time.time()
    E_pred, T_pred = compute_fnn_predictions(dual, primary, xy, DEVICE, batch_size=1024)
    print(f"[dual] combined predictions in {time.time() - t0:.1f}s  "
          f"E mean={E_pred.mean():.3g} std={E_pred.std():.3g}  "
          f"T mean={T_pred.mean():.3g} std={T_pred.std():.3g}")

    # Recon targets = v6 primary encoding [dir_x, dir_y, dir_z, log_e_norm] in raw
    # units (pdg dropped — the combined response describes the whole event).
    # z-score stats are computed directly from the data being trained on; they
    # ship inside recon.pt, so stage 4 stays consistent automatically.
    target   = primary[:, :4].clone().float()                                # (N, 4)
    tgt_mean = target.mean(dim=0)                                            # (4,)
    tgt_std  = target.std(dim=0).clamp(min=1e-8)                             # (4,)
    print(f"[target raw] "
          f"dx in [{target[:,0].min():.3f}, {target[:,0].max():.3f}]  "
          f"dy in [{target[:,1].min():.3f}, {target[:,1].max():.3f}]  "
          f"dz in [{target[:,2].min():.3f}, {target[:,2].max():.3f}]  "
          f"logE in [{target[:,3].min():.3f}, {target[:,3].max():.3f}]")
    print(f"[target stats] mean={tgt_mean.tolist()}  std={tgt_std.tolist()}")

    # Shower-level split
    train_idx, val_idx = shower_level_split(strat_ids, VAL_FRAC, SEED)
    print(f"[split] train pairs={len(train_idx)}  val pairs={len(val_idx)}")

    # Recon-input z-score stats computed from the ACTUAL inputs (xy coordinates
    # and the COMBINED E/T predictions) — the combined event distributions are
    # not any single species model's stats. One shared scalar per feature kind
    # across all detector slots (matches the permutation augmentation).
    per_det_mean = torch.stack([
        xy[..., 0].mean(),        # x
        xy[..., 1].mean(),        # y
        E_pred.mean(),            # E (combined)
        T_pred.mean(),            # T (combined)
    ])
    per_det_std = torch.stack([
        xy[..., 0].std(),
        xy[..., 1].std(),
        E_pred.std(),
        T_pred.std(),
    ]).clamp(min=1e-8)
    in_mean = per_det_mean.repeat(N_DETECTORS)   # (400,)
    in_std  = per_det_std.repeat(N_DETECTORS)
    print(f"[norm] recon per-det stats (from combined predictions)  "
          f"mean={per_det_mean.tolist()}  std={per_det_std.tolist()}")
    # No .to(DEVICE) needed here — stats are pushed to the model via
    # `set_normalization(...)` and live on the model's device as buffers.

    full_ds  = TensorDataset(xy, E_pred, T_pred, target)
    train_ds = Subset(full_ds, train_idx.tolist())
    val_ds   = Subset(full_ds, val_idx.tolist())
    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    # Model — v6 Reconstruction mirrors FNNSurrogate's architecture and
    # normalization contract (hidden=512x3, dropout=0.1, z-score baked into
    # forward via registered buffers).
    torch.manual_seed(SEED)
    recon = Reconstruction(
        n_det=N_DETECTORS,
        input_features=RECON_INPUT_FEATURES,
        output_dim=4,
        hidden=512,
        dropout=0.1,
    ).to(DEVICE)
    recon.set_normalization(
        in_mean  = in_mean.to(DEVICE),
        in_std   = in_std.to(DEVICE),
        out_mean = tgt_mean.to(DEVICE),
        out_std  = tgt_std.to(DEVICE),
    )
    n_params = sum(p.numel() for p in recon.parameters() if p.requires_grad)
    print(f"[model] recon params={n_params:,}")

    optimizer = torch.optim.Adam(recon.parameters(), lr=LR)

    log = []
    best_val   = float("inf")
    best_epoch = -1

    for epoch in range(N_EPOCHS):
        t_epoch = time.time()
        recon.train()
        sums  = [0.0] * 5
        n_tr  = 0
        for xy_b, E_b, T_b, tgt_b in train_loader:
            xy_b  = xy_b .to(DEVICE, non_blocking=True)
            E_b   = E_b  .to(DEVICE, non_blocking=True)
            T_b   = T_b  .to(DEVICE, non_blocking=True)
            tgt_b = tgt_b.to(DEVICE, non_blocking=True)

            # Permutation augmentation (input-only; target stays fixed).
            xy_b, E_b, T_b = permute_detectors_recon(xy_b, E_b, T_b)

            pred = recon(build_recon_input(xy_b, E_b, T_b))   # (B, 4) raw
            losses = _per_axis_loss(pred, tgt_b)              # (total, dx, dy, dz, logE)
            loss = losses[0]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(recon.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

            B = xy_b.shape[0]
            for i, v in enumerate(losses):
                sums[i] += v.item() * B
            n_tr += B
        n_tr = max(n_tr, 1)
        tr = {k: sums[i] / n_tr for i, k in enumerate(_AXIS_KEYS)}

        recon.eval()
        va = _validate(recon, val_loader, DEVICE)

        dt = time.time() - t_epoch
        print(f"[epoch {epoch+1:3d}/{N_EPOCHS}] "
              f"train={tr['tot']:.4f} val={va['tot']:.4f} "
              f"(dx={va['dx']:.4f} dy={va['dy']:.4f} dz={va['dz']:.4f} logE={va['logE']:.4f})  {dt:.1f}s")
        log.append(dict(
            epoch=epoch + 1,
            train=tr['tot'],
            train_dx=tr['dx'], train_dy=tr['dy'], train_dz=tr['dz'], train_logE=tr['logE'],
            val=va['tot'],
            val_dx=va['dx'],   val_dy=va['dy'],   val_dz=va['dz'],   val_logE=va['logE'],
            dt=dt,
        ))

        if va['tot'] < best_val - 1e-6:
            best_val   = va['tot']
            best_epoch = epoch + 1
            _save_ckpt(
                os.path.join(RECON_FOLDER, "recon.pt"),
                recon, epoch + 1, va,
                in_mean, in_std, tgt_mean, tgt_std,
            )

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

    # Move full training set to GPU and pre-compute raw flat input (model does z-score).
    xy_train  = xy[train_idx].to(DEVICE)
    E_train   = E_pred[train_idx].to(DEVICE)
    T_train   = T_pred[train_idx].to(DEVICE)
    tgt_train = target[train_idx].to(DEVICE)
    inp_all   = build_recon_input(xy_train, E_train, T_train)
    del xy_train, E_train, T_train  # free intermediates
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
    N_train = inp_all.shape[0]
    # Track best L-BFGS val and save recon.pt whenever it improves — same
    # pattern as 02_train_fnn.py. Previously only the LAST iter was checked,
    # so a better mid-iter (e.g. before L-BFGS overshoots) was lost.
    lbfgs_best_val  = adam_best_val
    lbfgs_best_iter = -1

    def closure():
        nonlocal lbfgs_best_val, lbfgs_best_iter
        lbfgs_optimizer.zero_grad()
        # Chunked forward+backward — sum-reduction per chunk, divided by
        # N_train so the accumulated gradient equals the mean-reduced gradient
        # over the whole training set (caps peak GPU memory at O(chunk)).
        sums = [0.0] * 5
        for lo in range(0, N_train, LBFGS_CHUNK):
            hi = min(lo + LBFGS_CHUNK, N_train)
            pred_c = recon(inp_all[lo:hi])
            losses = _per_axis_loss(pred_c, tgt_train[lo:hi], reduction="sum")
            (losses[0] / N_train).backward()
            for i, v in enumerate(losses):
                sums[i] += v.item()
        tr = {k: sums[i] / N_train for i, k in enumerate(_AXIS_KEYS)}

        # Validation (no_grad — does not affect L-BFGS gradients).
        va = _validate(recon, val_loader, DEVICE)

        it = len(lbfgs_iter_log)
        lbfgs_iter_log.append(dict(
            iter=it, loss=tr['tot'],
            mse_dx=tr['dx'], mse_dy=tr['dy'],
            mse_dz=tr['dz'], mse_logE=tr['logE'],
            val=va['tot'],
            val_dx=va['dx'], val_dy=va['dy'], val_dz=va['dz'], val_logE=va['logE'],
        ))
        marker = ""
        if va['tot'] < lbfgs_best_val - 1e-6:
            lbfgs_best_val  = va['tot']
            lbfgs_best_iter = it
            _save_ckpt(
                os.path.join(RECON_FOLDER, "recon.pt"),
                recon, N_EPOCHS + 1, va,
                in_mean, in_std, tgt_mean, tgt_std,
                phase="lbfgs", lbfgs_iter=it,
            )
            marker = "  <- NEW BEST (saved)"
        print(f"  [lbfgs iter {it:3d}] loss={tr['tot']:.6f} "
              f"(dx={tr['dx']:.6f} dy={tr['dy']:.6f} "
              f"dz={tr['dz']:.6f} logE={tr['logE']:.6f})  "
              f"val={va['tot']:.6f}{marker}")
        # Return a detached scalar — L-BFGS only calls .item()/float() on the
        # returned value for line search comparisons; gradients are read from
        # `param.grad`, which we populated above via chunked accumulation.
        return torch.tensor(tr['tot'], device=DEVICE)

    final_loss = lbfgs_optimizer.step(closure)

    # Free GPU memory
    del inp_all, tgt_train

    dt_lbfgs = time.time() - t_lbfgs
    n_iters = len(lbfgs_iter_log)
    lbfgs_abort = torch.isnan(final_loss) or torch.isinf(final_loss)

    if lbfgs_abort:
        print(f"[lbfgs] ABORT — NaN/Inf after {n_iters} iters")
        recon.load_state_dict(adam_ckpt["state_dict"])

    print(f"[lbfgs] {n_iters} iterations in {dt_lbfgs:.1f}s")

    # Per-iter best save already happened inside the closure. recon.pt holds
    # the best-ever weights across both Adam and L-BFGS phases.
    overall_best_val = lbfgs_best_val
    lbfgs_log = []

    if not lbfgs_abort and lbfgs_iter_log:
        last = lbfgs_iter_log[-1]
        lbfgs_log.append(dict(
            epoch=N_EPOCHS + 1, phase="lbfgs",
            train=last["loss"],
            train_dx=last["mse_dx"], train_dy=last["mse_dy"],
            train_dz=last["mse_dz"], train_logE=last["mse_logE"],
            val=last["val"],
            val_dx=last["val_dx"], val_dy=last["val_dy"],
            val_dz=last["val_dz"], val_logE=last["val_logE"],
            dt=dt_lbfgs,
        ))

    if lbfgs_best_iter >= 0:
        print(f"[lbfgs] best val={lbfgs_best_val:.6f} at iter {lbfgs_best_iter} "
              f"(Adam was {adam_best_val:.6f}, gain={adam_best_val-lbfgs_best_val:.6f})")
    else:
        print(f"[lbfgs] no improvement over Adam best {adam_best_val:.6f}  "
              f"(recon.pt unchanged)")

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

    # Auto-render the end-to-end (FNN -> recon) target-vs-prediction scatter
    # next to recon.pt using the tensors + models we already have in memory.
    # Refresh recon's state_dict from recon.pt so we plot the best-val
    # iterate, not the last L-BFGS step. Failures shouldn't tank the run.
    try:
        _best = torch.load(os.path.join(RECON_FOLDER, "recon.pt"), map_location=DEVICE)
        recon.load_state_dict(_best["state_dict"])
        recon.eval()
        # Module filename starts with a digit; can't be regular-imported.
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "_plot_tvp",
            os.path.join(_HERE, "plots", "02_plot_nn_target_vs_pred.py"),
        )
        assert _spec is not None and _spec.loader is not None, "importlib spec/loader missing"
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _mod.plot_recon_only(
            fnn=dual, recon=recon,
            primary=primary, xy=xy,
            val_idx=val_idx,
        )
    except Exception as exc:
        print(f"[plot-tvp] skipped ({exc!r})")


if __name__ == "__main__":
    main()
