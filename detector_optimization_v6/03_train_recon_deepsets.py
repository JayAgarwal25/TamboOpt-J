"""Train a DeepSets reconstruction network on frozen dual-surrogate outputs.

Drop-in replacement for 03_train_recon.py using DeepSetsRecon instead of the
flat MLP. Permutation invariance is structural — no augmentation needed.

Architecture:
    token_i = [x_i, y_i, E_comb_i, T_comb_i]     (4 features, per detector)
    h_i     = encoder(token_i)                     shared encoder
    c       = context_proj(cat[mean h_i, max h_i])  invariant pool (maxmean)
    out     = decoder(c)            → [dir_x, dir_y, dir_z, log_e_norm]

Normalization: in_mean/in_std are (4,) — one scalar per feature kind,
broadcast over all detector slots. Output stats are (4,) like the flat MLP.

Writes to RECON_FOLDER + "_deepsets" — safe to run alongside 03_train_recon.py.

Run:
    python 03_train_recon_deepsets.py
    python 03_train_recon_deepsets.py --epochs 2 --lbfgs-iters 3   # smoke run
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset

import modules_v6   # noqa: F401
from modules_v6.dual_surrogate  import load_dual_surrogate, combine_species_outputs
from modules_v6.reconstruction  import DeepSetsRecon
from modules_v6.constants import (
    RECON_FOLDER, TRAINING_DATASET_FOLDER, FNN_FOLDER,
    N_DETECTORS, T_LOG_SCALE,
)

OUTPUT_FOLDER = RECON_FOLDER + "_deepsets"

# ── Architecture ─────────────────────────────────────────────────────────────
HIDDEN   = 256
CONTEXT  = 128
N_ENC    = 3
N_DEC    = 3
POOL     = "maxmean"   # "mean" or "maxmean"; maxmean doubles context_proj input

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 256
N_EPOCHS   = 300
LR         = 3e-5
GRAD_CLIP  = 10.0
VAL_FRAC   = 0.10
SEED       = 1
NUM_WORKERS = 0
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── L-BFGS ───────────────────────────────────────────────────────────────────
LBFGS_LR           = 1.0
LBFGS_MAX_ITER     = 500
LBFGS_HISTORY_SIZE = 20
# Encoder activations dominate: chunk × n_det × hidden × bytes/layer.
# chunk=4096 keeps peak memory ~3 GB on a 40 GB GPU (vs 32 768 for flat MLP).
LBFGS_CHUNK        = 4_096
RESUME_CKPT_INTERVAL = 25   # save a rolling resume checkpoint every N Adam epochs

RECON_INPUT_FEATURES = 4   # (x, y, E, T) per detector


def shower_level_split(strategy_ids: torch.Tensor,
                       val_frac: float,
                       seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shower-level split so all strategies for one shower share a side."""
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


@torch.no_grad()
def compute_fnn_predictions(model: nn.Module,
                             primary: torch.Tensor,
                             xy:      torch.Tensor,
                             device:  torch.device,
                             batch_size: int = 1024
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run dual surrogate forward on the whole corpus. Returns CPU tensors."""
    model.eval()
    N = int(primary.shape[0])
    E_pred = torch.empty((N, N_DETECTORS), dtype=torch.float32)
    T_pred = torch.empty((N, N_DETECTORS), dtype=torch.float32)
    for lo in range(0, N, batch_size):
        hi = min(lo + batch_size, N)
        pred = model(primary[lo:hi].to(device, non_blocking=True),
                     xy[lo:hi].to(device, non_blocking=True))   # (B, 100, 2)
        E_pred[lo:hi] = pred[..., 0].cpu()
        T_pred[lo:hi] = pred[..., 1].cpu()
    return E_pred, T_pred


@torch.no_grad()
def build_kernel_combined_labels(E_raw: torch.Tensor, T_raw: torch.Tensor,
                                 strat_ids: torch.Tensor, device: torch.device,
                                 chunk: int = 100_000
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combine per-species KERNEL labels into the recon's (E,T) input space.

    `E_raw`/`T_raw` are the stored ground-truth kernel labels (raw counts / times),
    one SINGLE-SPECIES shower per row. The recon input space is the physically
    combined electron+muon signal — `combine_species_outputs` applied to
    log1p(counts) / log1p(time*T_LOG_SCALE) — the same space the dual surrogate
    emits and that `eval_true_utility.KernelDualLabels` builds at eval time.

    Row layout (from build_training_pairs): rows are blocked by strategy; within
    each strategy block the first k_sp rows are electron showers 0..k_sp-1 and the
    next k_sp are the paired muon showers (same primary/vertex). So for global row
    r: s = r // rows_per_strat; j = (r % rows_per_strat) % k_sp; electron row =
    s*rows_per_strat + j, muon row = + k_sp. BOTH duplicated rows of an event map
    to the same combined label, so the row set and the train/val split are byte
    identical to the FNN-prediction path — only the (E,T) VALUES change (clean A/B).
    """
    N, n_det = E_raw.shape
    n_strat = int(strat_ids.max().item()) + 1
    assert N % n_strat == 0, "rows not divisible by n_strat"
    rows_per_strat = N // n_strat
    assert rows_per_strat % 2 == 0, "strategy block is not e/mu halves"
    k_sp = rows_per_strat // 2
    expected = torch.arange(n_strat).repeat_interleave(rows_per_strat)
    assert torch.equal(strat_ids.cpu(), expected), \
        "strategy_ids not contiguous-blocked; kernel-pairing assumption invalid"

    r      = torch.arange(N)
    s      = r // rows_per_strat
    j      = (r % rows_per_strat) % k_sp
    e_row  = s * rows_per_strat + j
    mu_row = s * rows_per_strat + k_sp + j

    E_comb = torch.empty((N, n_det), dtype=torch.float32)
    T_comb = torch.empty((N, n_det), dtype=torch.float32)
    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        er, mr = e_row[lo:hi], mu_row[lo:hi]
        pe = torch.stack([torch.log1p(E_raw[er].to(device)),
                          torch.log1p(T_raw[er].to(device) * T_LOG_SCALE)], dim=-1)
        pm = torch.stack([torch.log1p(E_raw[mr].to(device)),
                          torch.log1p(T_raw[mr].to(device) * T_LOG_SCALE)], dim=-1)
        comb = combine_species_outputs(pe, pm)          # (b, n_det, 2)
        E_comb[lo:hi] = comb[..., 0].cpu()
        T_comb[lo:hi] = comb[..., 1].cpu()
    return E_comb, T_comb


# Per-axis log keys. 4-output = direction + energy (default, unchanged). The
# 7-output variant additionally reconstructs the decay vertex (vE, vN, vU).
_BASE_AXIS_KEYS = ("tot", "dx", "dy", "dz", "logE")


def _axis_keys(output_dim: int):
    if output_dim == 7:
        return _BASE_AXIS_KEYS + ("vE", "vN", "vU")
    return _BASE_AXIS_KEYS


def _per_axis_loss(pred: torch.Tensor, tgt: torch.Tensor,
                   reduction: str = "mean", std: torch.Tensor = None):
    """(total, per-column…) MSE, length 1 + pred.shape[1]. Generalizes from the
    4-output (dir+E) recon to the 7-output (+vertex) recon without hardcoding.

    `std` (per-axis, shape (D,)): if given, the MSE is computed in z-scored space
    (pred/std, tgt/std) so axes on very different physical scales — e.g. the
    metre-scale vertex vs the O(1) direction — contribute comparably."""
    if std is not None:
        pred = pred / std
        tgt  = tgt / std
    per = [F.mse_loss(pred[:, j], tgt[:, j], reduction=reduction)
           for j in range(pred.shape[1])]
    return (sum(per), *per)


@torch.no_grad()
def _validate(recon: nn.Module, loader: DataLoader, device: torch.device,
              std: torch.Tensor = None) -> dict:
    keys = _axis_keys(recon.output_dim)
    sums = [0.0] * len(keys)
    n = 0
    for xy_b, E_b, T_b, tgt_b in loader:
        xy_b  = xy_b .to(device, non_blocking=True)
        E_b   = E_b  .to(device, non_blocking=True)
        T_b   = T_b  .to(device, non_blocking=True)
        tgt_b = tgt_b.to(device, non_blocking=True)
        inp   = torch.stack([xy_b[..., 0], xy_b[..., 1], E_b, T_b], dim=-1)
        losses = _per_axis_loss(recon(inp), tgt_b, std=std)
        B = xy_b.shape[0]
        for i, v in enumerate(losses):
            sums[i] += v.item() * B
        n += B
    n = max(n, 1)
    return {k: sums[i] / n for i, k in enumerate(keys)}


def _save_ckpt(path: str,
               recon: DeepSetsRecon,
               epoch: int,
               val: dict,
               in_mean: torch.Tensor, in_std: torch.Tensor,
               tgt_mean: torch.Tensor, tgt_std: torch.Tensor,
               **extra) -> None:
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
        "config": dict(
            model_type="deepsets",
            hidden=HIDDEN, context=CONTEXT,
            n_enc=N_ENC, n_dec=N_DEC, pool=POOL,
            output_dim=recon.output_dim,
        ),
        **extra,
    }, path)


def _plot_curves(log, path: str, adam_epochs: int = 0,
                 lbfgs_iter_log=None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        AXIS_STYLES = [("dx", "-"), ("dy", "--"), ("dz", ":"), ("logE", "-.")]

        adam_log = [e for e in log if e.get("phase") != "lbfgs"]
        ep = [e["epoch"] for e in adam_log]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(ep, [e["train"] for e in adam_log], color="C0", label="train")
        axes[0].plot(ep, [e["val"]   for e in adam_log], color="C1", label="val")
        for name, ls in AXIS_STYLES:
            axes[1].plot(ep, [e[f"train_{name}"] for e in adam_log], color="C0", linestyle=ls)
            axes[1].plot(ep, [e[f"val_{name}"]   for e in adam_log], color="C1", linestyle=ls)

        if lbfgs_iter_log:
            lb_ep = [adam_epochs + 1 + e["iter"] for e in lbfgs_iter_log]
            axes[0].plot(lb_ep, [e["loss"] for e in lbfgs_iter_log], color="C0")
            axes[0].plot(lb_ep, [e["val"]  for e in lbfgs_iter_log], color="C1")
            for name, ls in AXIS_STYLES:
                axes[1].plot(lb_ep, [e[f"mse_{name}"] for e in lbfgs_iter_log], color="C0", linestyle=ls)
                axes[1].plot(lb_ep, [e[f"val_{name}"] for e in lbfgs_iter_log], color="C1", linestyle=ls)
            axes[0].set_yscale("log"); axes[1].set_yscale("log")

        if adam_epochs > 0:
            for ax in axes:
                ax.axvline(adam_epochs, color="gray", linestyle="--", alpha=0.5,
                           label="Adam→L-BFGS")

        axes[0].set_xlabel("epoch / iter"); axes[0].set_ylabel("MSE (normalized labels)")
        axes[0].set_title("total");  axes[0].grid(alpha=0.3); axes[0].legend(fontsize=9)
        axes[1].set_xlabel("epoch / iter"); axes[1].set_ylabel("MSE")
        axes[1].set_title("per-axis"); axes[1].grid(alpha=0.3)
        axes[1].legend(handles=[
            Line2D([], [], color="C0",                    label="train"),
            Line2D([], [], color="C1",                    label="val"),
            Line2D([], [], color="black", linestyle="-",  label="dx"),
            Line2D([], [], color="black", linestyle="--", label="dy"),
            Line2D([], [], color="black", linestyle=":",  label="dz"),
            Line2D([], [], color="black", linestyle="-.", label="logE"),
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
    ap.add_argument("--epochs",        type=int,  default=N_EPOCHS)
    ap.add_argument("--lbfgs-iters",   type=int,  default=LBFGS_MAX_ITER)
    ap.add_argument("--fnn_folder",    type=str,  default=None,
                    help="Override FNN_FOLDER: directory containing fnn_electron.pt "
                         "and fnn_muon.pt.  Use this after adaptive-loop FNN fine-tune "
                         "so the recon is retrained with the updated surrogate's predictions.")
    ap.add_argument("--output_folder", type=str,  default=None,
                    help="Override the recon output directory (default: "
                         "RECON_FOLDER + '_deepsets').  Use a round-suffixed path "
                         "(e.g. …_deepsets_r1) to keep the base recon intact.")
    ap.add_argument("--label_source", type=str, default="fnn", choices=("fnn", "kernel"),
                    help="Recon input (E,T) source: 'fnn' = dual-surrogate predictions "
                         "(default, current behaviour); 'kernel' = the stored ground-truth "
                         "kernel labels E.pt/T.pt, e+mu combined into the same space.")
    ap.add_argument("--output_dim", type=int, default=4, choices=(4, 7),
                    help="Recon target width: 4 = direction+energy (default); "
                         "7 = also reconstruct the decay vertex (rel_E/N/U).")
    ap.add_argument("--noise_scale", type=float, default=0.0,
                    help="Noise augmentation amplitude (fnn source only). 0 = off "
                         "(= C0). >0 adds fresh noise to the FNN means each batch so "
                         "the recon becomes robust to the real (kernel) shower "
                         "fluctuations it meets at deployment; scales the residual.")
    ap.add_argument("--noise_mode", type=str, default="gaussian",
                    choices=("gaussian", "bootstrap"),
                    help="Noise model when noise_scale>0. 'gaussian': per-detector "
                         "Gaussian, std from the (kernel-FNN) residual (independent "
                         "across detectors). 'bootstrap': resample a WHOLE real "
                         "residual row (preserves cross-detector + E-T correlations "
                         "and tails), stratified by shower brightness. Bootstrap is "
                         "the physically honest arm; gaussian is the amplitude sweep.")
    ap.add_argument("--normalize_loss", action="store_true",
                    help="Compute the per-axis MSE in z-scored (normalized) space so "
                         "every output axis contributes comparably. Needed for the "
                         "7-output recon, where the metre-scale vertex target otherwise "
                         "dominates the raw-space loss and starves direction/energy. "
                         "Default off keeps the 4-output runs byte-identical.")
    args = ap.parse_args()
    label_source   = args.label_source
    output_dim     = int(args.output_dim)
    noise_scale    = float(args.noise_scale)
    noise_mode     = args.noise_mode
    normalize_loss = bool(args.normalize_loss)
    N_EPOCHS, LBFGS_MAX_ITER = int(args.epochs), int(args.lbfgs_iters)
    if args.fnn_folder:
        global FNN_FOLDER
        FNN_FOLDER = args.fnn_folder
        import modules_v6.constants as _C
        _C.FNN_FOLDER = args.fnn_folder
    if args.output_folder:
        global OUTPUT_FOLDER
        OUTPUT_FOLDER = args.output_folder

    print("=" * 72)
    print("v6/03_train_recon_deepsets.py — DeepSets recon on dual-species preds")
    print("=" * 72)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"training data : {TRAINING_DATASET_FOLDER}")
    print(f"fnn ckpts     : {FNN_FOLDER}  (fnn_electron.pt + fnn_muon.pt)")
    print(f"output        : {OUTPUT_FOLDER}")
    print(f"device        : {DEVICE}  batch={BATCH_SIZE}  epochs={N_EPOCHS}")
    print(f"arch          : hidden={HIDDEN} context={CONTEXT} n_enc={N_ENC} n_dec={N_DEC} pool={POOL}")

    t0 = time.time()
    primary   = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()
    xy        = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "xy.pt")).float()
    strat_ids = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "strategy_ids.pt")).long()
    print(f"[load] corpus in {time.time() - t0:.1f}s  primary={tuple(primary.shape)}")

    # `dual` is always loaded: it produces the recon input for the FNN source and
    # is reused by the post-hoc target-vs-pred plot.
    dual = load_dual_surrogate(FNN_FOLDER, DEVICE)

    # Recon input (E,T) source — the ONE variable this A/B varies (with output_dim).
    t0 = time.time()
    if label_source == "kernel":
        E_raw = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "E.pt")).float()
        T_raw = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "T.pt")).float()
        E_in, T_in = build_kernel_combined_labels(E_raw, T_raw, strat_ids, DEVICE)
        del E_raw, T_raw
        print(f"[kernel] combined ground-truth labels in {time.time() - t0:.1f}s  "
              f"E mean={E_in.mean():.3g} std={E_in.std():.3g}  "
              f"T mean={T_in.mean():.3g} std={T_in.std():.3g}")
    else:
        E_in, T_in = compute_fnn_predictions(dual, primary, xy, DEVICE)
        print(f"[dual] predictions in {time.time() - t0:.1f}s  "
              f"E mean={E_in.mean():.3g} std={E_in.std():.3g}  "
              f"T mean={T_in.mean():.3g} std={T_in.std():.3g}")

    # Target columns: cols 0-3 = (dir_x, dir_y, dir_z, log_e_norm) always; for the
    # 7-output recon add the decay vertex (cols 5-7 = rel_E/N/U; col 4 = pdg is a
    # conditioning label, not a reconstruction target).
    tgt_cols = [0, 1, 2, 3, 5, 6, 7] if output_dim == 7 else [0, 1, 2, 3]
    target   = primary[:, tgt_cols].clone().float()
    tgt_mean = target.mean(dim=0)
    tgt_std  = target.std(dim=0).clamp(min=1e-8)
    print(f"[target] source={label_source} output_dim={output_dim} cols={tgt_cols}")
    print(f"[target] mean={tgt_mean.tolist()}  std={tgt_std.tolist()}")

    train_idx, val_idx = shower_level_split(strat_ids, VAL_FRAC, SEED)
    print(f"[split] train={len(train_idx)}  val={len(val_idx)}")

    # ── Noise augmentation (fnn source only) ──────────────────────────────────
    # `noise_sigma_*` (gaussian) or `R_E/R_T`+`stratum_pool` (bootstrap) are built
    # from the real (kernel - FNN) residual, the exact train->deploy input shift.
    # Selection/validation then uses the KERNEL labels (the deployment condition),
    # not the clean means.  All tensors CPU; small per-batch slices move to device.
    noise_on      = noise_scale > 0.0
    noise_sigma_E = noise_sigma_T = None        # gaussian
    R_E = R_T = None                            # bootstrap residual bank (N, n_det) CPU
    stratum_pool  = None                        # list[S] of CPU LongTensors of train rows
    row_stratum   = torch.zeros(int(E_in.shape[0]), dtype=torch.long)   # dataset column
    E_ker = T_ker = None                        # kept for kernel-val when noise_on
    if noise_on:
        if label_source != "fnn":
            raise SystemExit("--noise_scale is only valid with --label_source fnn")
        E_raw = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "E.pt")).float()
        T_raw = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "T.pt")).float()
        E_ker, T_ker = build_kernel_combined_labels(E_raw, T_raw, strat_ids, DEVICE)
        del E_raw, T_raw
        resid_E = E_ker - E_in                  # (N, n_det) CPU
        resid_T = T_ker - T_in
        if noise_mode == "gaussian":
            noise_sigma_E = resid_E.std(dim=0).clamp(min=1e-8).to(DEVICE)   # (n_det,)
            noise_sigma_T = resid_T.std(dim=0).clamp(min=1e-8).to(DEVICE)
            print(f"[noise] gaussian scale={noise_scale}  per-det residual std "
                  f"E mean={noise_sigma_E.mean():.3g}  T mean={noise_sigma_T.mean():.3g}")
        else:  # bootstrap: whole real residual rows, stratified by FNN-mean brightness
            R_E, R_T = resid_E, resid_T                                   # CPU banks
            brightness = E_in.sum(dim=1)                                  # (N,)
            S = 8
            edges = torch.quantile(brightness[train_idx].double(),
                                   torch.linspace(0, 1, S + 1).double())
            row_stratum = torch.bucketize(brightness, edges[1:-1].to(brightness.dtype))
            stratum_pool = [train_idx[row_stratum[train_idx] == s] for s in range(S)]
            # A stratum with no train rows can't be sampled; fold it into the global pool.
            stratum_pool = [p if len(p) > 0 else train_idx for p in stratum_pool]
            print(f"[noise] bootstrap scale={noise_scale}  S={S} strata  "
                  f"pool sizes={[int(len(p)) for p in stratum_pool]}")
        del resid_E, resid_T

    # Per-feature z-score: one scalar per feature kind, broadcast over all slots.
    in_mean = torch.stack([xy[..., 0].mean(), xy[..., 1].mean(),
                           E_in.mean(),        T_in.mean()])           # (4,)
    in_std  = torch.stack([xy[..., 0].std(),  xy[..., 1].std(),
                           E_in.std(),         T_in.std()]).clamp(min=1e-8)
    print(f"[norm] per-feat mean={in_mean.tolist()}  std={in_std.tolist()}")

    # Train set carries a per-row stratum column (0 when not bootstrap). Validation
    # uses KERNEL labels when noise_on (deployment condition drives checkpoint pick).
    train_full = TensorDataset(xy, E_in, T_in, target, row_stratum)
    train_ds   = Subset(train_full, train_idx.tolist())
    if noise_on:
        val_full = TensorDataset(xy, E_ker, T_ker, target)
        print("[val] selecting/validating on KERNEL labels (deployment condition)")
    else:
        val_full = TensorDataset(xy, E_in, T_in, target)
    val_ds = Subset(val_full, val_idx.tolist())
    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    torch.manual_seed(SEED)
    recon = DeepSetsRecon(
        n_det=N_DETECTORS, input_features=RECON_INPUT_FEATURES, output_dim=output_dim,
        hidden=HIDDEN, context=CONTEXT, n_enc=N_ENC, n_dec=N_DEC, pool=POOL,
    ).to(DEVICE)
    recon.set_normalization(
        in_mean  = in_mean.to(DEVICE),
        in_std   = in_std.to(DEVICE),
        out_mean = tgt_mean.to(DEVICE),
        out_std  = tgt_std.to(DEVICE),
    )
    n_params = sum(p.numel() for p in recon.parameters() if p.requires_grad)
    print(f"[model] DeepSetsRecon  params={n_params:,}")

    # Per-axis z-scored loss (balances physical scales) when --normalize_loss.
    # recon.out_std == tgt_std, set above.
    loss_std = recon.out_std if normalize_loss else None
    if normalize_loss:
        print(f"[loss] normalized per-axis MSE; out_std={recon.out_std.tolist()}")

    optimizer = torch.optim.Adam(recon.parameters(), lr=LR)

    # gpu_requeue can preempt mid-training; recon_resume.pt lets the job
    # continue from the last RESUME_CKPT_INTERVAL-epoch checkpoint rather than
    # restarting cold.  adam_done=True means Adam finished; jump to L-BFGS.
    resume_path = os.path.join(OUTPUT_FOLDER, "recon_resume.pt")
    adam_done   = False
    start_epoch = 0
    log        = []
    best_val   = float("inf")
    best_epoch = -1
    if os.path.exists(resume_path):
        _r = torch.load(resume_path, map_location=DEVICE)
        if _r.get("adam_done"):
            adam_done = True
            print("[resume] Adam done, jumping to L-BFGS")
        else:
            recon.load_state_dict(_r["state_dict"])
            optimizer.load_state_dict(_r["optimizer"])
            start_epoch, log = _r["epoch"], _r["log"]
            best_val, best_epoch = _r["best_val"], _r["best_epoch"]
            print(f"[resume] epoch {start_epoch}  best_val={best_val:.6f}")

    for epoch in range(start_epoch if not adam_done else N_EPOCHS, N_EPOCHS):
        t_epoch = time.time()
        recon.train()
        sums = [0.0] * (1 + output_dim)
        n_tr = 0
        for xy_b, E_b, T_b, tgt_b, strat_b in train_loader:
            xy_b  = xy_b .to(DEVICE, non_blocking=True)
            E_b   = E_b  .to(DEVICE, non_blocking=True)
            T_b   = T_b  .to(DEVICE, non_blocking=True)
            tgt_b = tgt_b.to(DEVICE, non_blocking=True)

            # Noise augmentation (train only): FRESH noise per batch so the recon
            # sees many noisy realizations of each mean, not one fixed sample (that
            # is what T1 memorized). gaussian = per-detector; bootstrap = a whole
            # real residual row (keeps cross-detector/E-T correlation), stratified.
            if noise_on and noise_mode == "gaussian":
                E_b = (E_b + noise_scale * noise_sigma_E * torch.randn_like(E_b)).clamp(min=0.0)
                T_b = (T_b + noise_scale * noise_sigma_T * torch.randn_like(T_b)).clamp(min=0.0)
            elif noise_on:  # bootstrap
                pick = torch.empty(strat_b.shape[0], dtype=torch.long)
                for s in range(len(stratum_pool)):
                    m = (strat_b == s)
                    k = int(m.sum())
                    if k:
                        pool = stratum_pool[s]
                        pick[m] = pool[torch.randint(len(pool), (k,))]
                E_b = (E_b + noise_scale * R_E[pick].to(DEVICE)).clamp(min=0.0)
                T_b = (T_b + noise_scale * R_T[pick].to(DEVICE)).clamp(min=0.0)

            inp    = torch.stack([xy_b[..., 0], xy_b[..., 1], E_b, T_b], dim=-1)
            losses = _per_axis_loss(recon(inp), tgt_b, std=loss_std)

            optimizer.zero_grad(set_to_none=True)
            losses[0].backward()
            torch.nn.utils.clip_grad_norm_(recon.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

            B = xy_b.shape[0]
            for i, v in enumerate(losses):
                sums[i] += v.item() * B
            n_tr += B
        n_tr = max(n_tr, 1)
        tr = {k: sums[i] / n_tr for i, k in enumerate(_axis_keys(output_dim))}

        recon.eval()
        va = _validate(recon, val_loader, DEVICE, std=loss_std)

        dt = time.time() - t_epoch
        print(f"[epoch {epoch+1:3d}/{N_EPOCHS}] "
              f"train={tr['tot']:.4f} val={va['tot']:.4f} "
              f"(dx={va['dx']:.4f} dy={va['dy']:.4f} dz={va['dz']:.4f} "
              f"logE={va['logE']:.4f})  {dt:.1f}s")
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
            _save_ckpt(os.path.join(OUTPUT_FOLDER, "recon.pt"),
                       recon, epoch + 1, va, in_mean, in_std, tgt_mean, tgt_std)

        if (epoch + 1) % RESUME_CKPT_INTERVAL == 0:
            torch.save({
                "state_dict": recon.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1, "log": log,
                "best_val": best_val, "best_epoch": best_epoch,
            }, resume_path)

    if not adam_done:
        with open(os.path.join(OUTPUT_FOLDER, "recon_train_log.json"), "w") as f:
            json.dump({
                "log": log,
                "best_val_total": best_val,
                "best_epoch": best_epoch,
                "config": dict(
                    batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, lr=LR,
                    grad_clip=GRAD_CLIP, val_frac=VAL_FRAC, seed=SEED,
                    hidden=HIDDEN, context=CONTEXT, n_enc=N_ENC, n_dec=N_DEC, pool=POOL,
                    output_dim=output_dim, label_source=label_source, noise_scale=noise_scale, noise_mode=noise_mode, normalize_loss=normalize_loss,
                ),
            }, f, indent=2)
        _plot_curves(log, os.path.join(OUTPUT_FOLDER, "recon_train_curves.png"))
        print(f"[adam done] best val {best_val:.4f} at epoch {best_epoch}")
        torch.save({"adam_done": True}, resume_path)

    if noise_on:
        # Noise-aug runs SKIP L-BFGS: a single fixed-noise full-batch L-BFGS would
        # minimize the T1 "one fixed realization" objective and could undo the
        # fresh-noise robustness. recon.pt (best on kernel-val) is the final model.
        print("[noise] skipping L-BFGS fine-tuning for the noise-augmented run.")
        print(f"[done] best recon (kernel-val) {best_val:.4f} at epoch {best_epoch}"
              f"  -> {OUTPUT_FOLDER}")
        return

    # ── Phase 2: L-BFGS fine-tuning ─────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Phase 2: L-BFGS fine-tuning (full-batch)")
    print("=" * 72)

    adam_ckpt = torch.load(os.path.join(OUTPUT_FOLDER, "recon.pt"), map_location=DEVICE)
    recon.load_state_dict(adam_ckpt["state_dict"])
    adam_best_val = adam_ckpt["val_total"]
    print(f"[lbfgs] loaded Adam best  epoch={adam_ckpt['epoch']}  val={adam_best_val:.6f}")

    recon.eval()   # dropout off; grad stays True

    # Pre-stack full training set as (N_train, n_det, 4) on GPU.
    xy_train  = xy[train_idx].to(DEVICE)
    E_train   = E_in[train_idx].to(DEVICE)
    T_train   = T_in[train_idx].to(DEVICE)
    tgt_train = target[train_idx].to(DEVICE)
    # (L-BFGS only runs for non-noise-aug runs; noise-aug returned above.)
    inp_all   = torch.stack([xy_train[..., 0], xy_train[..., 1],
                             E_train, T_train], dim=-1)    # (N_train, n_det, 4)
    del xy_train, E_train, T_train
    print(f"[lbfgs] full train set on {DEVICE}: {tgt_train.shape[0]} samples")

    lbfgs_optimizer = torch.optim.LBFGS(
        recon.parameters(),
        lr=LBFGS_LR,
        max_iter=LBFGS_MAX_ITER,
        history_size=LBFGS_HISTORY_SIZE,
        line_search_fn="strong_wolfe",
    )

    lbfgs_iter_log  = []
    lbfgs_best_val  = adam_best_val
    lbfgs_best_iter = -1
    N_train = inp_all.shape[0]
    t_lbfgs = time.time()

    def closure():
        nonlocal lbfgs_best_val, lbfgs_best_iter
        lbfgs_optimizer.zero_grad()
        sums = [0.0] * (1 + output_dim)
        for lo in range(0, N_train, LBFGS_CHUNK):
            hi = min(lo + LBFGS_CHUNK, N_train)
            losses = _per_axis_loss(recon(inp_all[lo:hi]),
                                    tgt_train[lo:hi], reduction="sum", std=loss_std)
            (losses[0] / N_train).backward()
            for i, v in enumerate(losses):
                sums[i] += v.item()
        tr = {k: sums[i] / N_train for i, k in enumerate(_axis_keys(output_dim))}
        va = _validate(recon, val_loader, DEVICE, std=loss_std)

        it = len(lbfgs_iter_log)
        lbfgs_iter_log.append(dict(
            iter=it, loss=tr['tot'],
            mse_dx=tr['dx'], mse_dy=tr['dy'], mse_dz=tr['dz'], mse_logE=tr['logE'],
            val=va['tot'],
            val_dx=va['dx'], val_dy=va['dy'], val_dz=va['dz'], val_logE=va['logE'],
        ))
        marker = ""
        if va['tot'] < lbfgs_best_val - 1e-6:
            lbfgs_best_val  = va['tot']
            lbfgs_best_iter = it
            _save_ckpt(os.path.join(OUTPUT_FOLDER, "recon.pt"),
                       recon, N_EPOCHS + 1, va, in_mean, in_std, tgt_mean, tgt_std,
                       phase="lbfgs", lbfgs_iter=it)
            marker = "  <- NEW BEST (saved)"
        print(f"  [lbfgs iter {it:3d}] loss={tr['tot']:.6f} "
              f"val={va['tot']:.6f}{marker}")
        return torch.tensor(tr['tot'], device=DEVICE)

    final_loss = lbfgs_optimizer.step(closure)
    del inp_all, tgt_train
    if os.path.exists(resume_path):
        os.remove(resume_path)

    dt_lbfgs = time.time() - t_lbfgs
    lbfgs_abort = torch.isnan(final_loss) or torch.isinf(final_loss)
    if lbfgs_abort:
        print(f"[lbfgs] ABORT — NaN/Inf after {len(lbfgs_iter_log)} iters")
        recon.load_state_dict(adam_ckpt["state_dict"])
    print(f"[lbfgs] {len(lbfgs_iter_log)} iterations in {dt_lbfgs:.1f}s")

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
              f"(Adam was {adam_best_val:.6f}, gain={adam_best_val - lbfgs_best_val:.6f})")
    else:
        print(f"[lbfgs] no improvement over Adam  (recon.pt unchanged)")

    full_log = log + lbfgs_log
    with open(os.path.join(OUTPUT_FOLDER, "recon_train_log.json"), "w") as f:
        json.dump({
            "log": full_log,
            "lbfgs_iter_log": lbfgs_iter_log,
            "best_val_total": overall_best_val,
            "best_epoch": best_epoch if overall_best_val == adam_best_val else N_EPOCHS + 1,
            "adam_best_val": adam_best_val,
            "adam_best_epoch": best_epoch,
            "config": dict(
                batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, lr=LR,
                grad_clip=GRAD_CLIP, val_frac=VAL_FRAC, seed=SEED,
                lbfgs_lr=LBFGS_LR, lbfgs_max_iter=LBFGS_MAX_ITER,
                lbfgs_history_size=LBFGS_HISTORY_SIZE,
                hidden=HIDDEN, context=CONTEXT, n_enc=N_ENC, n_dec=N_DEC, pool=POOL,
                output_dim=output_dim, label_source=label_source, noise_scale=noise_scale, noise_mode=noise_mode, normalize_loss=normalize_loss,
            ),
        }, f, indent=2)
    _plot_curves(full_log, os.path.join(OUTPUT_FOLDER, "recon_train_curves.png"),
                 adam_epochs=N_EPOCHS, lbfgs_iter_log=lbfgs_iter_log)
    print(f"[done] best recon val {overall_best_val:.4f}  -> {OUTPUT_FOLDER}")

    # Auto-render target-vs-prediction scatter using best checkpoint.
    try:
        _best = torch.load(os.path.join(OUTPUT_FOLDER, "recon.pt"), map_location=DEVICE)
        recon.load_state_dict(_best["state_dict"])
        recon.eval()
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "_plot_tvp",
            os.path.join(_HERE, "plots", "02_plot_nn_target_vs_pred.py"),
        )
        assert _spec is not None and _spec.loader is not None
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _mod.plot_recon_only(fnn=dual, recon=recon, primary=primary, xy=xy,
                             val_idx=val_idx,
                             output_path=os.path.join(OUTPUT_FOLDER,
                                                      "recon_target_vs_pred.png"))
    except Exception as exc:
        print(f"[plot-tvp] skipped ({exc!r})")


if __name__ == "__main__":
    main()
