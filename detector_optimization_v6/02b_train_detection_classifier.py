"""Train a DETECTION classifier: P(detected | primary, layout), a single
Adam(OneCycle) training session on top of the same DeepSets machinery the
E/T surrogates use (modules_v6/detection_classifier.py).

Motivation: the project's utility currently only scores reconstruction
quality on events that already fired; there is no term for whether an
event is seen AT ALL. This trainer supplies that missing aperture surrogate
as a binary classifier, "detecting yes vs no", trained with a single sigmoid
logit + BCEWithLogitsLoss (the numerically stable two-class form of a
softmax, see modules_v6/detection_classifier.py's forward() docstring for
why a 2-logit softmax would be a redundant parameterization of the same
model).

CRITICAL data note: `E.pt` on disk is ALREADY log1p(counts).
01_build_dataset_northeast.py applies `E = torch.log1p(E)` right before
saving (see that file's main(), the `[log1p] E range ...` print). Recovering
raw per-detector counts for the N_MIN/E_MIN threshold below therefore
requires `torch.expm1(E)`; thresholding the log1p values directly would
silently move the counts cut onto a nonlinear log scale. `build_detection_
labels` below does this inversion before any label arithmetic.

Physics note on species rows: the dataset (01_build_dataset_northeast.py /
modules_v6/fnn_surrogate_ne.build_training_pairs) stores TWO rows per
physical event: an electron-component row and a paired muon-component row,
sharing the same primary and layout (see constants.py's DUAL_SPECIES_IDS_PATH
comment). A label built from a single row only counts one species'
component, which is not what "the event was detected" physically means.
`--combine-species` (default on) adds the electron and muon raw counts of
the SAME event, per detector, before thresholding, using the exact
strategy-major row-pairing arithmetic that 03_train_recon_deepsets.py's
`build_kernel_combined_labels` uses to combine per-species KERNEL labels
(factored out here as `pair_row_indices` so tests can call it directly).
Both rows of a pair get the identical label and both are kept, so the row
set and the train/val split are unchanged by combining, exactly as
`build_kernel_combined_labels` keeps them unchanged.

Run:

    cd TambOpt/detector_optimization_v6
    python 02b_train_detection_classifier.py
    python 02b_train_detection_classifier.py --n-min 3 --e-min 2.0
    python 02b_train_detection_classifier.py --no-combine-species   # per-species labels
    python 02b_train_detection_classifier.py --epochs 2             # smoke run
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
from torch.utils.data import TensorDataset, DataLoader, Subset

import modules_v6  # triggers sys.path injection for v3 + v4
from modules_v6.detection_classifier import DetectionClassifier
from modules_v6 import run_world
from modules_v6.constants import (
    N_DETECTORS, PRIMARY_DIM,
)

# ── Config ───────────────────────────────────────────────────────────────────
# Bound in main() from the resolved run world, never at import.
OUTPUT_FOLDER = None

BATCH_SIZE          = 256
N_EPOCHS            = 60
LR                  = 1e-5     # OneCycle initial
LR_MAX              = 3e-4     # OneCycle peak (overridable via --lr)
LR_MIN              = 1e-6     # OneCycle floor
ONECYCLE_PCT_START  = 0.10
GRAD_CLIP           = 10.0
VAL_FRAC            = 0.10
SEED                = 0
NUM_WORKERS         = 0
DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESUME_CKPT_INTERVAL = 10   # save a rolling resume checkpoint every N epochs (gpu_requeue preemption)

# DeepSets shape. Smaller than the E/T surrogates (DS_HIDDEN=256 in
# 02_train_fnn_deepsets.py) since a single yes/no logit is a much lower-rank
# target than per-detector (E, T) regression.
DC_HIDDEN   = 128
DC_CONTEXT  = 64
DC_N_ENC    = 3
DC_N_DEC    = 3
DC_DROPOUT  = 0.0
DC_POOL     = "maxmean"

E_MIN_DEFAULT = 1.0   # raw-count threshold per detector for "fired"
N_MIN_DEFAULT = 5     # minimum number of fired detectors for "detected"

# Fraction below/above which the (n_min, e_min) criterion is treated as
# degenerate (all-or-nothing labels make the whole classifier meaningless).
DEGENERATE_LO = 0.01
DEGENERATE_HI = 0.99


def shower_level_split(strategy_ids: torch.Tensor,
                       val_frac: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shower-level split. All strategy entries of a shower share a split.
    Pairs are strategy-major: pair k's shower index is k - strat[k]*n_showers.

    Copied verbatim from 02_train_fnn_deepsets.py rather than imported,
    since that file's name starts with a digit, which Python cannot import as a
    module name.
    """
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


def pair_row_indices(strategy_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """For every row r, return the (e_row, mu_row) global indices of the
    electron/muon components of the physical event r belongs to.

    Identical arithmetic to 03_train_recon_deepsets.py's
    build_kernel_combined_labels: rows are blocked by strategy (layout);
    within a strategy block of size rows_per_strat, the first half (k_sp
    rows) are electron showers 0..k_sp-1 and the next k_sp rows are the
    paired muon showers of the SAME primaries (build_training_pairs writes
    them in that order so both species see the same layout). For global row
    r: s = r // rows_per_strat, j = (r % rows_per_strat) % k_sp,
    e_row = s*rows_per_strat + j, mu_row = e_row + k_sp. This holds
    regardless of whether r ITSELF is an electron or muon row, so both rows
    of a pair resolve to the identical (e_row, mu_row); that is exactly
    why combining on this pairing and re-broadcasting to every row assigns
    the same label to both rows of an event.
    """
    N = int(strategy_ids.shape[0])
    n_strat = int(strategy_ids.max().item()) + 1
    assert N % n_strat == 0, "rows not divisible by n_strat"
    rows_per_strat = N // n_strat
    assert rows_per_strat % 2 == 0, "strategy block is not e/mu halves"
    k_sp = rows_per_strat // 2
    expected = torch.arange(n_strat).repeat_interleave(rows_per_strat)
    assert torch.equal(strategy_ids.cpu(), expected), \
        "strategy_ids not contiguous-blocked; e/mu row pairing assumption invalid"

    r      = torch.arange(N)
    s      = r // rows_per_strat
    j      = (r % rows_per_strat) % k_sp
    e_row  = s * rows_per_strat + j
    mu_row = s * rows_per_strat + k_sp + j
    return e_row, mu_row


def build_detection_labels(E_log1p:       torch.Tensor,
                           strategy_ids:  torch.Tensor,
                           e_min:         float,
                           n_min:         int,
                           combine_species: bool) -> torch.Tensor:
    """Build the float {0., 1.} detection label per row.

    `E_log1p` is the on-disk E.pt tensor, ALREADY log1p(counts) (see the
    module docstring's CRITICAL note), inverted here with expm1 before any
    counting/thresholding.

        detected := (# detectors with raw counts > e_min) >= n_min

    When combine_species, the electron and muon raw counts of the SAME
    physical event are added per-detector before thresholding (a real event
    is both components together, and the array's actual trigger sees their
    sum, not either alone); both rows of the pair receive the identical
    label via pair_row_indices. When not combine_species, each row (a
    single-species component in isolation) is thresholded on its own,
    useful only as a diagnostic, since it is not the physical detection
    condition.
    """
    counts = torch.expm1(E_log1p)             # (N, n_det) raw counts, undo the disk log1p
    if combine_species:
        e_row, mu_row = pair_row_indices(strategy_ids)
        combined = counts[e_row] + counts[mu_row]      # (N, n_det), broadcast to every row
        n_fired = (combined > e_min).sum(dim=1)
    else:
        n_fired = (counts > e_min).sum(dim=1)
    return (n_fired >= n_min).float()


def compute_input_normalization(primary: torch.Tensor, xy: torch.Tensor) -> dict:
    """Input-only z-score stats: identical arithmetic to the INPUT half of
    modules_v6.fnn_surrogate.compute_normalization (primary kept per-feature,
    all xy_x slots sharing one stat and all xy_y slots sharing another, so
    detector permutation leaves the z-score unchanged) but computed without
    requiring E/T output tensors, since this classifier has no z-scored
    physical output channel to fit stats for. The returned dict's "in_mean"/
    "in_std" keys are exactly what DetectionClassifier.set_normalization
    reads, so it is a drop-in stats source even though it lacks
    compute_normalization's "out_mean"/"out_std" keys.
    """
    n_det = xy.shape[1]
    p_mean = primary.mean(dim=0)
    p_std  = primary.std(dim=0).clamp(min=1e-10)

    xy_x = xy[..., 0]
    xy_y = xy[..., 1]
    x_mean, x_std = xy_x.mean(), xy_x.std()
    y_mean, y_std = xy_y.mean(), xy_y.std()

    xy_mean = torch.stack([x_mean.expand(n_det), y_mean.expand(n_det)], dim=-1).reshape(-1)
    xy_std  = torch.stack([x_std.expand(n_det),  y_std.expand(n_det)],  dim=-1).reshape(-1)

    return dict(in_mean=torch.cat([p_mean, xy_mean]),
               in_std=torch.cat([p_std,  xy_std]))


def compute_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank-based (Mann-Whitney U) AUC-ROC, implemented without sklearn to
    avoid adding a dependency for one metric.

    AUC = P(score(random positive) > score(random negative)), which equals
    the Mann-Whitney U statistic normalized by n_pos*n_neg:

        AUC = (sum_of_ranks(positives) - n_pos*(n_pos+1)/2) / (n_pos * n_neg)

    where ranks are 1-indexed ascending ranks over ALL scores, with tied
    scores given the average rank of their tie block (the standard
    tie-correction, and the reason a fully-tied random dataset scores 0.5
    rather than an arbitrary 0/1 depending on sort order).
    """
    scores = scores.detach().cpu().double()
    labels = labels.detach().cpu().double()
    n_pos = int((labels == 1).sum().item())
    n_neg = int((labels == 0).sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = torch.argsort(scores)
    sorted_scores = scores[order]
    n = int(sorted_scores.shape[0])
    ranks_sorted = torch.empty(n, dtype=torch.double)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0   # 1-indexed average rank over tie block [i, j]
        ranks_sorted[i:j + 1] = avg_rank
        i = j + 1

    ranks = torch.empty(n, dtype=torch.double)
    ranks[order] = ranks_sorted
    sum_ranks_pos = ranks[labels == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_binary_metrics(logits: torch.Tensor, labels: torch.Tensor,
                           threshold: float = 0.5) -> dict:
    """Accuracy/precision/recall/F1 at a fixed 0.5 probability threshold,
    plus threshold-free AUC. Computed once over the FULL split (not
    batch-averaged) since precision/recall/F1/AUC are not linear in the
    batch and averaging per-batch values would bias them, especially with
    an imbalanced minority class and small batches.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    tp = float(((preds == 1) & (labels == 1)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    tn = float(((preds == 0) & (labels == 0)).sum())
    accuracy  = (tp + tn) / max(tp + fp + fn + tn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall    = tp / max(tp + fn, 1.0)
    f1        = 2 * precision * recall / max(precision + recall, 1e-12)
    auc       = compute_auc(probs, labels)
    return dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1, auc=auc)


def _check_degenerate(frac: float, label: str) -> None:
    if frac < DEGENERATE_LO or frac > DEGENERATE_HI:
        print(f"[WARNING] {label} detected fraction = {frac:.4f} is nearly "
              f"degenerate (outside [{DEGENERATE_LO}, {DEGENERATE_HI}]); "
              f"the (n_min, e_min) criterion barely separates the corpus; "
              f"a classifier trained on this label is close to meaningless.")


def _ckpt_config(args) -> dict:
    return dict(
        model_type="detection_deepsets",
        n_det=N_DETECTORS, primary_dim=PRIMARY_DIM,
        hidden=DC_HIDDEN, context=DC_CONTEXT, n_enc=DC_N_ENC, n_dec=DC_N_DEC,
        pool=DC_POOL, dropout=DC_DROPOUT,
        e_min=args.e_min, n_min=args.n_min, combine_species=args.combine_species,
    )


def _plot_curves(log, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep = [e["epoch"] for e in log]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(ep, [e["train_loss"] for e in log], color="C0", label="train")
        axes[0].plot(ep, [e["val_loss"]   for e in log], color="C1", label="val")
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("BCE loss")
        axes[0].set_title("loss"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=9)

        axes[1].plot(ep, [e["val_auc"]    for e in log], color="C1", label="AUC")
        axes[1].plot(ep, [e["val_f1"]     for e in log], color="C2", label="F1")
        axes[1].plot(ep, [e["val_accuracy"] for e in log], color="C3", label="accuracy")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("metric")
        axes[1].set_title("val metrics"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=9)

        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] skipped ({exc!r})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_folder", type=str, default=None,
                    help="Where to write the checkpoint/log/plot "
                         "(default: RUN_LOCATION/test_v6_run_02b_detection).")
    ap.add_argument("--epochs", type=int, default=N_EPOCHS,
                    help="Adam epochs (default from config).")
    ap.add_argument("--e-min", dest="e_min", type=float, default=E_MIN_DEFAULT,
                    help=f"Raw-count threshold per detector for 'fired' (default {E_MIN_DEFAULT}).")
    ap.add_argument("--n-min", dest="n_min", type=int, default=N_MIN_DEFAULT,
                    help=f"Minimum number of fired detectors for 'detected' (default {N_MIN_DEFAULT}).")
    ap.add_argument("--combine-species", dest="combine_species", action="store_true",
                    default=True,
                    help="Add electron+muon raw counts of the same event before "
                         "thresholding (default: on; this is the physical event).")
    ap.add_argument("--no-combine-species", dest="combine_species", action="store_false",
                    help="Threshold each species row independently (diagnostic only).")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR_MAX,
                    help=f"OneCycleLR peak learning rate (default {LR_MAX}).")
    run_world.add_run_world_args(ap)
    args = ap.parse_args()

    global OUTPUT_FOLDER
    W = run_world.resolve(args, need_write=True)
    args.dataset_folder = W.dataset_folder
    OUTPUT_FOLDER = args.output_folder or os.path.join(
        W.root, "test_v6_run_02b_detection")
    args.output_folder = OUTPUT_FOLDER

    os.makedirs(args.output_folder, exist_ok=True)

    print("=" * 72)
    print("v6/02b_train_detection_classifier.py: DeepSets detection classifier")
    print("=" * 72)
    print(f"device          : {DEVICE}")
    print(f"data input dir  : {args.dataset_folder}")
    print(f"output dir      : {args.output_folder}")
    print(f"batch           : {args.batch_size}   epochs: {args.epochs}")
    print(f"lr              : {LR} -> {args.lr} -> {LR_MIN} OneCycleLR (pct_start={ONECYCLE_PCT_START})")
    print(f"deepsets        : hidden={DC_HIDDEN} context={DC_CONTEXT} "
          f"enc={DC_N_ENC} dec={DC_N_DEC} pool={DC_POOL}")
    print(f"label           : detected = (#detectors with raw counts > {args.e_min}) "
          f">= {args.n_min}   combine_species={args.combine_species}")

    # ── Load ─────────────────────────────────────────────────────────────
    t0 = time.time()
    primary      = torch.load(os.path.join(args.dataset_folder, "primary.pt")).float()
    xy           = torch.load(os.path.join(args.dataset_folder, "xy.pt")).float()
    E            = torch.load(os.path.join(args.dataset_folder, "E.pt")).float()
    strategy_ids = torch.load(os.path.join(args.dataset_folder, "strategy_ids.pt")).long()
    print(f"[load] corpus in {time.time()-t0:.1f}s  primary={tuple(primary.shape)}  "
          f"xy={tuple(xy.shape)}  E={tuple(E.shape)}")

    labels = build_detection_labels(E, strategy_ids, args.e_min, args.n_min, args.combine_species)
    overall_frac = float(labels.mean())
    print(f"[label] overall detected fraction = {overall_frac:.4f} "
          f"({int(labels.sum())}/{labels.shape[0]})")
    _check_degenerate(overall_frac, "overall")

    train_idx, val_idx = shower_level_split(strategy_ids, VAL_FRAC, SEED)
    train_frac = float(labels[train_idx].mean())
    val_frac_d = float(labels[val_idx].mean())
    print(f"[split] train pairs={len(train_idx)} (detected frac={train_frac:.4f})  "
          f"val pairs={len(val_idx)} (detected frac={val_frac_d:.4f})")
    _check_degenerate(train_frac, "train")
    _check_degenerate(val_frac_d, "val")

    # ── Normalization (input-only, see compute_input_normalization) ───────
    norm_stats = compute_input_normalization(primary, xy)

    full_ds  = TensorDataset(primary, xy, labels)
    train_ds = Subset(full_ds, train_idx.tolist())
    val_ds   = Subset(full_ds, val_idx.tolist())
    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    # ── Class-imbalance-aware loss ─────────────────────────────────────────
    # Detected events can be a small minority under a strict (n_min, e_min);
    # pos_weight (computed from the TRAIN split only, never val) rescales the
    # positive term of BCEWithLogitsLoss so a classifier can't minimize loss
    # by just predicting "not detected" for everything.
    n_pos_train = float(labels[train_idx].sum())
    n_neg_train = float(len(train_idx)) - n_pos_train
    pos_weight  = torch.tensor([n_neg_train / max(n_pos_train, 1.0)], device=DEVICE)
    print(f"[loss] pos_weight={pos_weight.item():.4f} "
          f"(train n_pos={int(n_pos_train)} n_neg={int(n_neg_train)})")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Model ────────────────────────────────────────────────────────────
    torch.manual_seed(SEED)
    model = DetectionClassifier(
        n_det=N_DETECTORS, primary_dim=PRIMARY_DIM,
        hidden=DC_HIDDEN, context=DC_CONTEXT,
        n_enc=DC_N_ENC, n_dec=DC_N_DEC, dropout=DC_DROPOUT, pool=DC_POOL,
    ).to(DEVICE)
    model.set_normalization(norm_stats)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] DetectionClassifier params={n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=ONECYCLE_PCT_START,
        anneal_strategy="cos",
        div_factor=args.lr / LR,
        final_div_factor=LR / LR_MIN,
    )

    ckpt_path   = os.path.join(args.output_folder, "detection_classifier.pt")
    log_path    = os.path.join(args.output_folder, "detection_train_log.json")
    curves_path = os.path.join(args.output_folder, "detection_train_curves.png")
    resume_path = os.path.join(args.output_folder, "detection_resume.pt")

    # gpu_requeue can preempt mid-training; detection_resume.pt lets the job
    # continue from the last RESUME_CKPT_INTERVAL-epoch checkpoint rather
    # than restarting cold.
    start_epoch = 0
    log        = []
    best_val_auc = -1.0
    best_epoch   = -1
    if os.path.exists(resume_path):
        _r = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(_r["state_dict"])
        optimizer.load_state_dict(_r["optimizer"])
        scheduler.load_state_dict(_r["scheduler"])
        start_epoch, log = _r["epoch"], _r["log"]
        best_val_auc, best_epoch = _r["best_val_auc"], _r["best_epoch"]
        print(f"[resume] epoch {start_epoch}  best_val_auc={best_val_auc:.4f} at epoch {best_epoch}")

    # ── Train (Adam + OneCycleLR only, no L-BFGS phase for a classifier) ──
    for epoch in range(start_epoch, args.epochs):
        t_epoch = time.time()
        model.train()
        tr_loss_sum, n_tr = 0.0, 0
        for p_b, xy_b, y_b in train_loader:
            p_b  = p_b.to(DEVICE, non_blocking=True)
            xy_b = xy_b.to(DEVICE, non_blocking=True)
            y_b  = y_b.to(DEVICE, non_blocking=True)

            logit = model(p_b, xy_b)
            loss = criterion(logit, y_b)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            scheduler.step()   # OneCycleLR steps per-batch

            B = p_b.shape[0]
            tr_loss_sum += loss.item() * B
            n_tr += B
        train_loss = tr_loss_sum / max(n_tr, 1)

        model.eval()
        val_logits_all, val_labels_all = [], []
        va_loss_sum, n_va = 0.0, 0
        with torch.no_grad():
            for p_b, xy_b, y_b in val_loader:
                p_b  = p_b.to(DEVICE, non_blocking=True)
                xy_b = xy_b.to(DEVICE, non_blocking=True)
                y_b  = y_b.to(DEVICE, non_blocking=True)
                logit = model(p_b, xy_b)
                loss = criterion(logit, y_b)
                B = p_b.shape[0]
                va_loss_sum += loss.item() * B
                n_va += B
                val_logits_all.append(logit.cpu())
                val_labels_all.append(y_b.cpu())
        val_loss = va_loss_sum / max(n_va, 1)
        val_logits = torch.cat(val_logits_all)
        val_labels = torch.cat(val_labels_all)
        val_metrics = compute_binary_metrics(val_logits, val_labels)

        dt = time.time() - t_epoch
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch+1:3d}/{args.epochs}] train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_metrics['accuracy']:.4f}  "
              f"val_prec={val_metrics['precision']:.4f}  val_rec={val_metrics['recall']:.4f}  "
              f"val_f1={val_metrics['f1']:.4f}  val_auc={val_metrics['auc']:.4f}  "
              f"lr={lr_now:.1e}  {dt:.1f}s")

        log.append(dict(
            epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
            val_accuracy=val_metrics["accuracy"], val_precision=val_metrics["precision"],
            val_recall=val_metrics["recall"], val_f1=val_metrics["f1"], val_auc=val_metrics["auc"],
            lr=lr_now, dt=dt,
        ))

        # Select the best checkpoint by val AUC, not accuracy: with an
        # imbalanced positive class, always-predict-negative can score a
        # high accuracy while being useless; AUC is threshold-free and
        # insensitive to the class-balance choice made by (n_min, e_min).
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch + 1
            print(f"[checkpoint] new best val AUC={best_val_auc:.4f} at epoch {best_epoch} -> saving")
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": best_epoch,
                "val_metrics": val_metrics,
                "norm_stats": norm_stats,
                "config": _ckpt_config(args),
            }, ckpt_path)

        if (epoch + 1) % RESUME_CKPT_INTERVAL == 0:
            torch.save({
                "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1, "log": log,
                "best_val_auc": best_val_auc, "best_epoch": best_epoch,
            }, resume_path)

    with open(log_path, "w") as f:
        json.dump({
            "log": log,
            "best_val_auc": best_val_auc,
            "best_epoch": best_epoch,
            "overall_detected_frac": overall_frac,
            "train_detected_frac": train_frac,
            "val_detected_frac": val_frac_d,
            "config": dict(
                batch_size=args.batch_size, epochs=args.epochs, lr=LR, lr_max=args.lr, lr_min=LR_MIN,
                val_frac=VAL_FRAC, seed=SEED, **_ckpt_config(args),
            ),
        }, f, indent=2)
    _plot_curves(log, curves_path)

    if os.path.exists(resume_path):
        os.remove(resume_path)

    # A past bug in this project silently shipped an untrained epoch-1
    # checkpoint (the "best" save never fired again after epoch 1); always
    # print which epoch the FINALLY-SAVED checkpoint on disk actually is, by
    # reading it back rather than trusting the in-memory best_epoch.
    saved = torch.load(ckpt_path, map_location="cpu")
    print(f"[done] best val AUC={best_val_auc:.4f}  "
          f"selected epoch (in-memory)={best_epoch}  "
          f"selected epoch (on-disk checkpoint)={saved['epoch']}  ->  {ckpt_path}")


if __name__ == "__main__":
    main()
