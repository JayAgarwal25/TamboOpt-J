"""Target vs prediction scatter for the trained FNN and recon nets.

Loads the cached corpus (primary / xy / E / T / strategy_ids) and the two
frozen checkpoints (fnn.pt, recon.pt), evaluates each on its respective
shower-level validation split, and saves scatter plots of target vs
prediction with a 1:1 reference line.

FNN plot        : flattened (E, T) over all detectors in the val split.
Recon plot      : raw primary encoding (dir_x, dir_y, dir_z, log_e_norm) over
                  the val split. The recon runs on FNN-predicted (E, T) rather
                  than ground truth, so the scatter reflects the end-to-end
                  FNN -> recon error.

Artifacts:
    outputs/fnn_target_vs_pred.png
    outputs/recon_target_vs_pred.png

Run from the v6 folder:

    cd TambOpt/detector_optimization_v6
    python plots/02_plot_nn_target_vs_pred.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_V6_DIR = os.path.dirname(_HERE)
if _V6_DIR not in sys.path:
    sys.path.insert(0, _V6_DIR)

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import modules_v6  # noqa: F401 — triggers sys.path injection for v3 + v4
from modules_v6.fnn_surrogate import FNNSurrogate
from modules_v6.constants import (
    TRAINING_DATASET_FOLDER, FNN_FOLDER, RECON_FOLDER,
    N_DETECTORS, PRIMARY_DIM,
)
from modules_v6.reconstruction import Reconstruction


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seeds match 02_train_fnn.py and 03_train_recon.py
FNN_VAL_SEED   = 0
RECON_VAL_SEED = 1
VAL_FRAC       = 0.10
BATCH          = 1024


def shower_level_val_idx(strategy_ids: torch.Tensor,
                         val_frac: float,
                         seed: int) -> torch.Tensor:
    """Reproduce the shower-level val indices used during training."""
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
    val_mask = is_val[shower_of_pair]
    return torch.nonzero(val_mask).squeeze(-1)


def _scatter(ax, x, y, title: str):
    ax.scatter(x, y, s=2, alpha=0.25, linewidths=0)
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.6, label="y = x")
    ax.set_xlabel("target")
    ax.set_ylabel("prediction")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def load_fnn() -> FNNSurrogate:
    fnn_ckpt = torch.load(os.path.join(FNN_FOLDER, "fnn.pt"), map_location=DEVICE)
    norm_stats = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "norm_stats.pt"))
    fnn = FNNSurrogate(n_det=N_DETECTORS, primary_dim=PRIMARY_DIM,
                       hidden=512, dropout=0.1).to(DEVICE)
    fnn.load_state_dict(fnn_ckpt["state_dict"])
    fnn.set_normalization(norm_stats)
    fnn.eval()
    print(f"[load] fnn.pt  epoch={fnn_ckpt.get('epoch','?')}  "
          f"val={fnn_ckpt.get('val_total','?')}")
    return fnn


@torch.no_grad()
def fnn_predict(fnn: FNNSurrogate,
                primary: torch.Tensor,
                xy: torch.Tensor):
    N = primary.shape[0]
    E_pred = torch.empty((N, N_DETECTORS), dtype=torch.float32)
    T_pred = torch.empty((N, N_DETECTORS), dtype=torch.float32)
    for lo in range(0, N, BATCH):
        hi = min(lo + BATCH, N)
        pred = fnn(primary[lo:hi].to(DEVICE), xy[lo:hi].to(DEVICE))
        E_pred[lo:hi] = pred[..., 0].cpu()
        T_pred[lo:hi] = pred[..., 1].cpu()
    return E_pred, T_pred


def plot_fnn(primary, xy, E_true, T_true, val_idx, output_path):
    fnn = load_fnn()
    p = primary[val_idx]
    x = xy[val_idx]
    E_t = E_true[val_idx]
    T_t = T_true[val_idx]
    E_p, T_p = fnn_predict(fnn, p, x)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    _scatter(axes[0], E_t.flatten().numpy(), E_p.flatten().numpy(),
             f"FNN  E  (N={E_t.numel():,} detector-samples)")
    _scatter(axes[1], T_t.flatten().numpy(), T_p.flatten().numpy(),
             f"FNN  T  (N={T_t.numel():,} detector-samples)")
    fig.suptitle("FNN target vs prediction — val split", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    print(f"[save] {output_path}")


def plot_recon(primary, xy, val_idx, output_path, fnn: FNNSurrogate):
    recon_ckpt = torch.load(os.path.join(RECON_FOLDER, "recon.pt"), map_location=DEVICE)
    cfg = recon_ckpt.get("config", {})
    recon = Reconstruction(
        n_det=int(recon_ckpt["num_detectors"]),
        input_features=int(recon_ckpt["input_features"]),
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
    print(f"[load] recon.pt  epoch={recon_ckpt.get('epoch','?')}  "
          f"val={recon_ckpt.get('val_total','?')}")

    p = primary[val_idx]
    x = xy[val_idx]

    # Recon sees FNN predictions, not ground-truth (E, T) — same as 04_optimize.
    E_pred, T_pred = fnn_predict(fnn, p, x)

    # Target = v6 primary encoding [dir_x, dir_y, dir_z, log_e_norm] in raw units.
    target = p[:, :4].float()

    N = p.shape[0]
    pred = torch.empty((N, 4), dtype=torch.float32)
    with torch.no_grad():
        for lo in range(0, N, BATCH):
            hi = min(lo + BATCH, N)
            xy_b = x[lo:hi].to(DEVICE)
            E_b  = E_pred[lo:hi].to(DEVICE)
            T_b  = T_pred[lo:hi].to(DEVICE)
            feats = torch.stack([xy_b[..., 0], xy_b[..., 1], E_b, T_b], dim=-1)
            flat  = feats.reshape(feats.shape[0], -1)
            pred[lo:hi] = recon(flat).cpu()

    labels = ("dir_x", "dir_y", "dir_z", "log_e_norm")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    for i, name in enumerate(labels):
        _scatter(axes[i], target[:, i].numpy(), pred[:, i].numpy(), f"Recon  {name}")
    fig.suptitle(f"Recon target vs prediction — val split  (N={N:,})", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    print(f"[save] {output_path}")


def main():
    print("=" * 72)
    print("v6/plots/02_plot_nn_target_vs_pred.py")
    print("=" * 72)

    primary   = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()
    xy        = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "xy.pt")).float()
    E_true    = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "E.pt")).float()
    T_true    = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "T.pt")).float()
    strat_ids = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "strategy_ids.pt")).long()
    print(f"[load] primary={tuple(primary.shape)}  xy={tuple(xy.shape)}")

    fnn_val_idx   = shower_level_val_idx(strat_ids, VAL_FRAC, FNN_VAL_SEED)
    recon_val_idx = shower_level_val_idx(strat_ids, VAL_FRAC, RECON_VAL_SEED)
    print(f"[split] fnn val pairs={len(fnn_val_idx):,}  "
          f"recon val pairs={len(recon_val_idx):,}")

    out_dir = os.path.join(_V6_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    plot_fnn(
        primary, xy, E_true, T_true, fnn_val_idx,
        os.path.join(out_dir, "fnn_target_vs_pred.png"),
    )

    fnn = load_fnn()
    plot_recon(
        primary, xy, recon_val_idx,
        os.path.join(out_dir, "recon_target_vs_pred.png"),
        fnn,
    )


if __name__ == "__main__":
    main()
