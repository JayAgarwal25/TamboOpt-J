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
    python plots/02_plot_nn_target_vs_pred.py --dual   # dual-species surrogate

Dual mode (--dual): the FNN scatter is rendered PER SPECIES — each per-species
DeepSets model (fnn_electron.pt / fnn_muon.pt) is compared against its own
species-filtered corpus subset (split on the Step-1 species_ids sidecar), since
the corpus E/T ground truth is per-species.
The recon scatter uses the combined DualSpeciesSurrogate on the full corpus,
exactly as 03_train_recon.py does. Outputs:
    FNN_FOLDER/fnn_electron_target_vs_pred.png
    FNN_FOLDER/fnn_muon_target_vs_pred.png
    RECON_FOLDER_deepsets/recon_target_vs_pred.png

Note the recon folder: 03_train_recon_deepsets.py writes to
RECON_FOLDER + "_deepsets", so that is where --dual reads recon.pt and writes
its scatter. Plain RECON_FOLDER belongs to the older flat-MLP 03_train_recon.py.
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
from modules_v6 import run_world
from modules_v6.constants import (
    N_DETECTORS, PRIMARY_DIM
)

# Bound in main() from the resolved run world, never at import.
TRAINING_DATASET_FOLDER = None
FNN_FOLDER = None
RECON_FOLDER = None
from modules_v6.reconstruction import build_recon_from_ckpt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 03_train_recon_deepsets.py writes to RECON_FOLDER + "_deepsets" (its line 50),
# not RECON_FOLDER — that plain folder only exists for the older flat-MLP
# 03_train_recon.py run.
RECON_DEEPSETS_FOLDER = RECON_FOLDER + "_deepsets"

# Seeds match 02_train_fnn.py and 03_train_recon.py
FNN_VAL_SEED   = 0
RECON_VAL_SEED = 1
VAL_FRAC       = 0.10
BATCH          = 1024

# Mirror the log-T transform applied inside 02_train_fnn.py: the FNN is
# trained with log1p(T * T_LOG_SCALE) as its canonical T target, so the
# ground-truth T tensor must be passed through the same transform before the
# FNN scatter is apples-to-apples. Imported from constants so this cannot go
# stale against the trainer.
from modules_v6.constants import T_LOG_SCALE


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


def _scatter(ax, x, y, title: str, vmin=None, vmax=None, lo=None, hi=None):
    """Density-coloured target-vs-prediction panel.

    Hexbin with a log-scale colour normalization so the heavy bulk near
    (0, 0) doesn't wash out the rare high-value tail; `mincnt=1` leaves
    empty bins blank so the y = x reference line stays readable. Pass
    `vmin` / `vmax` (in raw counts) to pin the colour scale across plots."""
    from matplotlib.colors import LogNorm, Normalize
    lo = 0.0 if lo is None else lo
    hi = float(max(x.max(), y.max())) if hi is None else hi
    # norm = LogNorm(vmin=1, vmax=vmax) 
    norm = Normalize(vmin=vmin, vmax=vmax) if (vmin is not None or vmax is not None) else Normalize()
    hb = ax.hexbin(x, y, gridsize=80, cmap="viridis", norm=norm,
                   mincnt=1, extent=(lo, hi, lo, hi))
    plt.colorbar(hb, ax=ax, label="count", pad=0.02, fraction=0.046)
    ax.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=2.0,
            alpha=0.85, label="y = x")
    ax.set_xlabel("target"); ax.set_ylabel("prediction")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, framealpha=1)


def load_fnn() -> FNNSurrogate:
    # Read width + dropout from the saved config and prefer the FNN's own
    # norm_stats (02_train_fnn.py updates the T slots in-memory for log-T
    # training and ships the modified stats inside fnn.pt; disk norm_stats.pt
    # still holds raw-T values).
    fnn_ckpt = torch.load(os.path.join(FNN_FOLDER, "fnn.pt"), map_location=DEVICE)
    cfg = fnn_ckpt.get("config", {})
    fnn = FNNSurrogate(
        n_det=N_DETECTORS, primary_dim=PRIMARY_DIM,
        hidden=int(cfg.get("hidden", 512)),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(DEVICE)
    fnn.load_state_dict(fnn_ckpt["state_dict"])
    norm_stats = fnn_ckpt.get(
        "norm_stats",
        torch.load(os.path.join(TRAINING_DATASET_FOLDER, "norm_stats.pt")),
    )
    fnn.set_normalization(norm_stats)
    fnn.eval()
    print(f"[load] fnn.pt  epoch={fnn_ckpt.get('epoch','?')}  "
          f"val={fnn_ckpt.get('val_total','?')}  "
          f"hidden={int(cfg.get('hidden', 512))} "
          f"lbfgs_iter={fnn_ckpt.get('lbfgs_iter','?')}")
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


def load_recon(folder: str = RECON_DEEPSETS_FOLDER):
    """Mirror of load_fnn() for the recon checkpoint. Used by the standalone
    CLI path; training scripts pass an already-trained recon in.

    Dispatches on the checkpoint's own config["model_type"] via
    build_recon_from_ckpt, so flat-MLP ("mlp") and DeepSets ("deepsets")
    checkpoints both load. Hardcoding Reconstruction here predated
    03_train_recon_deepsets.py and died on a state_dict shape mismatch
    against its checkpoints."""
    recon_ckpt = torch.load(os.path.join(folder, "recon.pt"),
                            map_location=DEVICE, weights_only=False)
    recon = build_recon_from_ckpt(recon_ckpt, N_DETECTORS, DEVICE)
    print(f"[load] {folder}/recon.pt  "
          f"model={recon_ckpt.get('config', {}).get('model_type', 'mlp')}  "
          f"epoch={recon_ckpt.get('epoch','?')}  "
          f"val={recon_ckpt.get('val_total','?')} "
          f"lbfgs_iter={recon_ckpt.get('lbfgs_iter','?')}")
    return recon


# --------------------------------------------------------------------------- #
# Dual-species loading. Mirrors 02_train_fnn_deepsets.py (per-species FNN) and
# 03_train_recon.py (combined dual surrogate for recon).
# --------------------------------------------------------------------------- #
SPECIES_TAGS = (("electron", 0), ("muon", 1))   # (tag, species id: 0=electron, 1=muon)


def load_species_fnn(species: str):
    """Load one per-species surrogate (fnn_electron.pt / fnn_muon.pt) from
    FNN_FOLDER. Uses build_surrogate_from_ckpt so flat-MLP or DeepSets configs
    both work, with the checkpoint's own per-species norm stats applied."""
    from modules_v6.deepsets_surrogate import build_surrogate_from_ckpt
    path = os.path.join(FNN_FOLDER, f"fnn_{species}.pt")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    fnn = build_surrogate_from_ckpt(ckpt, N_DETECTORS, PRIMARY_DIM, DEVICE)
    cfg = ckpt.get("config", {})
    print(f"[load] fnn_{species}.pt  model={cfg.get('model_type', 'fnn')}  "
          f"epoch={ckpt.get('epoch', '?')}  val={ckpt.get('val_total', '?')}")
    return fnn


def _render_fnn_scatter(fnn, primary, xy, E_true, T_true, val_idx, output_path,
                        vmin_E=10, vmax_E=4000, vmin_T=10, vmax_T=2500):
    """Pure rendering — no I/O for models or corpus. Caller supplies a loaded
    FNN in eval mode plus the in-memory tensors. T_true must already be
    log1p(T*1e8)-transformed (matching what the FNN was trained against).
    vmin/vmax_{E,T} pin each panel's hexbin colour (count) scale (per species)."""
    p   = primary[val_idx]
    x   = xy[val_idx]
    E_t = E_true[val_idx]
    T_t = T_true[val_idx]
    E_p, T_p = fnn_predict(fnn, p, x)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    _scatter(axes[0], E_t.flatten().numpy(), E_p.flatten().numpy(),
             f"FNN  log1p(E)  (N={E_t.numel():,} detector-samples)",
             vmin=vmin_E, vmax=vmax_E)
    _scatter(axes[1], T_t.flatten().numpy(), T_p.flatten().numpy(),
             f"FNN  log1p(T·1e8)  (N={T_t.numel():,} detector-samples)",
             vmin=vmin_T, vmax=vmax_T)
    fig.suptitle("FNN target vs prediction — val split", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    print(f"[save] {output_path}")


def _render_recon_scatter(fnn, recon, primary, xy, val_idx, output_path):
    """Pure rendering — caller supplies both nets (eval mode) and the
    in-memory primary/xy tensors. Recon target is `primary[val_idx, :4]`."""
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
            feats = torch.stack([xy_b[..., 0], xy_b[..., 1], E_b, T_b], dim=-1)  # (B, n_det, 4)
            pred[lo:hi] = recon(feats).cpu()                                     # DeepSets recon takes (B, n_det, 4)

    labels = ("dir_x", "dir_y", "dir_z", "log_e_norm")
    vmin_s = (1, 1, 1, 1)
    # vmax_s = (100, 100, 100, 200)
    vmax_s = (80, 200, 200, 200)
    # vmax_s = (200, 300, 300, 500)
    hi = (1, 1, 0.5, 1)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    for i, name in enumerate(labels):
        _scatter(axes[i], target[:, i].numpy(), pred[:, i].numpy(), f"Recon  {name}", vmin=vmin_s[i], vmax=vmax_s[i], hi=hi[i]) # TODO
    fig.suptitle(f"Recon target vs prediction — val split  (N={N:,})", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    print(f"[save] {output_path}")


def _load_corpus():
    """Load shared tensors + strategy ids. Applies log1p(T*1e8) so T_true
    matches the FNN's training target space (see 02_train_fnn.py).

    Only used by the standalone CLI / when training scripts call into the
    plotters without providing their already-loaded tensors."""
    primary   = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()
    xy        = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "xy.pt")).float()
    E_true    = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "E.pt")).float()
    T_true    = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "T.pt")).float()
    strat_ids = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "strategy_ids.pt")).long()
    T_true = torch.log1p(T_true * T_LOG_SCALE)
    return primary, xy, E_true, T_true, strat_ids


def plot_fnn_only(*, fnn=None,
                  primary=None, xy=None,
                  E_true=None, T_true=None,
                  val_idx=None,
                  species=None,
                  output_path=None):
    """Render fnn_target_vs_pred.png. Every argument is optional: anything
    left as None gets loaded from disk so the standalone CLI still works.

    Training-script callers (02_train_fnn.py) pass everything they already
    have in memory — fnn (with best weights reloaded), primary, xy, E_all,
    T_all (already log1p-transformed in 02), val_idx — and no disk I/O for
    the corpus is performed. T_true MUST be in log-T space if provided.

    `species` ("electron"/"muon") selects the per-species hexbin colour
    scale from FNN_DUAL_VLIM; left None it falls back to the generic
    _render_fnn_scatter defaults.
    """
    if primary is None or xy is None or E_true is None or T_true is None:
        primary, xy, E_true, T_true, strat_ids_disk = _load_corpus()
    else:
        strat_ids_disk = None
    if val_idx is None:
        if strat_ids_disk is None:
            strat_ids_disk = torch.load(
                os.path.join(TRAINING_DATASET_FOLDER, "strategy_ids.pt")
            ).long()
        val_idx = shower_level_val_idx(strat_ids_disk, VAL_FRAC, FNN_VAL_SEED)
    if fnn is None:
        fnn = load_fnn()
    if output_path is None:
        os.makedirs(FNN_FOLDER, exist_ok=True)
        output_path = os.path.join(FNN_FOLDER, "fnn_target_vs_pred.png")
    _render_fnn_scatter(fnn, primary, xy, E_true, T_true, val_idx, output_path,
                        **FNN_DUAL_VLIM.get(species, {}))


def plot_recon_only(*, fnn=None, recon=None,
                    primary=None, xy=None,
                    val_idx=None,
                    output_path=None,
                    recon_folder=RECON_DEEPSETS_FOLDER):
    """Render recon_target_vs_pred.png. Like `plot_fnn_only`, every argument
    is optional. Training-script callers (03_train_recon.py) pass fnn +
    recon (best weights reloaded) + primary + xy + val_idx; no disk I/O for
    those is then performed.

    `recon_folder` is where a None `recon` is loaded from and where a None
    `output_path` lands — keep the checkpoint and its scatter in the same
    run folder."""
    if primary is None or xy is None:
        primary, xy, _E, _T, strat_ids_disk = _load_corpus()
    else:
        strat_ids_disk = None
    if val_idx is None:
        if strat_ids_disk is None:
            strat_ids_disk = torch.load(
                os.path.join(TRAINING_DATASET_FOLDER, "strategy_ids.pt")
            ).long()
        val_idx = shower_level_val_idx(strat_ids_disk, VAL_FRAC, RECON_VAL_SEED)
    if fnn is None:
        fnn = load_fnn()
    if recon is None:
        recon = load_recon(recon_folder)
    if output_path is None:
        os.makedirs(recon_folder, exist_ok=True)
        output_path = os.path.join(recon_folder, "recon_target_vs_pred.png")
    _render_recon_scatter(fnn, recon, primary, xy, val_idx, output_path)


# Per-species hexbin colour (count) limits for the dual FNN scatters:
# (vmin_E, vmax_E, vmin_T, vmax_T). Muon signals are denser than electron.
FNN_DUAL_VLIM = {
    "electron": dict(vmin_E=0, vmax_E=2000,  vmin_T=0, vmax_T=2000),
    "muon":     dict(vmin_E=0, vmax_E=3000, vmin_T=0, vmax_T=1000),
}


def plot_fnn_dual(output_dir=None):
    """Per-species FNN scatter for the dual-species surrogate. Each species'
    DeepSets model is evaluated against its OWN species subset (split on the
    Step-1 species_ids sidecar, the corpus E/T being per-species), reproducing
    02_train_fnn_deepsets.py's split. Writes
    FNN_FOLDER/fnn_<species>_target_vs_pred.png per species. Per-species colour
    scales come from FNN_DUAL_VLIM."""
    primary, xy, E_true, T_true, strat_ids = _load_corpus()
    species_ids = torch.load(
        os.path.join(TRAINING_DATASET_FOLDER, "species_ids.pt")).long()
    if output_dir is None:
        output_dir = FNN_FOLDER
    os.makedirs(output_dir, exist_ok=True)

    for tag, species_val in SPECIES_TAGS:
        idx = torch.nonzero(species_ids == species_val).squeeze(-1)
        if idx.numel() == 0:
            print(f"[skip] no {tag} rows (species id {species_val}) in corpus")
            continue
        fnn = load_species_fnn(tag)
        # val_idx is positional within the filtered subset; pass the subset
        # tensors so _render_fnn_scatter indexes them consistently.
        val_idx = shower_level_val_idx(strat_ids[idx], VAL_FRAC, FNN_VAL_SEED)
        out = os.path.join(output_dir, f"fnn_{tag}_target_vs_pred.png")
        _render_fnn_scatter(fnn, primary[idx], xy[idx],
                            E_true[idx], T_true[idx], val_idx, out,
                            **FNN_DUAL_VLIM[tag])


def plot_recon_dual(output_path=None, recon_folder=RECON_DEEPSETS_FOLDER):
    """Recon scatter for the dual-species surrogate. The combined
    DualSpeciesSurrogate (fnn_electron.pt + fnn_muon.pt) feeds the recon on the
    FULL corpus — identical to 03_train_recon.py. recon.pt itself is a single
    (non-per-species) net, so load_recon() is reused unchanged."""
    from modules_v6.dual_surrogate import load_dual_surrogate
    dual = load_dual_surrogate(FNN_FOLDER, DEVICE)
    plot_recon_only(fnn=dual, output_path=output_path,
                    recon_folder=recon_folder)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dual", action="store_true",
                    help="dual-species surrogate: per-species FNN scatters + "
                         "combined-surrogate recon scatter")
    run_world.add_run_world_args(ap)
    args = ap.parse_args()

    global TRAINING_DATASET_FOLDER, FNN_FOLDER, RECON_FOLDER
    W = run_world.resolve(args, need_write=False)
    TRAINING_DATASET_FOLDER = W.dataset_folder
    FNN_FOLDER              = W.fnn_folder
    RECON_FOLDER            = W.recon_folder

    print("=" * 72)
    print("v6/plots/02_plot_nn_target_vs_pred.py" + ("  [dual]" if args.dual else ""))
    print("=" * 72)
    if args.dual:
        plot_fnn_dual()
        plot_recon_dual()
    else:
        plot_fnn_only()
        plot_recon_only()


if __name__ == "__main__":
    main()
