#!/usr/bin/env python3
"""
Fired-only validation check: how good is the base electron/muon FNN
specifically on detectors that actually fired, relative to the fired-only
aleatoric ceiling (max_R2.total_fired_vs_fired=0.399), rather than the
optimistic all-detector ceiling (0.551) that blends in the ~57% of trivially-
easy non-fired detectors.

Reproduces the base-FNN training script's own val split (shower_level_split,
VAL_FRAC=0.10, SEED=0) so this is genuine held-out data, not train data.
"""
import os, sys, json
import torch

_V6 = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v6"
sys.path.insert(0, _V6)
import modules_v6  # noqa: F401
sys.path.insert(0, _V6)  # 02_ scripts are importable by path (filename starts with a digit)
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("train_fnn_deepsets", os.path.join(_V6, "02_train_fnn_deepsets.py"))
train_mod = _ilu.module_from_spec(_spec)
# Avoid running main() on import.
_orig_name = train_mod.__name__
_spec.loader.exec_module(train_mod)

from modules_v6.constants import TRAINING_DATASET_FOLDER, FNN_FOLDER
from modules_v6.deepsets_surrogate import DeepSetsSurrogate

DEVICE = torch.device("cpu")
VAL_FRAC = 0.10
SEED = 0
FIRE_EPS = 1.0e-3
RUN_BASE_FNN = f"{FNN_FOLDER}"  # test_v6_run_02_recentered (base, pre-adaptive-loop)

floor = json.load(open(os.path.join(_V6, "aleatoric_floor_sigma200.json")))
std_fired_E = floor["corpus_std_fired"]["E"]
std_fired_T = floor["corpus_std_fired"]["T"]
floor_fired_vs_fired_total = floor["floor_zmse"]["total_fired_vs_fired"]
r2max_fired = floor["max_R2"]["total_fired_vs_fired"]
print(f"[floor] corpus_std_fired: E={std_fired_E:.4f} T={std_fired_T:.4f}")
print(f"[floor] floor_zmse.total_fired_vs_fired={floor_fired_vs_fired_total:.4f}  "
      f"max_R2.total_fired_vs_fired={r2max_fired:.4f}")

print("\n[load] corpus ...")
primary   = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()
xy        = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "xy.pt")).float()
E_all     = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "E.pt")).float()   # already log1p(E)
T_all_raw = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "T.pt")).float()   # raw T, not yet log1p
strat_ids = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "strategy_ids.pt")).long()
species_ids = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "species_ids.pt")).long()
print(f"[load] primary={tuple(primary.shape)}  E={tuple(E_all.shape)}")

# compute_aleatoric_floor.py pools BOTH species together before computing
# corpus_std_fired / floor_zmse.total_fired_vs_fired (E.pt/T.pt hold electron+muon
# rows combined, undifferentiated). Comparing a single species' achieved R²
# against that pooled ceiling is an apples-to-oranges mismatch -- pool fired
# errors from both species' models before computing the final z-mse/R², exactly
# mirroring the floor script's own convention, so the two numbers are on the
# same footing.
pooled_err_E_fired, pooled_err_T_fired = [], []

for tag, species_val, T_LOG_SCALE in (("electron", 0, train_mod.T_LOG_SCALE),
                                       ("muon", 1, train_mod.T_LOG_SCALE)):
    idx = torch.nonzero(species_ids == species_val).squeeze(-1)
    p_s, xy_s, E_s = primary[idx], xy[idx], E_all[idx]
    T_s = torch.log1p(T_all_raw[idx] * T_LOG_SCALE)
    strat_s = strat_ids[idx]

    train_idx, val_idx = train_mod.shower_level_split(strat_s, VAL_FRAC, SEED)
    print(f"\n[{tag}] species rows={p_s.shape[0]}  val pairs={len(val_idx)}")

    ckpt_path = os.path.join(RUN_BASE_FNN, f"fnn_{tag}.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model = DeepSetsSurrogate(
        n_det=cfg["n_det"], primary_dim=cfg["primary_dim"],
        hidden=cfg["hidden"], context=cfg["context"],
        n_enc=cfg["n_enc"], n_dec=cfg["n_dec"],
        dropout=cfg.get("dropout", 0.0), pool=cfg.get("pool", "mean"),
    ).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[{tag}] loaded {ckpt_path}  epoch={ckpt['epoch']}  reported val_total={ckpt['val_total']:.4f}")

    p_val, xy_val = p_s[val_idx].to(DEVICE), xy_s[val_idx].to(DEVICE)
    E_val, T_val = E_s[val_idx].to(DEVICE), T_s[val_idx].to(DEVICE)

    preds_E, preds_T = [], []
    CHUNK = 4096
    with torch.no_grad():
        for start in range(0, p_val.shape[0], CHUNK):
            end = start + CHUNK
            pred = model(p_val[start:end], xy_val[start:end])   # (B, n_det, 2), raw log1p space
            preds_E.append(pred[..., 0])
            preds_T.append(pred[..., 1])
    pred_E = torch.cat(preds_E, dim=0)
    pred_T = torch.cat(preds_T, dim=0)

    fired = E_val > FIRE_EPS
    n_fired = int(fired.sum())
    n_total = fired.numel()
    print(f"[{tag}] val (primary,detector) pairs={n_total}  fired={n_fired} ({100*n_fired/n_total:.1f}%)")

    err_E = (pred_E - E_val) ** 2
    err_T = (pred_T - T_val) ** 2

    # Per-species number, for reference only -- NOT directly comparable to the
    # pooled ceiling (kept to show it disagrees with the pooled result below).
    zmse_E_fired_species = float(err_E[fired].mean()) / (std_fired_E ** 2)
    zmse_T_fired_species = float(err_T[fired].mean()) / (std_fired_T ** 2)
    zmse_total_fired_species = 0.5 * (zmse_E_fired_species + zmse_T_fired_species)
    print(f"[{tag}] PER-SPECIES fired z-mse(E,T,total) = "
          f"({zmse_E_fired_species:.4f}, {zmse_T_fired_species:.4f}, {zmse_total_fired_species:.4f})  "
          f"achieved R²={1-zmse_total_fired_species:.4f}  "
          f"[reference only -- not comparable to the pooled ceiling on its own]")

    pooled_err_E_fired.append(err_E[fired])
    pooled_err_T_fired.append(err_T[fired])

    # All-detector blended z-mse using this same run, for a sanity cross-check
    # against the checkpoint's own reported val_total.
    corpus_std_E, corpus_std_T = floor["corpus_std"]["E"], floor["corpus_std"]["T"]
    zmse_E_all = float(err_E.mean()) / (corpus_std_E ** 2)
    zmse_T_all = float(err_T.mean()) / (corpus_std_T ** 2)
    zmse_total_all = 0.5 * (zmse_E_all + zmse_T_all)
    print(f"[{tag}] all-detector  z-mse(E,T,total) = "
          f"({zmse_E_all:.4f}, {zmse_T_all:.4f}, {zmse_total_all:.4f})   "
          f"[cross-check vs ckpt's own out_std-normalized val_total={ckpt['val_total']:.4f} -- "
          f"expect close but not identical, different normalization std source]")

# ── Pooled (electron+muon combined) fired R², matching the floor script's own
# pooling convention exactly -- THIS is the number comparable to R²max_fired.
pooled_E = torch.cat(pooled_err_E_fired)
pooled_T = torch.cat(pooled_err_T_fired)
zmse_E_pooled = float(pooled_E.mean()) / (std_fired_E ** 2)
zmse_T_pooled = float(pooled_T.mean()) / (std_fired_T ** 2)
zmse_total_pooled = 0.5 * (zmse_E_pooled + zmse_T_pooled)
r2_achieved_pooled = 1 - zmse_total_pooled
frac_of_ceiling_pooled = r2_achieved_pooled / r2max_fired

print("\n" + "=" * 70)
print("POOLED (electron+muon combined) fired-only result")
print("=" * 70)
print(f"n_fired pooled: E={pooled_E.numel()}  T={pooled_T.numel()}")
print(f"pooled fired z-mse(E,T,total) = ({zmse_E_pooled:.4f}, {zmse_T_pooled:.4f}, {zmse_total_pooled:.4f})")
print(f"pooled fired-only achieved R² = {r2_achieved_pooled:.4f}")
print(f"fired ceiling R²max           = {r2max_fired:.4f}")
print(f"fraction of fired ceiling reached = {100*frac_of_ceiling_pooled:.1f}%")
