#!/usr/bin/env python3
"""
Investigating why recon_r1 (trained on FNN_r1's predictions) converged to a
notably worse val (0.0040) than the original recon (trained on the base
FNN's predictions, val=0.0011), despite FNN_r1's own reported val loss being
BETTER (0.343/0.371 vs 0.408/0.428) and near-identical aggregate prediction
statistics (mean/std) on the base corpus.

This directly compares base FNN vs FNN_r1 on the SAME base-only held-out
split (not the differently-split numbers from each model's own training
log), giving a rigorous, controlled z-mse comparison. Also breaks results
down per strategy (of the 7 base layout strategies) to see whether any
degradation is uniform or concentrated.
"""
import os, sys, json
import torch
import numpy as np

_V6 = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jagarwal/TambOpt-zlt/detector_optimization_v6"
sys.path.insert(0, _V6)
import modules_v6  # noqa: F401
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("train_fnn_deepsets", os.path.join(_V6, "02_train_fnn_deepsets.py"))
train_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(train_mod)

from modules_v6.constants import TRAINING_DATASET_FOLDER, FNN_FOLDER
from modules_v6.deepsets_surrogate import DeepSetsSurrogate

DEVICE = torch.device("cpu")
VAL_FRAC = 0.10
SEED = 0
FIRE_EPS = 1.0e-3
BASE_FNN_DIR = FNN_FOLDER                    # test_v6_run_02_recentered
R1_FNN_DIR   = FNN_FOLDER + "_r1"            # test_v6_run_02_recentered_r1

floor = json.load(open(os.path.join(_V6, "aleatoric_floor_sigma200.json")))
corpus_std_E, corpus_std_T = floor["corpus_std"]["E"], floor["corpus_std"]["T"]
print(f"[norm] using floor's corpus_std for BOTH models (consistent, direct comparison): "
      f"E={corpus_std_E:.4f} T={corpus_std_T:.4f}")

print("\n[load] corpus ...")
primary   = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()
xy        = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "xy.pt")).float()
E_all     = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "E.pt")).float()   # already log1p(E)
T_all_raw = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "T.pt")).float()   # raw T, not yet log1p
strat_ids = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "strategy_ids.pt")).long()
species_ids = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "species_ids.pt")).long()
print(f"[load] primary={tuple(primary.shape)}  E={tuple(E_all.shape)}")

strategy_names = [s[0] for s in train_mod.__dict__.get("_STRATEGIES", [])] if hasattr(train_mod, "_STRATEGIES") else None


def load_deepsets(ckpt_path):
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
    return model, ckpt


@torch.no_grad()
def predict(model, p, xy_, chunk=4096):
    preds_E, preds_T = [], []
    for start in range(0, p.shape[0], chunk):
        end = start + chunk
        pred = model(p[start:end], xy_[start:end])
        preds_E.append(pred[..., 0])
        preds_T.append(pred[..., 1])
    return torch.cat(preds_E, dim=0), torch.cat(preds_T, dim=0)


overall = {}
for tag, species_val, T_LOG_SCALE in (("electron", 0, train_mod.T_LOG_SCALE),
                                       ("muon", 1, train_mod.T_LOG_SCALE)):
    idx = torch.nonzero(species_ids == species_val).squeeze(-1)
    p_s, xy_s, E_s = primary[idx], xy[idx], E_all[idx]
    T_s = torch.log1p(T_all_raw[idx] * T_LOG_SCALE)
    strat_s = strat_ids[idx]

    train_idx, val_idx = train_mod.shower_level_split(strat_s, VAL_FRAC, SEED)
    print(f"\n[{tag}] species rows={p_s.shape[0]}  val pairs={len(val_idx)}")

    p_val, xy_val = p_s[val_idx].to(DEVICE), xy_s[val_idx].to(DEVICE)
    E_val, T_val = E_s[val_idx].to(DEVICE), T_s[val_idx].to(DEVICE)
    strat_val = strat_s[val_idx]

    model_base, ckpt_base = load_deepsets(os.path.join(BASE_FNN_DIR, f"fnn_{tag}.pt"))
    model_r1, ckpt_r1     = load_deepsets(os.path.join(R1_FNN_DIR, f"fnn_{tag}.pt"))
    print(f"[{tag}] base ckpt val_total(own split)={ckpt_base['val_total']:.4f}   "
          f"r1 ckpt val_total(own split)={ckpt_r1['val_total']:.4f}")

    predE_base, predT_base = predict(model_base, p_val, xy_val)
    predE_r1,   predT_r1   = predict(model_r1, p_val, xy_val)

    def zmse(predE, predT):
        errE = (predE - E_val) ** 2
        errT = (predT - T_val) ** 2
        zE = float(errE.mean()) / (corpus_std_E ** 2)
        zT = float(errT.mean()) / (corpus_std_T ** 2)
        return 0.5 * (zE + zT), zE, zT, errE, errT

    z_base, zE_base, zT_base, errE_base, errT_base = zmse(predE_base, predT_base)
    z_r1,   zE_r1,   zT_r1,   errE_r1,   errT_r1   = zmse(predE_r1, predT_r1)

    print(f"[{tag}] SAME base-only held-out split, SAME normalizer:")
    print(f"[{tag}]   base FNN  z-mse(E,T,total) = ({zE_base:.4f}, {zT_base:.4f}, {z_base:.4f})")
    print(f"[{tag}]   r1   FNN  z-mse(E,T,total) = ({zE_r1:.4f}, {zT_r1:.4f}, {z_r1:.4f})")
    print(f"[{tag}]   change: {100*(z_r1 - z_base)/z_base:+.1f}%  (positive = r1 is WORSE on base-only data)")

    # Per-strategy breakdown.
    print(f"[{tag}] per-strategy z-mse (total), base -> r1:")
    strat_report = {}
    for s_id in sorted(torch.unique(strat_val).tolist()):
        mask = strat_val == s_id
        zb = 0.5 * (float(errE_base[mask].mean()) / corpus_std_E**2 + float(errT_base[mask].mean()) / corpus_std_T**2)
        zr = 0.5 * (float(errE_r1[mask].mean())   / corpus_std_E**2 + float(errT_r1[mask].mean())   / corpus_std_T**2)
        name = strategy_names[s_id] if strategy_names and s_id < len(strategy_names) else f"strategy_{s_id}"
        strat_report[name] = dict(base=zb, r1=zr, pct_change=100*(zr-zb)/zb)
        print(f"[{tag}]   {name:20s} (n={int(mask.sum()):>6}): base={zb:.4f}  r1={zr:.4f}  ({100*(zr-zb)/zb:+.1f}%)")

    # ── Fired vs non-fired breakdown ────────────────────────────────────────
    # Hypothesis: FNN_r1's blended average improved, but maybe that's driven by
    # the trivially-easy non-fired detectors getting even easier, while the
    # informative FIRED detectors (the ones recon actually needs) got worse.
    fired = E_val > FIRE_EPS
    print(f"[{tag}] fired fraction: {float(fired.float().mean()):.3f}")
    fired_report = {}
    for label, mask in (("fired", fired), ("non_fired", ~fired)):
        zb = 0.5 * (float(errE_base[mask].mean()) / corpus_std_E**2 + float(errT_base[mask].mean()) / corpus_std_T**2)
        zr = 0.5 * (float(errE_r1[mask].mean())   / corpus_std_E**2 + float(errT_r1[mask].mean())   / corpus_std_T**2)
        fired_report[label] = dict(base=zb, r1=zr, pct_change=100*(zr-zb)/zb, n=int(mask.sum()))
        print(f"[{tag}]   {label:10s} (n={int(mask.sum()):>8}): base={zb:.4f}  r1={zr:.4f}  "
              f"({100*(zr-zb)/zb:+.1f}%, positive = r1 WORSE)")

    # ── Within-event cross-detector pattern check ───────────────────────────
    # Hypothesis: FNN_r1 might be just as accurate on average per detector,
    # but if it flattens out the RELATIVE differences BETWEEN the 100
    # detectors within a single event (compared to base), that would hurt
    # recon (which needs the cross-detector pattern to triangulate direction)
    # without showing up much in a plain per-example MSE.
    def within_event_stats(predE):
        # predE: (n_val, 100) -> per-row (per-event) std across detectors,
        # and per-row correlation with the true E pattern for that event.
        pred_std = predE.std(dim=1)                                     # (n_val,)
        true_std = E_val.std(dim=1)                                     # (n_val,)
        pred_c = predE - predE.mean(dim=1, keepdim=True)
        true_c = E_val - E_val.mean(dim=1, keepdim=True)
        denom = (pred_c.norm(dim=1) * true_c.norm(dim=1)).clamp(min=1e-8)
        corr = (pred_c * true_c).sum(dim=1) / denom                     # (n_val,) Pearson r per event
        return float(pred_std.mean()), float(true_std.mean()), float(corr.mean())

    ps_base, ts_base, corr_base = within_event_stats(predE_base)
    ps_r1,   ts_r1,   corr_r1   = within_event_stats(predE_r1)
    print(f"[{tag}] within-event cross-detector E pattern:")
    print(f"[{tag}]   true  within-event std (avg over events) = {ts_base:.4f}")
    print(f"[{tag}]   base  pred  within-event std             = {ps_base:.4f}  "
          f"(pred/true ratio={ps_base/ts_base:.3f})")
    print(f"[{tag}]   r1    pred  within-event std             = {ps_r1:.4f}  "
          f"(pred/true ratio={ps_r1/ts_r1:.3f})")
    print(f"[{tag}]   base  mean per-event corr(pred,true) across 100 detectors = {corr_base:.4f}")
    print(f"[{tag}]   r1    mean per-event corr(pred,true) across 100 detectors = {corr_r1:.4f}")

    overall[tag] = dict(
        base_zmse_total=z_base, r1_zmse_total=z_r1,
        pct_change=100*(z_r1-z_base)/z_base,
        per_strategy=strat_report,
        fired_breakdown=fired_report,
        within_event=dict(
            true_std=ts_base, base_pred_std=ps_base, r1_pred_std=ps_r1,
            base_corr=corr_base, r1_corr=corr_r1,
        ),
    )

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for tag, r in overall.items():
    print(f"  {tag:10s}: base z-mse={r['base_zmse_total']:.4f}  r1 z-mse={r['r1_zmse_total']:.4f}  "
          f"({r['pct_change']:+.1f}%)")

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune_regression_check_results.json")
with open(out_path, "w") as f:
    json.dump(overall, f, indent=2)
print(f"\nSaved to {out_path}")
