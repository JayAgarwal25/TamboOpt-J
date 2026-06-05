"""Honest FNN evaluation: conditional-on-fired R² + fire precision/recall.

Total z-scored val-MSE is a flattering metric here — 68% of detector-samples are
near-zero (unfired), so a predict-near-mean model scores a deceptively good R².
The numbers that actually decide whether the surrogate is good enough for stage 4
are measured on the FIRED detectors (E > FIRE_THR in log1p units):

    - E R² on fired only        (target ≥ 0.6 to call the schedule fix a success)
    - T R² on fired only
    - fire-classification precision / recall  (target precision ≥ 0.8)

This reads `fnn.pt` from FNN_FOLDER and the corpus from TRAINING_DATASET_FOLDER,
recomputes the same shower-level val split + log-T transform the trainer uses,
and prints the conditional metrics. READ-ONLY: writes nothing.

Run:

    cd TambOpt/detector_optimization_v6
    python eval_fnn_fired_r2.py
"""
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch

import modules_v6  # noqa: F401
from modules_v6.fnn_surrogate import FNNSurrogate
from modules_v6.constants import (
    N_DETECTORS, PRIMARY_DIM,
    TRAINING_DATASET_FOLDER, FNN_FOLDER,
)

_spec = importlib.util.spec_from_file_location(
    "_t2", os.path.join(_HERE, "02_train_fnn.py"))
_t2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_t2)
shower_level_split = _t2.shower_level_split

FIRE_THR  = 0.1     # log1p-E above which a detector counts as "fired"
VAL_FRAC  = 0.10
SEED      = 0
EVAL_CAP  = 200_000  # cap val-rows scored, for speed; set 0 to use all
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Success bar — clearing this means the schedule fix (path c) worked; missing it
# is the trigger to fall through to the DeepSets rewrite (path a).
TARGET_E_R2_FIRED      = 0.60
TARGET_FIRE_PRECISION  = 0.80


def _r2(pred, tgt):
    ss_res = ((pred - tgt) ** 2).sum()
    ss_tot = ((tgt - tgt.mean()) ** 2).sum().clamp(min=1e-12)
    return float(1.0 - ss_res / ss_tot)


@torch.no_grad()
def main():
    print("=" * 72)
    print("v6/eval_fnn_fired_r2.py — conditional-on-fired FNN metrics")
    print("=" * 72)

    ckpt = torch.load(os.path.join(FNN_FOLDER, "fnn.pt"), map_location=DEVICE)
    cfg = ckpt.get("config", {})
    model = FNNSurrogate(n_det=N_DETECTORS, primary_dim=PRIMARY_DIM,
                         hidden=int(cfg.get("hidden", 1024)),
                         dropout=float(cfg.get("dropout", 0.1))).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    norm_stats = ckpt["norm_stats"]
    model.set_normalization(norm_stats)
    model.eval()
    print(f"[load] fnn.pt  epoch={ckpt.get('epoch','?')}  "
          f"val_total={ckpt.get('val_total','?')}  hidden={int(cfg.get('hidden',1024))}")

    primary   = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()
    xy        = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "xy.pt")).float()
    E_all     = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "E.pt")).float()
    T_all     = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "T.pt")).float()
    strat_ids = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "strategy_ids.pt")).long()

    T_all = torch.log1p(T_all * 1.0e8)   # match trainer's canonical target
    _, val_idx = shower_level_split(strat_ids, VAL_FRAC, SEED)
    if EVAL_CAP and val_idx.shape[0] > EVAL_CAP:
        g = torch.Generator().manual_seed(SEED)
        val_idx = val_idx[torch.randperm(val_idx.shape[0], generator=g)[:EVAL_CAP]]
    print(f"[eval] scoring {val_idx.shape[0]} val rows × {N_DETECTORS} detectors")

    # Predict in chunks.
    E_pred = torch.empty((val_idx.shape[0], N_DETECTORS))
    T_pred = torch.empty((val_idx.shape[0], N_DETECTORS))
    B = 2048
    for lo in range(0, val_idx.shape[0], B):
        sel = val_idx[lo:lo + B]
        p_b  = primary[sel].to(DEVICE)
        xy_b = xy[sel].to(DEVICE)
        out = model(p_b, xy_b)
        E_pred[lo:lo + B] = out[..., 0].cpu()
        T_pred[lo:lo + B] = out[..., 1].cpu()

    E_t = E_all[val_idx]; T_t = T_all[val_idx]
    fired = E_t > FIRE_THR
    pred_fire = E_pred > FIRE_THR

    # Conditional-on-fired regression quality.
    e_r2_all   = _r2(E_pred.flatten(), E_t.flatten())
    e_r2_fired = _r2(E_pred[fired], E_t[fired])
    t_r2_fired = _r2(T_pred[fired], T_t[fired])

    # Fire-classification quality.
    tp = (pred_fire & fired).sum().item()
    fp = (pred_fire & ~fired).sum().item()
    fn = (~pred_fire & fired).sum().item()
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)

    # Compression diagnostic (predict-the-mean signature).
    std_ratio = float(E_pred[fired].std() / E_t[fired].std().clamp(min=1e-12))

    print("-" * 72)
    print(f"  E R² (all detectors)   : {e_r2_all:6.3f}   (flattering — includes empties)")
    print(f"  E R² (fired only)      : {e_r2_fired:6.3f}   target ≥ {TARGET_E_R2_FIRED}")
    print(f"  T R² (fired only)      : {t_r2_fired:6.3f}")
    print(f"  fire precision         : {precision:6.3f}   target ≥ {TARGET_FIRE_PRECISION}")
    print(f"  fire recall            : {recall:6.3f}")
    print(f"  fired pred/target std  : {std_ratio:6.3f}   (1.0 = no compression; <1 = predict-the-mean)")
    print("-" * 72)

    passed = (e_r2_fired >= TARGET_E_R2_FIRED) and (precision >= TARGET_FIRE_PRECISION)
    if passed:
        print("✅ PASS — schedule fix (path c) cleared the bar. Keep the flat MLP.")
    else:
        print("❌ BELOW BAR — path (c) insufficient. Recommend the DeepSets rewrite (path a):")
        if e_r2_fired < TARGET_E_R2_FIRED:
            print(f"     • fired-E R² {e_r2_fired:.3f} < {TARGET_E_R2_FIRED} "
                  f"(magnitude still regresses to the mean)")
        if precision < TARGET_FIRE_PRECISION:
            print(f"     • fire precision {precision:.3f} < {TARGET_FIRE_PRECISION} "
                  f"(model over-fires; leaks energy onto empty detectors)")

    out_json = os.path.join(FNN_FOLDER, "eval_fired_r2.json")
    with open(out_json, "w") as f:
        json.dump(dict(
            e_r2_all=e_r2_all, e_r2_fired=e_r2_fired, t_r2_fired=t_r2_fired,
            fire_precision=precision, fire_recall=recall, fired_std_ratio=std_ratio,
            fire_thr=FIRE_THR, n_val_rows=int(val_idx.shape[0]),
            target_e_r2_fired=TARGET_E_R2_FIRED, target_fire_precision=TARGET_FIRE_PRECISION,
            passed=bool(passed),
        ), f, indent=2)
    print(f"[save] {out_json}")


if __name__ == "__main__":
    main()
