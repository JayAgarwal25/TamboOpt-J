"""Diagnostic: split the (kernel - FNN) residual into SYSTEMATIC (bias) vs
STOCHASTIC (variance) parts, per label channel. This decides whether noise
augmentation can even help: zero-mean noise can only fix the stochastic part; a
bias-dominated gap needs paired training, not augmentation.

We have one kernel realization per event, so we estimate the systematic component
as the conditional mean E[r | FNN signal] (binned by FNN output), and the
stochastic component as the within-bin variance. Reported two ways:
  - signal-binned:   systematic = Var_bin(E[r|bin]);  stochastic = E_bin(Var[r|bin])
  - per-detector:    systematic = Var_det(mean_ev r);  stochastic = E_det(Var_ev r)

    python eval/diag_residual_bias_variance.py --n-bins 25
"""
import argparse, os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from _pathfix import V6_ROOT  # noqa: F401 — idempotent, registers v6 root

import torch
import modules  # noqa: F401 — package import; keeps modules on the path
from modules.surrogates import load_dual_surrogate
from modules.constants import FNN_FOLDER, TRAINING_DATASET_FOLDER
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_m03", os.path.join(_ROOT, "03_train_recon_deepsets.py"))
_m03 = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_m03)


def _decompose(m, r, n_bins):
    """m, r flat (K,). Returns (systematic_var, stochastic_var, total_var) via
    binning r by the signal m into n_bins equal-count bins."""
    total = r.var(unbiased=False).item()
    order = torch.argsort(m)
    r_sorted = r[order]
    K = r.shape[0]
    edges = torch.linspace(0, K, n_bins + 1).long()
    bin_means, bin_vars, bin_w = [], [], []
    for b in range(n_bins):
        seg = r_sorted[edges[b]:edges[b + 1]]
        if seg.numel() == 0:
            continue
        bin_means.append(seg.mean()); bin_vars.append(seg.var(unbiased=False))
        bin_w.append(seg.numel())
    bm = torch.stack(bin_means); bv = torch.stack(bin_vars)
    w = torch.tensor(bin_w, dtype=torch.float64); w = w / w.sum()
    grand = (w * bm.double()).sum()
    systematic = (w * (bm.double() - grand) ** 2).sum().item()   # Var of E[r|bin]
    stochastic = (w * bv.double()).sum().item()                  # E of Var[r|bin]
    return systematic, stochastic, total


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bins", type=int, default=25)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    primary = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()
    xy      = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "xy.pt")).float()
    strat   = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "strategy_ids.pt")).long()
    dual = load_dual_surrogate(FNN_FOLDER, dev)
    E_fnn, T_fnn = _m03.compute_fnn_predictions(dual, primary, xy, dev)
    E_raw = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "E.pt")).float()
    T_raw = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "T.pt")).float()
    E_ker, T_ker = _m03.build_kernel_combined_labels(E_raw, T_raw, strat, dev)
    del E_raw, T_raw

    print("=" * 72)
    print("residual (kernel - FNN) bias/variance decomposition")
    print("systematic = deterministic FNN error (noise-aug CANNOT fix);")
    print("stochastic = aleatoric spread (noise-aug CAN fix, bootstrap best).")
    print("=" * 72)
    for name, m, r in (("E", E_fnn, E_ker - E_fnn), ("T", T_fnn, T_ker - T_fnn)):
        sysv, stov, tot = _decompose(m.flatten(), r.flatten(), args.n_bins)
        # per-detector view
        mu_d = r.mean(dim=0); var_ev_d = r.var(dim=0, unbiased=False)
        sys_det = mu_d.var(unbiased=False).item(); sto_det = var_ev_d.mean().item()
        print(f"\n[{name}] total residual var = {tot:.5g}")
        print(f"  signal-binned : systematic={sysv:.5g} ({100*sysv/max(tot,1e-12):.1f}%)  "
              f"stochastic={stov:.5g} ({100*stov/max(tot,1e-12):.1f}%)")
        print(f"  per-detector  : systematic={sys_det:.5g} ({100*sys_det/max(tot,1e-12):.1f}%)  "
              f"stochastic={sto_det:.5g} ({100*sto_det/max(tot,1e-12):.1f}%)")
    print("\nRule of thumb: if systematic >> stochastic, the gap is bias-dominated and "
          "noise augmentation is the wrong tool; if stochastic dominates, augmentation "
          "(bootstrap) should close most of it.")


if __name__ == "__main__":
    main()
