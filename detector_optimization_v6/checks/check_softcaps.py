"""Are CAP_THETA / CAP_PHI / CAP_E sized right, and does the E decode change them?

_soft_cap's rule is that the cap must sit well ABOVE the typical per-event
reward, or the bulk of events saturate together and the term stops telling
layouts apart. This measures the actual reward distribution on real data and
reports, per term:

  * percentiles of the UNCAPPED per-event reward 1/(err^2 + eps)
  * what fraction of events the current cap saturates
  * the SPREAD of the capped term across a set of layouts -- the number that
    decides whether the term discriminates at all

The energy term is reported under both decodes (exp(x)-1, the old one, and
10**x, the current one), so the effect of that change is measured rather than
assumed.

    sbatch check_softcaps.sh        # needs a GPU; too heavy for a login node
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import modules_v6  # noqa: F401 — sys.path injection for v3 + v4
from modules_v6.constants import (
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY, EAST_ENTRY, LAYER_EAST_DX,
    N_PLANES, N_DETECTORS, TRAINING_DATASET_FOLDER, OPT_FOLDER,
    LOG_E_MIN, LOG_E_MAX,
)
from modules_v6.opt_core import (
    load_models, reconstructability, LAYOUT_THRESHOLD, RECONSTRUCT_THRESHOLD,
    TAU_LAYOUT, TAU_RECONSTRUCT, CAP_THETA, CAP_PHI, CAP_E,
    W_THETA, W_PHI, W_E, W_DIV,
)
from modules_v6.tr_geometry_ne import sample_initial_layout_ne
from modules_v4.tr_geometry import load_tr_mountain

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def decode(primary, mode):
    """(E_GeV, theta, phi) under either energy decode."""
    dir_z = primary[:, 2].clamp(-1.0, 1.0)
    log_e = primary[:, 3] * (LOG_E_MAX - LOG_E_MIN) + LOG_E_MIN
    E = torch.pow(10.0, log_e) if mode == "pow10" else torch.exp(log_e) - 1.0
    theta = torch.arccos(dir_z)
    phi = torch.atan2(primary[:, 1], primary[:, 0])
    phi = torch.where(phi < 0, phi + 2.0 * np.pi, phi)
    return E, theta, phi


@torch.no_grad()
def rewards(x, y, primary, fnn, recon, mode):
    """Per-event (r, inv_err_theta, inv_err_phi, inv_err_E), all UNCAPPED."""
    B = primary.shape[0]
    xy = torch.stack([x, y], dim=-1).unsqueeze(0).expand(B, -1, -1)
    pred_ET = fnn(primary, xy)
    feats = torch.stack([xy[..., 0], xy[..., 1], pred_ET[..., 0], pred_ET[..., 1]], dim=-1)
    pred = recon(feats)
    E_p, th_p, ph_p = decode(pred, mode)
    E_t, th_t, ph_t = decode(primary, mode)
    E_p = E_p.clamp(min=1.0)
    r = reconstructability(torch.expm1(pred_ET[..., 0]),
                           layout_threshold=LAYOUT_THRESHOLD, tau_layout=TAU_LAYOUT,
                           reconstruct_threshold=RECONSTRUCT_THRESHOLD,
                           tau_reconstruct=TAU_RECONSTRUCT)
    return (r,
            1.0 / ((th_p - th_t) ** 2 + .001),
            1.0 / ((ph_p - ph_t) ** 2 + .001),
            1.0 / ((torch.log10(E_p) - torch.log10(E_t)) ** 2 + .01))


def soft_cap(v, cap):
    return cap * np.tanh(v / cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-events", type=int, default=8000)
    ap.add_argument("--n-layouts", type=int, default=12)
    ap.add_argument("--jitter", type=float, default=25.0,
                    help="metres of gaussian jitter for the probe layouts")
    ap.add_argument("--run-dir", nargs="+", help="dirs holding layouts_all.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    mt = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                          east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX,
                          n_planes=N_PLANES)
    fnn, recon = load_models(DEVICE)
    prim = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt"),
                      map_location="cpu", weights_only=False).float()
    idx = torch.randperm(prim.shape[0])[:args.n_events]
    prim = prim[idx].to(DEVICE)

    layouts = []
    for scheme in ("center", "grid"):
        e, n = sample_initial_layout_ne(mt, n_units=N_DETECTORS, scheme=scheme)
        layouts.append((scheme, torch.tensor(e).float(), torch.tensor(n).float()))
    base_e, base_n = layouts[0][1], layouts[0][2]
    for i in range(args.n_layouts):
        layouts.append((f"jitter{i}",
                        base_e + torch.randn_like(base_e) * args.jitter,
                        base_n + torch.randn_like(base_n) * args.jitter))
    for d in (args.run_dir or []):
        p = os.path.join(d, "layouts_all.pt")
        if os.path.exists(p):
            al = torch.load(p, map_location="cpu", weights_only=False)["aligned"]
            for k in range(al.shape[0]):
                layouts.append((f"{os.path.basename(d)[-6:]}#{k}",
                                al[k, :, 0].float(), al[k, :, 1].float()))

    print(f"device {DEVICE}   events {prim.shape[0]}   layouts {len(layouts)}", flush=True)
    print(f"current caps: THETA={CAP_THETA} PHI={CAP_PHI} E={CAP_E}\n", flush=True)

    terms = [("theta", CAP_THETA, W_THETA), ("phi", CAP_PHI, W_PHI), ("E", CAP_E, W_E)]
    for mode in ("expm1", "pow10"):
        per_layout = {t: [] for t, _, _ in terms}
        pooled = {t: [] for t, _, _ in terms}
        for name, e, n in layouts:
            r, it, ip, ie = rewards(e.to(DEVICE), n.to(DEVICE), prim, fnn, recon, mode)
            r = r.cpu().numpy()
            for (t, cap, w), v in zip(terms, (it, ip, ie)):
                v = v.cpu().numpy()
                pooled[t].append(v)
                per_layout[t].append(float(np.mean(r * soft_cap(v, cap))))

        print(f"=== energy decode: {mode} "
              f"{'(current)' if mode == 'pow10' else '(previous)'}", flush=True)
        for t, cap, w in terms:
            v = np.concatenate(pooled[t])
            q = np.percentile(v, [50, 75, 90, 95, 99])
            u = np.array(per_layout[t])
            wu = u * w / W_DIV
            print(f"  {t:5s} uncapped reward  p50={q[0]:8.1f} p75={q[1]:8.1f} "
                  f"p90={q[2]:8.1f} p95={q[3]:8.1f} p99={q[4]:8.1f} max={v.max():8.1f}")
            print(f"        cap={cap:<6.0f} saturates {np.mean(v > 0.9 * cap) * 100:5.1f}% "
                  f"of events   (suggested cap ~ p95 = {q[3]:.0f})")
            print(f"        weighted term across layouts: mean={wu.mean():7.3f} "
                  f"sd={wu.std():.4f} range={wu.max() - wu.min():.4f}"
                  f"   rel-spread={wu.std() / max(abs(wu.mean()), 1e-9) * 100:.2f}%",
                  flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
