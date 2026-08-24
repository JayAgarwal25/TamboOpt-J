"""Bounded, physics-interpretable recon resolution — companion to eval_true_utility.py.

Reports MEDIAN and 68/95% containment of the recon's error on a fixed event batch
and a fixed layout, for a chosen recon, under BOTH label sources:
    angular   Δψ   [deg]      energy  |Δlog10 E| [dex]      vertex  ‖Δv‖ [m] (7-out)
  input = FNN   → "surrogate" view (the mirage)
  input = KERNEL→ "true" view (deployment)

Key methodology (per the pre-run verification):
  - --held-out (default): score on VALIDATION events (shower_level_split, seed/frac
    matching 03), NOT training events, so a regularizer like noise-aug isn't
    under-credited. --all-events restores the old first-N behaviour.
  - a predict-the-population-mean PRIOR baseline is always printed, so "regressed to
    the prior" (e.g. at high noise_scale) is visible rather than hidden by a median.
  - vertex is reported PER-AXIS with a skill score = median|Δaxis| / prior_std_axis
    (skill << 1 = genuinely localized), not just a 3D norm.

    python eval/eval_recon_resolution.py --recon_dir <dir> --layout grid
    python eval/eval_recon_resolution.py --recon_dir <dir> --layout /path/to/layout_best.pt
"""
import argparse, math, os, sys
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import numpy as np, torch
import common as _common  # noqa: E402  (shared eval setup)
import modules_v6  # noqa: F401
import showerdata
from modules_v6.opt_core import load_models
from modules_v6.fnn_surrogate_ne import place_clouds_enu
from modules_v6.tr_surface_map_ne import SurfaceUpMap
from modules_v4.tr_geometry import load_tr_mountain
from modules_v6.constants import (
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES, LOG_E_MIN, LOG_E_MAX,
    TRAINING_DATASET_FOLDER, DUAL_SHOWER_CACHE_PATH, DUAL_POSITIONS_PATH,
    RECON_FOLDER,
    FNN_FOLDER,
)
_etu = _common.load_true_utility(_ROOT)

# Split constants — MUST match 03_train_recon_deepsets.py (SEED, VAL_FRAC).
SPLIT_SEED = 1
VAL_FRAC   = 0.10


def heldout_event_indices(n_want):
    """Event indices (electron shower j) that landed in the recon's VAL split."""
    strat = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "strategy_ids.pt")).long()
    n_strat   = int(strat.max().item() + 1)
    n_showers = strat.shape[0] // n_strat
    g = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(n_showers, generator=g)
    n_val = max(1, int(round(VAL_FRAC * n_showers)))
    is_val = torch.zeros(n_showers, dtype=torch.bool); is_val[perm[:n_val]] = True
    n_pairs = torch.load(DUAL_POSITIONS_PATH).shape[0] // 2
    ev = torch.nonzero(is_val[:n_pairs]).squeeze(-1)   # electron-of-event-j in val
    return ev[:n_want]


def load_events_indices(idx, device):
    """Load + place electron/muon clouds for specific event indices."""
    idx = idx.long()
    hi = int(idx.max().item()) + 1
    positions_all = torch.load(DUAL_POSITIONS_PATH)
    n_pairs = positions_all.shape[0] // 2
    e_sub = showerdata.load(DUAL_SHOWER_CACHE_PATH, start=0, stop=hi)
    m_sub = showerdata.load(DUAL_SHOWER_CACHE_PATH, start=n_pairs, stop=n_pairs + hi)
    elec = torch.as_tensor(e_sub.points, dtype=torch.float32)[idx]
    muon = torch.as_tensor(m_sub.points, dtype=torch.float32)[idx]
    dirs = torch.as_tensor(e_sub.directions, dtype=torch.float32)[idx]
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-12)
    pos = positions_all[idx].float()
    place_clouds_enu(elec, pos, dirs, east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX)
    place_clouds_enu(muon, pos, dirs, east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX)
    return elec, muon, idx.shape[0], n_pairs


def _q(x):
    """median, 68%, 95% containment of a 1-D error tensor."""
    x = x.detach().float().cpu()
    return (float(x.median()),
            float(torch.quantile(x, 0.68)),
            float(torch.quantile(x, 0.95)))


def _angular_deg(pred_dir, true_dir):
    pd = pred_dir / pred_dir.norm(dim=1, keepdim=True).clamp(min=1e-12)
    td = true_dir / true_dir.norm(dim=1, keepdim=True).clamp(min=1e-12)
    cos = (pd * td).sum(dim=1).clamp(-1.0, 1.0)
    return torch.arccos(cos) * (180.0 / math.pi)


def _resolutions(pred, primary, has_vertex):
    """pred (B,D) raw encode units; primary (B,8). dict of (median,p68,p95) tuples."""
    out = {"dpsi_deg": _q(_angular_deg(pred[:, 0:3], primary[:, 0:3])),
           "dlogE_dex": _q((pred[:, 3] - primary[:, 3]).abs() * (LOG_E_MAX - LOG_E_MIN))}
    if has_vertex:
        out["dvertex_m"] = _q((pred[:, 4:7] - primary[:, 5:8]).norm(dim=1))
        for a, col in (("E", 4), ("N", 5), ("U", 6)):
            out[f"dv{a}_m"] = _q((pred[:, col] - primary[:, col + 1]).abs())
    return out


@torch.no_grad()
def _recon_pred(e_det, n_det, primary_batch, labeller, recon, device, chunk=None):
    """Predictions for every event, computed in event-chunks.

    Both label paths blow up with the event count: the kernel materializes
    (events, points, detectors), and the FNN/recon carry (events, detectors,
    hidden) activations. Chunking keeps peak memory flat in the event count, which
    is what lets this be scored on tens of thousands of held-out events instead of
    a few hundred. Every stage here is per-event independent, so the chunked result
    is identical to the single-shot one.

    A labeller advertising `needs_offset` is windowed over stored per-event data
    (the kernel holds the clouds), so it is told where the chunk starts."""
    B = int(primary_batch.shape[0])
    chunk = int(chunk) if chunk else B
    windowed = bool(getattr(labeller, "needs_offset", False))
    e_d, n_d = e_det.to(device), n_det.to(device)
    outs = []
    for lo in range(0, B, chunk):
        hi = min(lo + chunk, B)
        prim_c = primary_batch[lo:hi]
        xy = torch.stack([e_d, n_d], -1).unsqueeze(0).expand(hi - lo, -1, -1)
        ET = labeller(prim_c, xy, offset=lo) if windowed else labeller(prim_c, xy)
        feats = torch.stack([xy[..., 0], xy[..., 1], ET[..., 0], ET[..., 1]], dim=-1)
        outs.append(recon(feats))
        del xy, ET, feats
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)


def _prior_pred(primary_all, B, has_vertex, device):
    """Constant predictor = population-mean primary (renormalized direction).
    Returns a (B, D) tensor broadcasting the prior over the batch."""
    D = 7 if has_vertex else 4
    p = torch.zeros(D, device=device)
    d = primary_all[:, 0:3].mean(0); p[0:3] = d / d.norm().clamp(min=1e-12)
    p[3] = primary_all[:, 3].mean()
    if has_vertex:
        p[4:7] = primary_all[:, 5:8].mean(0)
    return p.unsqueeze(0).expand(B, -1)


def _fmt(res, has_vertex):
    m = res
    s = (f"Δψ med={m['dpsi_deg'][0]:6.2f}° (68%={m['dpsi_deg'][1]:.1f} 95%={m['dpsi_deg'][2]:.1f})   "
         f"|Δlog10E| med={m['dlogE_dex'][0]:.3f} dex")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recon_dir", type=str, default=None)
    ap.add_argument("--fnn_folder", type=str, default=None,
                    help="directory holding fnn_electron.pt and fnn_muon.pt. Without "
                         "this the surrogate comes from FNN_FOLDER in constants, which "
                         "is NOT necessarily the surrogate --recon_dir's recon was "
                         "trained against; pairing a recon with a different surrogate "
                         "silently invalidates the 'surrogate (FNN)' row.")
    ap.add_argument("--n-events", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layout", type=str, default="grid",
                    help="'grid', 'center', or path to a layout_best.pt")
    ap.add_argument("--all-events", action="store_true",
                    help="score the first N events (old behaviour) instead of held-out val")
    ap.add_argument("--corpus", type=str, default=None,
                    help="Score a separate held-out shower corpus. Every event in it "
                         "is held out from ALL stages (not just the recon's val "
                         "split), and they can be read contiguously, so this is both "
                         "the stronger test and the one that scales to large N.")
    ap.add_argument("--chunk", type=int, default=256,
                    help="events per forward chunk; bounds peak memory so --n-events "
                         "can be large. Results are identical to a single shot.")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mtn = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                           east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES)
    surf = SurfaceUpMap.from_mountain(mtn).to(dev)
    primary_all = torch.load(os.path.join(TRAINING_DATASET_FOLDER, "primary.pt")).float()

    if args.corpus:
        # Every event here is held out from all stages, so take them contiguously.
        # primary.pt is row-aligned with the TRAINING corpus, so re-encode instead.
        elec, muon, B, n_pairs = _etu.load_events(args.n_events, dev,
                                                  corpus_override=args.corpus)
        prim = _common.build_primaries(args.corpus, B, mtn).to(dev)
        ev_desc = f"{B} of {n_pairs} events from a corpus held out from ALL stages"
    elif args.all_events:
        elec, muon, B, n_pairs = _etu.load_events(args.n_events, dev)
        prim = primary_all[:B].to(dev)
        ev_desc = f"first {B} events (INCLUDES training)"
    else:
        idx = heldout_event_indices(args.n_events)
        elec, muon, B, n_pairs = load_events_indices(idx, dev)
        prim = primary_all[idx].to(dev)
        ev_desc = f"{B} events held out from the RECON only (stage-2 split differs)"

    kfn = _etu.KernelDualLabels(elec, muon, surf, dev, chunk=args.chunk)
    fnn, recon = load_models(dev,
                             fnn_folder=args.fnn_folder or FNN_FOLDER,
                             recon_dir=args.recon_dir or RECON_FOLDER + "_deepsets")
    has_vertex = int(getattr(recon, "output_dim", 4)) == 7

    if args.layout == "grid":
        e_l, n_l = _etu.grid_layout(mtn)
    elif args.layout == "center":
        e_l, n_l = _etu.center_layout(mtn)
    else:
        _etu.LAYOUT_PATH = args.layout; e_l, n_l = _etu.load_layout(mtn)

    print("=" * 72)
    print("recon resolution — kernel vs surrogate labels, same recon + layout")
    print("=" * 72)
    print(f"device      : {dev}")
    print(f"recon_dir   : {args.recon_dir or '(constants default)'}")
    print(f"fnn_folder  : {args.fnn_folder or '(constants default)'}")
    print(f"output_dim  : {getattr(recon, 'output_dim', 4)}  (vertex={'yes' if has_vertex else 'no'})")
    print(f"layout      : {args.layout}")
    print(f"events      : {ev_desc}")

    # prior (no-skill) baseline — same metric, constant population-mean prediction
    prior = _prior_pred(primary_all.to(dev), B, has_vertex, dev)
    pres = _resolutions(prior, prim, has_vertex)
    print(f"  {'PRIOR (predict mean)':20s}  {_fmt(pres, has_vertex)}")

    for name, lab in (("surrogate (FNN)", fnn), ("true (KERNEL)", kfn)):
        pred = _recon_pred(e_l, n_l, prim, lab, recon, dev, chunk=args.chunk)
        res = _resolutions(pred, prim, has_vertex)
        print(f"  {name:20s}  {_fmt(res, has_vertex)}")

    if has_vertex:
        prior_std = primary_all[:, 5:8].std(0)   # (E,N,U) prior spread [m]
        print("\n  vertex per-axis (median |Δ| [m]  |  skill = median|Δ|/prior_std):")
        for name, lab in (("surrogate (FNN)", fnn), ("true (KERNEL)", kfn)):
            pred = _recon_pred(e_l, n_l, prim, lab, recon, dev, chunk=args.chunk)
            res = _resolutions(pred, prim, has_vertex)
            parts = []
            for i, a in enumerate(("E", "N", "U")):
                med = res[f"dv{a}_m"][0]; sk = med / float(prior_std[i].clamp(min=1e-6))
                parts.append(f"{a}: {med:7.0f} m (skill {sk:.2f})")
            print(f"    {name:18s}  ‖Δv‖med={res['dvertex_m'][0]:.0f} m   " + "  ".join(parts))
        print(f"    prior_std [m]: E={prior_std[0]:.0f} N={prior_std[1]:.0f} U={prior_std[2]:.0f}")


if __name__ == "__main__":
    main()
