"""Classical baseline: how well does a plane-wave fit reconstruct direction?

Fits the shower front directly from detector positions and arrival times, with no
network anywhere, so it bounds what the timing alone carries. The learned recon
should beat it; if it does not, the network is not adding information.

    python eval/eval_classical_baseline.py --n-events 2000
"""
import argparse, math, os, sys
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import numpy as np, torch
import common as _common  # noqa: E402  (shared eval setup)
import modules_v6  # noqa: F401 -- sys.path injection for v3 + v4
from modules_v6.fnn_surrogate_ne import compute_labels_batch
from modules_v6.tr_surface_map_ne import SurfaceUpMap
from modules_v4.tr_geometry import load_tr_mountain
from modules_v6.constants import (
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
)
_etu = _common.load_true_utility()

# Vacuum speed of light [m/s]. Documented for the model in the module docstring; the
# fit itself never needs to divide by it, since normalizing beta[1:4] to a unit
# vector removes any uniform scale factor, 1/c included.
C_LIGHT = 2.998e8

# Fired-detector-count bands for the per-band breakdown.
BANDS = [(0, 4, "0-4"), (5, 14, "5-14"), (15, 29, "15-29"), (30, None, "30+")]


def fit_plane_wave(positions, times, weights, min_detectors=4, ref_dir=None):
    """Weighted least-squares plane-wave timing fit for a SINGLE event.

    Model t_i = t0 + (d . r_i) / c, linear in (t0, d/c). Design rows are
    (1, East, North, Up), scaled by sqrt(weight) so brighter detectors dominate.
    The direction is beta[1:4] renormalized; the common 1/c cancels.

    `ref_dir` resolves the sign. The model has no ambiguity in principle, but
    which sign of d the corpus's stored directions use is a thing to check
    against data rather than assume, so a caller can pin every event to one
    convention without touching that event's own truth.

    Args:
        positions     : (n, 3) detector (East, North, Up) [m].
        times         : (n,) combined arrival times [s].
        weights       : (n,) nonnegative fit weights, typically detector counts.
        min_detectors : below this the 4-unknown system is underdetermined and
            None is returned rather than an unreliable fit.
        ref_dir       : optional (3,) reference direction for sign resolution.

    Returns:
        (3,) unit direction tensor, or None if underdetermined or degenerate.
    """

    positions = torch.as_tensor(positions)
    times = torch.as_tensor(times)
    weights = torch.as_tensor(weights)
    n = int(positions.shape[0])
    if n < min_detectors:
        return None

    dtype = torch.float64
    pos64 = positions.to(dtype)
    t64 = times.to(dtype)
    w64 = weights.to(dtype).clamp(min=0.0)

    design = torch.cat(
        [torch.ones(n, 1, dtype=dtype, device=pos64.device), pos64], dim=1
    )                                                     # (n, 4): [1, East, North, Up]
    sqrt_w = torch.sqrt(w64).unsqueeze(-1)                 # (n, 1)
    design_w = design * sqrt_w
    times_w = (t64 * sqrt_w.squeeze(-1)).unsqueeze(-1)

    solution = torch.linalg.lstsq(design_w, times_w).solution.squeeze(-1)  # (4,)
    d_raw = solution[1:4]
    norm = d_raw.norm()
    if not torch.isfinite(norm) or float(norm) < 1e-30:
        return None
    d = (d_raw / norm).to(positions.dtype)

    if ref_dir is not None:
        ref = torch.as_tensor(ref_dir, dtype=d.dtype, device=d.device)
        ref = ref / ref.norm().clamp(min=1e-12)
        if float(torch.dot(d, ref)) < 0.0:
            d = -d
    return d


def _q(x):
    """median, 68%, 95% containment of a 1-D error tensor."""
    x = x.detach().float().cpu()
    return (float(x.median()),
            float(torch.quantile(x, 0.68)),
            float(torch.quantile(x, 0.95)))


def _angular_deg(pred_dir, true_dir):
    """(N, 3), (N, 3) unit-ish directions -> (N,) great-circle error [deg]."""
    pd = pred_dir / pred_dir.norm(dim=1, keepdim=True).clamp(min=1e-12)
    td = true_dir / true_dir.norm(dim=1, keepdim=True).clamp(min=1e-12)
    cos = (pd * td).sum(dim=1).clamp(-1.0, 1.0)
    return torch.arccos(cos) * (180.0 / math.pi)


def _band_of(n_fired):
    for lo, hi, label in BANDS:
        if n_fired >= lo and (hi is None or n_fired <= hi):
            return label
    return BANDS[-1][2]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-events", type=int, default=4000)
    ap.add_argument("--layout", type=str, default="grid",
                    help="'grid', 'center', or path to a layout_best.pt")
    ap.add_argument("--chunk", type=int, default=64,
                    help="events per compute_labels_batch chunk; bounds peak memory.")
    ap.add_argument("--fire-threshold", type=float, default=1.0,
                    help="a detector counts as fired when N_tot exceeds this.")
    ap.add_argument("--corpus", type=str, default=None,
                    help="Held-out shower corpus override (default: the constants "
                         "held-out corpus, see eval_true_utility._corpus_paths).")
    ap.add_argument("--min-detectors", type=int, default=4,
                    help="fewer fired detectors than this -> fit marked failed.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mtn = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                           east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX, n_planes=N_PLANES)
    surf = SurfaceUpMap.from_mountain(mtn).to(dev)

    corpus_path, _ = _common._corpus_paths(args.corpus)
    elec, muon, B, n_pairs = _etu.load_events(args.n_events, dev, corpus_override=args.corpus)
    prim = _common.build_primaries(corpus_path, B, mtn).to(dev)

    if args.layout == "grid":
        e_l, n_l = _etu.grid_layout(mtn)
    elif args.layout == "center":
        e_l, n_l = _etu.center_layout(mtn)
    else:
        _etu.LAYOUT_PATH = args.layout; e_l, n_l = _etu.load_layout(mtn)
    e_det, n_det = e_l.to(dev), n_l.to(dev)
    up_det = surf(n_det, e_det)                            # Up = g(North, East), once
    positions = torch.stack([e_det, n_det, up_det], dim=-1).cpu()   # (n_det, 3) East,North,Up

    # Population-level reference direction: the mean of the TRUE directions over
    # this whole batch, never a single event's own truth. Used both as the
    # constant PRIOR baseline and to pin the plane fit's sign convention.
    ref_dir = prim[:, 0:3].mean(dim=0)
    ref_dir = (ref_dir / ref_dir.norm().clamp(min=1e-12)).cpu()
    true_dirs = prim[:, 0:3].cpu()

    print("=" * 72)
    print("classical plane-wave timing fit -- non-ML baseline")
    print("=" * 72)
    print(f"device         : {dev}")
    print(f"layout         : {args.layout}")
    print(f"corpus         : {corpus_path}")
    print(f"events         : {B} of {n_pairs} pairs")
    print(f"fire_threshold : {args.fire_threshold}")
    print(f"min_detectors  : {args.min_detectors}")

    n_fired_all = torch.zeros(B, dtype=torch.long)
    plane_dirs = [None] * B
    failed = torch.zeros(B, dtype=torch.bool)
    core_e = torch.full((B,), float("nan"))
    core_n = torch.full((B,), float("nan"))

    for lo in range(0, B, args.chunk):
        hi = min(lo + args.chunk, B)
        # load_events keeps clouds HOST-resident (see eval_true_utility.KernelDualLabels'
        # docstring); stream each chunk to `dev` here, same as that class does.
        elec_c = elec[lo:hi].to(dev, non_blocking=True)
        muon_c = muon[lo:hi].to(dev, non_blocking=True)
        e_e, t_e = compute_labels_batch(elec_c, e_det, n_det, surf)
        e_m, t_m = compute_labels_batch(muon_c, e_det, n_det, surf)
        n_tot = (e_e + e_m).cpu()                          # (chunk, n_det)
        t_num = (e_e * t_e + e_m * t_m).cpu()
        t_tot = t_num / n_tot.clamp(min=1e-12)
        del e_e, t_e, e_m, t_m, elec_c, muon_c

        for b in range(hi - lo):
            i = lo + b
            mask = n_tot[b] > args.fire_threshold
            n_fired = int(mask.sum().item())
            n_fired_all[i] = n_fired
            if n_fired == 0:
                failed[i] = True
                continue
            w_b = n_tot[b][mask]
            core_e[i] = float((positions[:, 0][mask] * w_b).sum() / w_b.sum())
            core_n[i] = float((positions[:, 1][mask] * w_b).sum() / w_b.sum())
            if n_fired < args.min_detectors:
                failed[i] = True
                continue
            d = fit_plane_wave(positions[mask], t_tot[b][mask], w_b,
                               min_detectors=args.min_detectors, ref_dir=ref_dir)
            if d is None:
                failed[i] = True
            else:
                plane_dirs[i] = d

    ok = ~failed
    n_ok = int(ok.sum().item())

    prior_pred = ref_dir.unsqueeze(0).expand(B, -1)
    err_prior = _angular_deg(prior_pred, true_dirs)

    print()
    print(f"failed fits    : {B - n_ok} / {B}  ({100.0 * (B - n_ok) / B:.1f}%)")
    if n_ok > 0:
        plane_pred = torch.stack([plane_dirs[i] for i in range(B) if plane_dirs[i] is not None])
        plane_true = true_dirs[ok]
        err_plane = _angular_deg(plane_pred, plane_true)
        m, p68, p95 = _q(err_plane)
        print(f"  {'PLANE FIT':20s}  Δψ med={m:6.2f}° (68%={p68:.1f} 95%={p95:.1f})   (n={n_ok})")
    else:
        print(f"  {'PLANE FIT':20s}  no successful fits")
    m, p68, p95 = _q(err_prior)
    print(f"  {'PRIOR (predict mean)':20s}  Δψ med={m:6.2f}° (68%={p68:.1f} 95%={p95:.1f})   (n={B})")

    print()
    print("per-band breakdown (band = fired-detector count):")
    print(f"  {'band':8s} {'n_events':>9s} {'fail_frac':>10s} {'plane med':>10s} {'prior med':>10s}")
    for lo, hi, label in BANDS:
        band_mask = torch.tensor([_band_of(int(n_fired_all[i])) == label for i in range(B)])
        n_band = int(band_mask.sum().item())
        if n_band == 0:
            print(f"  {label:8s} {0:9d} {'--':>10s} {'--':>10s} {'--':>10s}")
            continue
        fail_frac = float(failed[band_mask].float().mean())
        band_ok = band_mask & ok
        if int(band_ok.sum().item()) > 0:
            band_pred = torch.stack(
                [plane_dirs[i] for i in range(B) if band_mask[i] and plane_dirs[i] is not None]
            )
            band_true = true_dirs[band_ok]
            plane_med = float(_angular_deg(band_pred, band_true).median())
            plane_med_s = f"{plane_med:.2f}"
        else:
            plane_med_s = "--"
        prior_med = float(_angular_deg(prior_pred[band_mask], true_dirs[band_mask]).median())
        print(f"  {label:8s} {n_band:9d} {fail_frac:10.2f} {plane_med_s:>10s} {prior_med:10.2f}")

    valid_core = ~torch.isnan(core_e)
    if int(valid_core.sum().item()) > 0:
        print()
        print(f"core centroid (count-weighted, reporting only, n={int(valid_core.sum().item())}):")
        print(f"  East  mean={float(core_e[valid_core].mean()):8.1f} m  std={float(core_e[valid_core].std()):7.1f} m")
        print(f"  North mean={float(core_n[valid_core].mean()):8.1f} m  std={float(core_n[valid_core].std()):7.1f} m")


if __name__ == "__main__":
    main()
