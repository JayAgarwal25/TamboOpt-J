"""Is a candidate input feature already implied by the detector coordinates?

Fits readouts to a target from two feature blocks: the raw coordinates, and the
richer representation under test. Reports held-out R^2 for a linear and an MLP
probe on each. The reading comes from the contrast, not from any single number.

If the coordinates alone already predict the target, the feature carries no
information the model does not have, and adding it cannot help. If the
coordinate probe is weak and the richer probe is strong, the feature is real.

The probes and the split come from probes.py, so every experiment here scores
on the same basis.
"""
import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from _pathfix import V6_ROOT  # noqa: F401 — idempotent, registers v6 root

import numpy as np
import torch
import torch.nn as nn

import modules  # noqa: F401 — package import; keeps modules on the path
from interpretability.probes import (r2, linear_probe, mlp_probe,
                                    subsample, train_test_split)
from modules.surrogates import load_dual_surrogate
from modules.geometry import SurfaceUpMap, load_tr_mountain
from modules.constants import (
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES,
    TRAINING_DATASET_FOLDER, FNN_FOLDER,
)


# ── Convention pinning ───────────────────────────────────────────────────────
def verify_surface_argument_order(surf, mountain, dev):
    """Which argument of SurfaceUpMap.forward is North?

    The detectors sit ON the surface, so evaluating the map at the mountain's own
    centroids must return those centroids' Up. Try both orders and demand that
    one wins by a wide margin; that is a property of the data, not of a comment.
    """
    cen = torch.as_tensor(mountain.centroids_ENU, dtype=torch.float32, device=dev)
    east, north, up = cen[:, 0], cen[:, 1], cen[:, 2]

    rms_north_first = torch.sqrt(torch.mean((surf(north, east) - up) ** 2)).item()
    rms_east_first  = torch.sqrt(torch.mean((surf(east, north) - up) ** 2)).item()
    spread = (up.max() - up.min()).item()

    print(f"[verify] surface arg order, reconstructing centroid Up "
          f"(relief spread {spread:.1f} m):")
    print(f"           surf(north, east) RMS = {rms_north_first:8.2f} m")
    print(f"           surf(east, north) RMS = {rms_east_first:8.2f} m")

    if rms_north_first < 0.2 * rms_east_first:
        print("[verify] -> confirmed forward(x=North, y=East)")
        return "north_first"
    if rms_east_first < 0.2 * rms_north_first:
        print("[verify] -> confirmed forward(x=East, y=North)")
        return "east_first"
    raise SystemExit("[verify] ABORT: neither argument order reconstructs the "
                     "surface clearly. Refusing to guess; every number after "
                     "this point would be meaningless.")


def verify_xy_column_order(xy_sample, mountain):
    """Which column of the corpus `xy` is North?

    North and East span different intervals on this mountain, so the assignment
    that puts both columns inside their own axis range is the right one.
    """
    east_lo, east_hi = float(mountain.centroids_ENU[:, 0].min()), float(mountain.centroids_ENU[:, 0].max())
    north_lo, north_hi = float(mountain.centroids_ENU[:, 1].min()), float(mountain.centroids_ENU[:, 1].max())
    c0_lo, c0_hi = float(xy_sample[..., 0].min()), float(xy_sample[..., 0].max())
    c1_lo, c1_hi = float(xy_sample[..., 1].min()), float(xy_sample[..., 1].max())

    def misfit(lo, hi, axis_lo, axis_hi):
        """How far the column pokes outside the axis, relative to the axis span."""
        span = max(axis_hi - axis_lo, 1e-6)
        return (max(0.0, axis_lo - lo) + max(0.0, hi - axis_hi)) / span

    # candidate A: col0=North, col1=East       candidate B: col0=East, col1=North
    mis_a = misfit(c0_lo, c0_hi, north_lo, north_hi) + misfit(c1_lo, c1_hi, east_lo, east_hi)
    mis_b = misfit(c0_lo, c0_hi, east_lo, east_hi) + misfit(c1_lo, c1_hi, north_lo, north_hi)

    print(f"[verify] mountain East  range [{east_lo:9.1f}, {east_hi:9.1f}]")
    print(f"[verify] mountain North range [{north_lo:9.1f}, {north_hi:9.1f}]")
    print(f"[verify] corpus xy col0 range [{c0_lo:9.1f}, {c0_hi:9.1f}]")
    print(f"[verify] corpus xy col1 range [{c1_lo:9.1f}, {c1_hi:9.1f}]")
    print(f"[verify] misfit (col0=North,col1=East) = {mis_a:.4f}")
    print(f"[verify] misfit (col0=East,col1=North) = {mis_b:.4f}")

    if mis_a + 1e-6 < mis_b * 0.5:
        print("[verify] -> corpus xy = (North, East)")
        return "north_first"
    if mis_b + 1e-6 < mis_a * 0.5:
        print("[verify] -> corpus xy = (East, North)")
        return "east_first"
    raise SystemExit("[verify] ABORT: the two coordinate ranges do not separate "
                     "the column order. Refusing to guess.")


# ── Probes ───────────────────────────────────────────────────────────────────






def surface_gradient(up_fn, north, east, step=10.0):
    """|grad g| by central differences, in metres of Up per metre of ground.

    Takes the Up lookup as a function of (north, east) so the caller's pinned
    argument order is applied once, in one place, instead of being re-derived
    here with a swap that has to stay in sync.
    """
    with torch.no_grad():
        dn = (up_fn(north + step, east) - up_fn(north - step, east)) / (2 * step)
        de = (up_fn(north, east + step) - up_fn(north, east - step)) / (2 * step)
    return torch.sqrt(dn ** 2 + de ** 2)


def binned_table(values, by, n_bins, label):
    """Signed bias and error magnitude of `values` in quantile bins of `by`.

    Both matter and they fail differently: a bias that drifts with the binning
    variable is a systematic the model could have learned, while a flat bias with
    a growing spread is noise the geometry does not explain.

    Returns the rows so the caller can plot and save them too.
    """
    qs = torch.quantile(by, torch.linspace(0, 1, n_bins + 1, device=by.device))
    print(f"\n  residual vs {label}:")
    print(f"    {'bin':>26}  {'n':>8}  {'bias':>9}  {'|res| med':>9}  {'|res| p84':>9}")
    rows = []
    for i in range(n_bins):
        lo, hi = qs[i], qs[i + 1]
        m = (by >= lo) & (by <= hi if i == n_bins - 1 else by < hi)
        if int(m.sum()) < 32:
            continue
        v = values[m]
        row = dict(lo=float(lo), hi=float(hi), n=int(m.sum()),
                   centre=float(by[m].median()),
                   bias=float(v.median()),
                   abs_med=float(v.abs().median()),
                   abs_p84=float(torch.quantile(v.abs(), 0.84)))
        rows.append(row)
        print(f"    [{row['lo']:10.3f}, {row['hi']:10.3f})  {row['n']:8d}  "
              f"{row['bias']:9.4f}  {row['abs_med']:9.4f}  {row['abs_p84']:9.4f}")
    return rows




def make_plots(plot_dir, species, res, preds, U, XY, grad, rows_u, rows_g):
    """Probe panel + residual profiles. Never lets a plotting failure lose data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    dev = U.device
    sub = subsample(U.shape[0], 40000, dev)
    u_s = U[sub].cpu().numpy()

    # ── figure 1: the probe contrast ────────────────────────────────────────
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    names = ["linear_from_xy", "linear_from_h", "mlp_from_xy", "mlp_from_h"]
    colours = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]

    ax[0, 0].bar(range(4), [res[k] for k in names], color=colours)
    ax[0, 0].set_xticks(range(4))
    ax[0, 0].set_xticklabels(["lin←xy", "lin←h", "mlp←xy", "mlp←h"])
    ax[0, 0].set_ylim(0.9, 1.001)
    ax[0, 0].set_ylabel("held-out $R^2$ for $u_d$")
    ax[0, 0].set_title("probe contrast (note truncated axis)")
    for i, k in enumerate(names):
        ax[0, 0].text(i, res[k] + 0.0015, f"{res[k]:.4f}", ha="center", fontsize=9)
    ax[0, 0].axhline(1.0, color="k", lw=0.6, ls=":")

    # linear-from-xy is the load-bearing one: 3 parameters, cannot memorise.
    for a, key, ttl in ((ax[0, 1], "linear_from_xy", "linear ← xy (3 params)"),
                        (ax[1, 0], "linear_from_h", "linear ← h (257 params)")):
        p = preds[key][sub].cpu().numpy()
        a.hexbin(u_s, p, gridsize=60, bins="log", cmap="viridis")
        lim = [float(min(u_s.min(), p.min())), float(max(u_s.max(), p.max()))]
        a.plot(lim, lim, "r-", lw=1.0)
        a.set_xlabel("true $u_d$ [m]"); a.set_ylabel("predicted $u_d$ [m]")
        a.set_title(f"{ttl}   $R^2$={res[key]:.4f}")

    # How planar is the surface actually? Residual of the 3-parameter fit.
    resid_plane = (preds["linear_from_xy"][sub] - U[sub]).cpu().numpy()
    sc = ax[1, 1].scatter(XY[sub, 0].cpu().numpy(), XY[sub, 1].cpu().numpy(),
                          c=resid_plane, s=2, cmap="coolwarm",
                          vmin=-3 * resid_plane.std(), vmax=3 * resid_plane.std())
    plt.colorbar(sc, ax=ax[1, 1], label="plane-fit residual [m]")
    ax[1, 1].set_xlabel("East [m]"); ax[1, 1].set_ylabel("North [m]")
    ax[1, 1].set_title(f"where the plane fails  (RMS {resid_plane.std():.1f} m)")

    fig.suptitle(f"surface representation probe — {species}", fontsize=13)
    fig.tight_layout()
    p1 = os.path.join(plot_dir, f"surface_probe_{species}_probes.png")
    fig.savefig(p1, dpi=140); plt.close(fig)
    print(f"[plot] wrote {p1}")

    # ── figure 2: residual profiles ─────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for a, rows, xl in ((ax[0], rows_u, "$u_d$ (terrain height) [m]"),
                        (ax[1], rows_g, r"$|\nabla g|$ (terrain steepness)")):
        c = [r["centre"] for r in rows]
        a.plot(c, [r["bias"] for r in rows], "o-", color="#4C78A8", label="bias (median)")
        a.plot(c, [r["abs_med"] for r in rows], "s-", color="#F58518", label="|res| median")
        a.plot(c, [r["abs_p84"] for r in rows], "^--", color="#B279A2", label="|res| p84")
        a.axhline(0, color="k", lw=0.6, ls=":")
        a.set_xlabel(xl); a.set_ylabel("residual [log1p counts]")
        a.legend(fontsize=8)
    ax[0].set_title("a rising trend here would support the feature")
    ax[1].set_title("steepness is where it would show most")
    fig.suptitle(f"surrogate residual vs terrain — {species}", fontsize=13)
    fig.tight_layout()
    p2 = os.path.join(plot_dir, f"surface_probe_{species}_residuals.png")
    fig.savefig(p2, dpi=140); plt.close(fig)
    print(f"[plot] wrote {p2}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fnn_folder", type=str, default=None,
                    help="directory with fnn_electron.pt and fnn_muon.pt. Defaults "
                         "to FNN_FOLDER from constants, which is probably NOT the "
                         "run you mean; pass it explicitly.")
    ap.add_argument("--dataset_folder", type=str, default=None,
                    help="corpus directory holding primary.pt, xy.pt, E.pt, "
                         "species_ids.pt. Defaults to TRAINING_DATASET_FOLDER.")
    ap.add_argument("--species", type=str, default="electron", choices=("electron", "muon"))
    ap.add_argument("--n-events", type=int, default=4096,
                    help="corpus rows sampled; each contributes n_det detector states")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", type=str, default=None)
    ap.add_argument("--csv", type=str, default=None,
                    help="write the binned residual tables as CSV")
    ap.add_argument("--plot_dir", type=str, default=None,
                    help="write the probe panel and residual profiles here "
                         "(created if missing); plotting failures never drop the "
                         "printed tables")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fnn_folder = args.fnn_folder or FNN_FOLDER
    dataset = args.dataset_folder or TRAINING_DATASET_FOLDER

    print("=" * 72)
    print("surface-representation probe — does the surrogate know Up = g(N, E)?")
    print("=" * 72)
    print(f"device        : {dev}")
    print(f"fnn_folder    : {fnn_folder}")
    print(f"dataset_folder: {dataset}")
    print(f"species       : {args.species}")

    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                                east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX,
                                n_planes=N_PLANES)
    surf = SurfaceUpMap.from_mountain(mountain).to(dev)

    # ── pin both conventions before touching the model ──────────────────────
    arg_order = verify_surface_argument_order(surf, mountain, dev)

    species_ids = torch.load(os.path.join(dataset, "species_ids.pt"))
    xy_all = torch.load(os.path.join(dataset, "xy.pt"))
    primary_all = torch.load(os.path.join(dataset, "primary.pt")).float()
    E_all = torch.load(os.path.join(dataset, "E.pt"))

    # Row routing must use the stage-1 species sidecar, NOT the primary's pdg
    # feature (that is the EM/hadronic class the model learns from). Mapping is
    # SPECIES_TAGS in 02_train_fnn_deepsets.py: 0 = electron, 1 = muon.
    want = 0 if args.species == "electron" else 1
    rows = torch.nonzero(species_ids == want, as_tuple=True)[0]
    if rows.numel() == 0:
        raise SystemExit(f"[abort] no rows with species_id == {want}")
    perm = torch.randperm(rows.numel(), generator=torch.Generator().manual_seed(args.seed))
    rows = rows[perm[:args.n_events]]
    print(f"[data] {rows.numel()} rows of species '{args.species}' "
          f"out of {int((species_ids == want).sum())}")

    col_order = verify_xy_column_order(xy_all[rows[:512]].float(), mountain)

    def split_ne(xy):
        """-> (north, east) whatever the corpus stores, using the pinned order."""
        return (xy[..., 0], xy[..., 1]) if col_order == "north_first" else (xy[..., 1], xy[..., 0])

    def up_of(north, east):
        return surf(north, east) if arg_order == "north_first" else surf(east, north)

    # ── capture the per-detector encoder state ──────────────────────────────
    dual = load_dual_surrogate(fnn_folder, dev)
    model = dual.electron if args.species == "electron" else dual.muon
    model.eval()

    captured = {}
    def hook(_mod, _inp, out):
        captured["h"] = out.detach()
    handle = model.encoder.register_forward_hook(hook)

    H, U, XY, RESID = [], [], [], []
    with torch.no_grad():
        for lo in range(0, rows.numel(), args.chunk):
            idx = rows[lo:lo + args.chunk]
            prim = primary_all[idx].to(dev)
            xy = xy_all[idx].float().to(dev)
            pred = model(prim, xy)                       # (b, n_det, 2), raw units
            h = captured["h"]                            # (b, n_det, hidden)
            north, east = split_ne(xy)
            u = up_of(north, east)                       # (b, n_det)
            # E.pt is already log1p(counts) on disk and the model's col 0 is the
            # same space, so this subtraction needs no transform.
            resid = pred[..., 0] - E_all[idx].float().to(dev)

            H.append(h.reshape(-1, h.shape[-1]).cpu())
            U.append(u.reshape(-1).cpu())
            # (East, North), matching the corpus column order pinned above, so
            # XY[:, 0] means the same thing here as it does on disk.
            XY.append(torch.stack([east.reshape(-1), north.reshape(-1)], dim=-1).cpu())
            RESID.append(resid.reshape(-1).cpu())
    handle.remove()

    H = torch.cat(H).to(dev)
    U = torch.cat(U).to(dev)
    XY = torch.cat(XY).to(dev)
    RESID = torch.cat(RESID).to(dev)
    # Guard the assembled array too, not just the on-disk one. The columns were
    # re-stacked above, and a silent transpose here would mislabel every plot
    # axis while leaving the probe scores untouched (feature order does not
    # matter to a regression), so nothing downstream would catch it.
    if verify_xy_column_order(XY.unsqueeze(0).cpu(), mountain) != "east_first":
        raise SystemExit("[abort] assembled XY is not (East, North); the plot "
                         "axes and the gradient call would both be wrong.")
    print(f"[data] {H.shape[0]} detector states, hidden={H.shape[1]}")
    print(f"[data] u_d range [{float(U.min()):.1f}, {float(U.max()):.1f}] m, "
          f"std {float(U.std()):.1f} m")

    tr, te = train_test_split(H.shape[0], dev, test_frac=0.2, seed=args.seed)

    print("\n" + "=" * 72)
    print("probes: predict u_d, held-out R^2 (20% test split)")
    print("=" * 72)
    res, preds = {}, {}
    res["linear_from_xy"], preds["linear_from_xy"] = linear_probe(XY[tr], U[tr], XY[te], U[te], XY)
    res["linear_from_h"],  preds["linear_from_h"]  = linear_probe(H[tr],  U[tr], H[te],  U[te],  H)
    res["mlp_from_xy"],    preds["mlp_from_xy"]    = mlp_probe(XY[tr], U[tr], XY[te], U[te], XY)
    res["mlp_from_h"],     preds["mlp_from_h"]     = mlp_probe(H[tr],  U[tr], H[te],  U[te],  H)
    n_par = {"linear_from_xy": XY.shape[1] + 1, "linear_from_h": H.shape[1] + 1,
             "mlp_from_xy": "~67k", "mlp_from_h": "~132k"}
    print(f"  {'probe':18s}  {'params':>8}  {'R^2':>8}  {'resid RMS [m]':>14}")
    for k in ("linear_from_xy", "linear_from_h", "mlp_from_xy", "mlp_from_h"):
        rms = float(U.std()) * float(max(0.0, 1.0 - res[k])) ** 0.5
        print(f"  {k:18s}  {str(n_par[k]):>8}  {res[k]:8.4f}  {rms:14.1f}")

    print("\n  reading:")
    if res["linear_from_xy"] > 0.9:
        print("    linear(xy) is already high, so the terrain is too flat here for")
        print("    this question to be meaningful. Treat the rest as uninformative.")
    elif res["linear_from_h"] > 0.9 and res["linear_from_h"] > res["linear_from_xy"] + 0.3:
        print("    Up is computed and LINEARLY DECODABLE from the encoder state.")
        print("    The network already built the surface; feeding u_d explicitly")
        print("    is redundant and the full retrain should buy little.")
    elif res["mlp_from_h"] < 0.5:
        print("    Up is NOT recoverable from the encoder state. The network never")
        print("    built the surface, so the feature has a real target.")
    else:
        print("    Up is present but only nonlinearly decodable. Ambiguous: the")
        print("    information is there, but not in a form later layers use cheaply.")

    # ── part 2: does error track the terrain? ───────────────────────────────
    print("\n" + "=" * 72)
    print("residual (surrogate - kernel, log1p counts) vs terrain")
    print("=" * 72)
    # XY is (East, North); up_of already encodes the pinned SurfaceUpMap order.
    grad = surface_gradient(up_of, north=XY[:, 1], east=XY[:, 0])
    rows_u = binned_table(RESID, U, 8, "u_d (terrain height)")
    rows_g = binned_table(RESID, grad, 8, "|grad g| (terrain steepness)")
    print("\n  A FLAT profile in both is evidence the surface is not limiting the")
    print("  surrogate. A profile that climbs with |grad g| is the signature the")
    print("  proposed feature would address.")

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("species,binned_by,lo,hi,centre,n,bias,abs_med,abs_p84\n")
            for tag, rows in (("u_d", rows_u), ("grad_g", rows_g)):
                for r in rows:
                    f.write(f"{args.species},{tag},{r['lo']:.6g},{r['hi']:.6g},"
                            f"{r['centre']:.6g},{r['n']},{r['bias']:.6g},"
                            f"{r['abs_med']:.6g},{r['abs_p84']:.6g}\n")
        print(f"[save] {args.csv}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"probes": res, "arg_order": arg_order, "col_order": col_order,
                       "species": args.species, "n_states": int(H.shape[0]),
                       "u_std_m": float(U.std()),
                       "resid_rms_m": {k: float(U.std()) * max(0.0, 1.0 - v) ** 0.5
                                       for k, v in res.items()},
                       "binned_u_d": rows_u, "binned_grad_g": rows_g}, f, indent=2)
        print(f"[save] {args.out_json}")

    # Plotting last and guarded: a failure here must never cost the tables above.
    if args.plot_dir:
        try:
            make_plots(args.plot_dir, args.species, res, preds, U, XY, grad, rows_u, rows_g)
        except Exception as exc:  # noqa: BLE001 — diagnostics must not die on plots
            print(f"[plot] SKIPPED, plotting failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
