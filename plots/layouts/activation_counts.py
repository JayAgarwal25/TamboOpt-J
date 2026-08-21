"""How many detectors does a layout actually light up, and with how many particles?

`eval_true_utility.py` answers "is the optimized layout better?" as one composite
scalar U, which folds theta/phi/E reconstruction quality together. This reports the
physical quantity underneath it — activated detectors per shower and detected
particles per shower — for the default grid and center layouts next to the optimized
one, so a difference in U can be attributed to something.

Counts come from the plane-aware KERNEL (`compute_labels_batch`, via
`eval_true_utility.KernelDualLabels`), not the surrogate. That matters twice: the
surrogate is what the optimizer maximized, so scoring with it measures its opinion
rather than the physics; and the activation statistic touches neither the surrogate
nor the recon, so it is a pure geometry result.

EVENTS — three separate leaks, closed by one choice:

  * fitted-on      the Step-0 heldout corpus is split off before generation
                   (HOLDOUT_FRAC, HOLDOUT_SEED) and never opened by Steps 1-4.
  * selected-on    `layout_best.pt` is the argmax over ~167 sweep chunks scored on
                   stage 4's scoring set, which is drawn from the TRAINING corpus —
                   disjoint from the holdout. Reporting a winner's score on the set
                   that picked it is optimistically biased even when no gradient
                   step touched that set.
  * already-used   `eval_true_utility.py` only ever loads rows 0..n_events, so
                   heldout pairs 5120+ have never been scored by anything. That
                   reserve (24,994 of the 30,114 pairs) is the default here.

`--event-offset` advances the slice, and lands in the JSON sidecar, so successive
evaluations do not quietly re-use the same events until they become a selection set.

The stage-4 scoring set appears in exactly one place: the `U_score` column, which
reproduces `score_on_set` so the figure can be tied to the run in its
`optimize_log.json`. It is IN-SAMPLE FOR SELECTION and labelled as such — it is
provenance, not a result.

    cd TambOpt
    python plots/layouts/activation_counts.py                       # whole reserve
    python plots/layouts/activation_counts.py --n-events 512        # quick look

An archived run — pass both scheme dirs' layouts; output lands beside them:

    R="$(python -c 'import sys; sys.path.insert(0, "."); \
        from modules.constants import RUN_LOCATION; print(RUN_LOCATION)')"
    R="$R/run 6 update energy calc common eval"
    python plots/layouts/activation_counts.py \\
        --layout "$R/..._center/layout_best.pt" "$R/..._grid/layout_best.pt"
"""
import argparse
import importlib.util
import json
import os
import sys

# `_HERE` is this file's own directory; `_V6` the repo root, found by walking up
# to the _pathfix.py marker instead of counting parents. Counting is what broke
# the old plots/single_species/ scripts: written for plots/*.py, they resolved
# the "repo root" to plots/ and could not import `modules` at all.
_HERE = os.path.dirname(os.path.abspath(__file__))
_V6 = _HERE
while _V6 != os.path.dirname(_V6) and not os.path.exists(os.path.join(_V6, "_pathfix.py")):
    _V6 = os.path.dirname(_V6)
for _p in (_V6, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

import modules  # noqa: F401 — package import; keeps modules on the path
import showerdata
from modules.geometry import load_tr_mountain
from modules.data import place_clouds_enu
from modules.geometry import SurfaceUpMap
from modules.geometry.placement import _ne_max_gap
from modules.optimize import (
    LAYOUT_THRESHOLD, TAU_LAYOUT, RECONSTRUCT_THRESHOLD, TAU_RECONSTRUCT,
    OFFMESH_PENALTY_W, PENALTY_ONSET_FRAC, utility_of_xy, load_models,
)
from modules.constants import (
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES, OPT_FOLDER, TRAINING_DATASET_FOLDER,
    HELDOUT_SHOWER_CACHE_PATH, HELDOUT_POSITIONS_PATH,
)
from true_utility import KernelDualLabels, _snap, grid_layout, center_layout

# Heldout pairs eval_true_utility.py has consumed (its --n-events default, always
# from row 0). Skipping them is what makes this script's sample untouched.
EVAL_TRUE_UTILITY_EVENTS = 5120

# A detector this rarely lit is drawn as "dark". Not 0: LAYOUT_THRESHOLD = 1.0 with
# TAU_LAYOUT = 5 gives a completely dark detector sigmoid(-5) = 0.0067, so a zero
# cut would call every detector live.
DARK_RATE = 0.01

DEFAULT_LAYOUT = os.path.join(OPT_FOLDER + "_lbfgs_ensemble_full_corpus_grid",
                              "layout_best.pt")


def _optimizer_config():
    """SEED / scoring-set sizes read from the optimizer itself, so the provenance
    score cannot drift from the run it is compared against. Loaded by path because
    the module name starts with a digit."""
    path = os.path.join(_V6, "scripts", "04_optimize_lbfgs_ensemble.py")
    spec = importlib.util.spec_from_file_location("_opt04", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return (int(mod.SEED), int(mod.SCORING_SET_PRIMARIES),
            int(mod.SCORING_BATCH_PRIMARIES))


def _load_layout(path, mountain):
    """(East, North) from a layout_best.pt, snapped like every other layout here."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    e, n = (raw["x"], raw["y"]) if isinstance(raw, dict) else (raw[:, 0], raw[:, 1])
    return _snap(mountain, e, n)


class Accum:
    """Per-shower statistics for one layout, accumulated across corpus blocks.

    Everything is concatenated and reduced ONCE at the end, never summed per block.
    Keeping the (B, n_det) gate matrix costs 10 MB for the whole reserve and buys
    exact block-invariance: running per-block partial sums in fp32 and adding them
    up made `det_rate` depend on --load-block at the 5e-8 level, because floating
    point addition does not regroup.
    """

    def __init__(self):
        self.parts = {k: [] for k in ("n_soft", "n_hard", "particles", "r", "p")}

    def add(self, out):
        for k in self.parts:
            self.parts[k].append(out[k])

    def finish(self):
        d = {k: np.concatenate(v) for k, v in self.parts.items()}
        d["det_rate"] = d.pop("p").mean(axis=0, dtype=np.float64)
        return d


@torch.no_grad()
def activation(kernel, e_det, n_det, device):
    """Per-shower activation of one layout, from kernel counts.

    This is `reconstructability`'s own soft gate (v3 `utility_functions.py`) with the
    intermediate detector count kept instead of collapsed straight into r:

        p = sigmoid(tau_layout * (counts - layout_threshold));  n = p.sum(dim=1)

    `expm1` because KernelDualLabels returns log1p(N_tot) — the surrogate's output
    space, which is why `utility_of_xy` un-logs it before the same call. `nan_to_num`
    mirrors what 01_build_dataset_northeast applies to the kernel labels, so one bad
    point cannot poison a whole layout's statistics.
    """
    xy = torch.stack([e_det, n_det], dim=-1).to(device).unsqueeze(0)   # (1, n_det, 2)
    # `primary_batch` is genuinely unused by KernelDualLabels (the clouds already fix
    # which events these are), so None is honest rather than a placeholder.
    out = kernel(None, xy)
    counts = torch.expm1(out[..., 0]).clamp_min(0.0)                   # (B, n_det)
    counts = torch.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
    p = torch.sigmoid(TAU_LAYOUT * (counts - LAYOUT_THRESHOLD))
    n_soft = p.sum(dim=1)
    return dict(
        n_soft=n_soft.cpu().numpy(),
        n_hard=(counts > LAYOUT_THRESHOLD).sum(dim=1).cpu().numpy(),
        particles=counts.sum(dim=1).double().cpu().numpy(),
        r=torch.sigmoid(TAU_RECONSTRUCT
                        * (n_soft - RECONSTRUCT_THRESHOLD)).cpu().numpy(),
        p=p.cpu().numpy(),                                 # (B, n_det) for det_rate
    )


def stream_counts(args, layouts, surface, device):
    """Run every layout over the heldout reserve, one block of pairs at a time.

    The whole reserve is ~25 GB of point clouds across both species, so
    `eval_true_utility.load_events` (which loads its slice whole) cannot be reused.
    Blocks are loaded once and scored against ALL layouts before being freed, so the
    corpus is read once no matter how many layouts are compared.
    """
    pos_all = torch.load(HELDOUT_POSITIONS_PATH)               # (2*n_pairs, 3) E,N,U
    n_pairs = pos_all.shape[0] // 2
    off = int(args.event_offset)
    if off >= n_pairs:
        raise SystemExit(f"--event-offset {off} past the corpus ({n_pairs} pairs)")
    if off < EVAL_TRUE_UTILITY_EVENTS:
        print(f"  [warn] offset {off} overlaps the {EVAL_TRUE_UTILITY_EVENTS} pairs "
              "eval_true_utility.py has already scored — these events are no longer "
              "untouched")
    avail = n_pairs - off
    B = avail if args.n_events <= 0 else min(int(args.n_events), avail)
    print(f"events      : pairs {off}..{off + B - 1} of {n_pairs} "
          f"({B} events, {avail} available past the offset)")

    # Metadata only — no point clouds — so E/theta/phi can be recorded for the whole
    # sample without a second pass over the corpus.
    meta = showerdata.load_inc_particles(HELDOUT_SHOWER_CACHE_PATH,
                                         start=off, stop=off + B)
    dirs = torch.as_tensor(meta.directions, dtype=torch.float32)
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-12)
    energies = torch.as_tensor(meta.energies, dtype=torch.float32).reshape(-1)

    acc = {name: Accum() for name in layouts}
    for lo in range(0, B, args.load_block):
        hi = min(lo + args.load_block, B)
        # Row i (electron) and row n_pairs+i (muon) are the same physical event, so
        # they share the vertex and direction — verified equal in the sidecar.
        clouds = []
        for start in (off + lo, n_pairs + off + lo):
            sub = showerdata.load(HELDOUT_SHOWER_CACHE_PATH,
                                  start=start, stop=start + (hi - lo))
            c = torch.as_tensor(sub.points, dtype=torch.float32)
            del sub
            bad = ~torch.isfinite(c).all(dim=-1)          # as Step 1 sanitizes
            if int(bad.sum()):
                c[bad] = 0.0
            clouds.append(place_clouds_enu(c, pos_all[off + lo:off + hi].float(),
                                           dirs[lo:hi], east_entry=EAST_ENTRY,
                                           layer_east_dx=LAYER_EAST_DX))
        kernel = KernelDualLabels(clouds[0], clouds[1], surface, device,
                                  chunk=args.kernel_chunk)
        for name, (e_det, n_det) in layouts.items():
            acc[name].add(activation(kernel, e_det, n_det, device))
        del clouds, kernel
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  [block] pairs {off + lo}..{off + hi - 1} done ({hi}/{B})",
              flush=True)

    stats = {name: a.finish() for name, a in acc.items()}
    theta = torch.arccos(dirs[:, 2].clamp(-1.0, 1.0))
    phi = torch.atan2(dirs[:, 1], dirs[:, 0]) % (2.0 * np.pi)   # East->North azimuth
    meta_out = dict(E=energies.numpy(), theta=theta.numpy(), phi=phi.numpy())
    return stats, meta_out, dict(offset=off, n_events=B, n_pairs=n_pairs)


@torch.no_grad()
def score_on_scoring_set(e_det, n_det, scoring_set, fnn, recon,
                         mesh_en, pen_r0, batch, device):
    """Mirror of `04_optimize_lbfgs_ensemble.score_on_set` for an already-snapped
    layout: penalised U in size-weighted slices, which is exact because every term
    in U is a per-event mean and the off-mesh penalty is layout-only."""
    x, y = e_det.to(device), n_det.to(device)
    tot, n_tot = 0.0, 0
    for s in range(0, scoring_set.shape[0], batch):
        b = scoring_set[s:s + batch].to(device)
        U, _, _ = utility_of_xy(x, y, b, fnn, recon, mesh_en=mesh_en,
                                penalty_w=OFFMESH_PENALTY_W, penalty_r0=pen_r0)
        tot += float(U.item()) * b.shape[0]
        n_tot += b.shape[0]
    return tot / max(n_tot, 1)


def provenance_scores(layouts, mountain, device):
    """`U_score` per layout on stage 4's own scoring set — the number its
    `optimize_log.json` logged, so a figure can be tied to the run it describes."""
    seed, n_score, batch = _optimizer_config()
    primary_all = torch.load(os.path.join(TRAINING_DATASET_FOLDER,
                                          "primary.pt")).float()
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(primary_all.shape[0], generator=g)[:min(n_score,
                                                                 primary_all.shape[0])]
    scoring_set = primary_all[idx]
    print(f"scoring set : {scoring_set.shape[0]} primaries (seed={seed}), "
          f"slices of {batch}  [in-sample for selection]")

    fnn, recon = load_models(device)
    cen = torch.as_tensor(mountain.centroids_ENU[:, :2], dtype=torch.float32).to(device)
    r0 = _ne_max_gap(mountain) * PENALTY_ONSET_FRAC
    return {name: score_on_scoring_set(e, n, scoring_set, fnn, recon, cen, r0,
                                       batch, device)
            for name, (e, n) in layouts.items()}


def _summary(d, U_score):
    n = d["n_soft"].size
    p10, p50, p90 = np.percentile(d["n_soft"], [10, 50, 90])
    return dict(
        n_events=int(n),
        soft_mean=float(d["n_soft"].mean()),
        soft_sem=float(d["n_soft"].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        soft_p10=float(p10), soft_p50=float(p50), soft_p90=float(p90),
        hard_mean=float(d["n_hard"].mean()),
        hard_sem=float(d["n_hard"].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        particles_mean=float(d["particles"].mean()),
        particles_median=float(np.median(d["particles"])),
        r_mean=float(d["r"].mean()),
        U_score=float(U_score),
    )


def report(stats, summ, baseline):
    print()
    print("ACTIVATED DETECTORS AND DETECTED PARTICLES PER SHOWER (kernel counts)")
    print(f"{'layout':<12}{'activated (soft)':>26}{'activated (hard)':>20}"
          f"{'particles/shower':>26}{'mean r':>9}{'U_score*':>10}")
    print(f"{'':<12}{'mean ±sem':>14}{'p10':>4}{'p50':>4}{'p90':>4}"
          f"{'mean ±sem':>20}{'mean':>14}{'median':>12}{'':>9}{'':>10}")
    for name, s in summ.items():
        print(f"{name:<12}"
              f"{s['soft_mean']:8.2f} ±{s['soft_sem']:<5.2f}"
              f"{s['soft_p10']:4.0f}{s['soft_p50']:4.0f}{s['soft_p90']:4.0f}"
              f"{s['hard_mean']:13.2f} ±{s['hard_sem']:<5.2f}"
              f"{s['particles_mean']:14.4g}{s['particles_median']:12.4g}"
              f"{s['r_mean']:9.4f}{s['U_score']:10.3f}")
    print("  * in-sample for selection — stage-4 scoring set, provenance only, NOT "
          "a result")

    if baseline in stats:
        print(f"\n  Δ vs {baseline} (paired over the same events):")
        b = stats[baseline]
        for name, d in stats.items():
            if name == baseline:
                continue
            for key, label in (("n_soft", "activated (soft)"),
                               ("particles", "particles/shower")):
                dd = d[key] - b[key]
                sem = dd.std(ddof=1) / np.sqrt(dd.size) if dd.size > 1 else 0.0
                print(f"    {name:<10} {label:<18} {dd.mean():+12.4g} ± {sem:.4g}")


def figure(args, layouts, stats, summ, mountain, out, event_info):
    cen = np.asarray(mountain.centroids_ENU)
    tri = mtri.Triangulation(cen[:, 0], cen[:, 1])
    gap = _ne_max_gap(mountain)

    region = None
    if args.region:
        # project_to_mountain_ne leaves a detector alone within max_gap of ANY
        # centroid, so the union of those disks is where a layout may legally sit —
        # and, since every stage-1 layout strategy ends with that call, where the
        # kernel's training support is. The mesh under it is only the surface.
        lo, hi = cen[:, :2].min(0) - 1.35 * gap, cen[:, :2].max(0) + 1.35 * gap
        XX, YY = np.meshgrid(np.linspace(lo[0], hi[0], 460),
                             np.linspace(lo[1], hi[1], 460))
        from scipy.spatial import cKDTree
        D = cKDTree(cen[:, :2]).query(np.stack([XX.ravel(), YY.ravel()], 1))[0]
        region = (XX, YY, D.reshape(XX.shape), lo, hi)

    vmax = max(float(s["det_rate"].max()) for s in stats.values())
    fig, axes = plt.subplots(1, len(layouts), squeeze=False,
                             figsize=(5.9 * len(layouts), 5.9), dpi=110)
    sc = None
    for ax, (name, (e_det, n_det)) in zip(axes[0], layouts.items()):
        if region is not None:
            XX, YY, D, lo, hi = region
            ax.contourf(XX, YY, D, levels=[0, gap], colors=["#bcd8ef"], alpha=.55)
            ax.contour(XX, YY, D, levels=[gap], colors=["#2a78d6"], linewidths=.9,
                       linestyles="--")
            ax.plot([], [], ls="--", c="#2a78d6", lw=.9,
                    label=f"valid region (≤{gap:.0f} m)")
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.tricontourf(tri, cen[:, 2], levels=24, cmap="Greys", alpha=.55, zorder=2)

        e, n = np.asarray(e_det), np.asarray(n_det)
        rate = stats[name]["det_rate"]
        live = rate > DARK_RATE
        ax.scatter(e[~live], n[~live], s=26, c="dimgray", marker="^", zorder=4,
                   label="never lit")
        if live.any():
            sc = ax.scatter(e[live], n[live], c=rate[live], s=48, cmap="plasma",
                            marker="^", norm=Normalize(0.0, vmax),
                            edgecolor="white", linewidths=.5, zorder=5,
                            label="activation rate")
        s = summ[name]
        ax.set_title(f"{name}\nactivated {s['soft_mean']:.1f} ± {s['soft_sem']:.2f} "
                     f"/100    particles {s['particles_median']:.3g}/shower",
                     fontsize=11)
        ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.legend(loc="upper left", fontsize=8, framealpha=.9)

    if sc is not None:
        cb = fig.colorbar(sc, ax=axes[0].tolist(), fraction=.025, pad=.02)
        cb.set_label("fraction of showers activating this detector", fontsize=10)
    fig.suptitle(f"Detector activation on the held-out reserve — kernel counts, "
                 f"pairs {event_info['offset']}..{event_info['offset'] + event_info['n_events'] - 1} "
                 f"({event_info['n_events']} events)", fontsize=13)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[save] {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event-offset", type=int, default=EVAL_TRUE_UTILITY_EVENTS,
                    help="first heldout pair to use; the default skips everything "
                         "eval_true_utility.py has already scored")
    ap.add_argument("--n-events", type=int, default=0,
                    help="0 = every pair past the offset")
    ap.add_argument("--load-block", type=int, default=2048,
                    help="pairs held in RAM at once; memory only, and bit-identical "
                         "at a fixed --kernel-chunk (verified 512 vs 192)")
    ap.add_argument("--kernel-chunk", type=int, default=128,
                    help="events per kernel call; memory only and per-event exact, "
                         "but NOT bit-identical across values — the kernel's fp32 "
                         "reductions pick different shapes, which moved counts by "
                         "~5e-6 relative in a measured 64-vs-32 comparison")
    ap.add_argument("--layout", nargs="+", default=[DEFAULT_LAYOUT],
                    help="optimized layout_best.pt path(s); the panel is labelled "
                         "with the run dir's trailing scheme token")
    ap.add_argument("--baselines", nargs="*", default=["grid", "center"],
                    choices=("grid", "center"))
    ap.add_argument("--no-region", dest="region", action="store_false",
                    help="omit the shaded band the projection admits")
    ap.add_argument("--no-score", dest="score", action="store_false",
                    help="skip the stage-4 U_score column (skips loading the models "
                         "and primary.pt)")
    ap.add_argument("-o", "--output",
                    help="output png; .json/.npz sidecars share its stem")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("activation counts — kernel ground truth on the untouched heldout reserve")
    print("=" * 78)
    print(f"device      : {device}")
    print(f"corpus      : {HELDOUT_SHOWER_CACHE_PATH}")

    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                                east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX,
                                n_planes=N_PLANES)
    surface = SurfaceUpMap.from_mountain(mountain).to(device)

    layouts = {}
    for name in args.baselines:
        layouts[name] = (grid_layout if name == "grid" else center_layout)(mountain)
    for path in args.layout:
        # ".../test_..._full_corpus_grid/layout_best.pt" -> "opt-grid"
        tag = "opt-" + os.path.basename(os.path.dirname(path)).split("_")[-1]
        layouts[tag] = _load_layout(path, mountain)
        print(f"layout      : {tag:<12} <- {path}")

    out = args.output or os.path.join(
        os.path.dirname(os.path.dirname(args.layout[0])), "layout_activation.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    stats, meta, event_info = stream_counts(args, layouts, surface, device)
    scores = (provenance_scores(layouts, mountain, device) if args.score
              else {k: float("nan") for k in layouts})
    summ = {name: _summary(stats[name], scores[name]) for name in layouts}

    report(stats, summ, args.baselines[0] if args.baselines else None)
    figure(args, layouts, stats, summ, mountain, out, event_info)

    stem = os.path.splitext(out)[0]
    with open(stem + ".json", "w") as f:
        json.dump(dict(corpus=HELDOUT_SHOWER_CACHE_PATH,
                       event_offset=event_info["offset"],
                       n_events=event_info["n_events"],
                       n_pairs_available=event_info["n_pairs"],
                       layout_paths={("opt-" + os.path.basename(
                           os.path.dirname(p)).split("_")[-1]): p
                           for p in args.layout},
                       layouts=summ), f, indent=2)
    np.savez_compressed(
        stem + ".npz", E=meta["E"], theta=meta["theta"], phi=meta["phi"],
        **{f"{name}__{k}": v for name, d in stats.items() for k, v in d.items()},
        **{f"{name}__xy": np.stack([np.asarray(e), np.asarray(n)], -1)
           for name, (e, n) in layouts.items()})
    print(f"[save] {stem}.json")
    print(f"[save] {stem}.npz")


if __name__ == "__main__":
    main()
