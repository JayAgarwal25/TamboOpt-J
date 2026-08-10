"""Animate the stage-4 detector layout from the logged position trajectory.

Reads `trajectory.pt` (written by 04_optimize_lbfgs_ensemble.py, see
POS_LOG_EVERY) and renders a GIF of the detectors moving over the mountain in
the (East, North) plane the optimizer actually works in. One panel per scheme.

Two phases are stored per chain and both can be shown:
  adam  — every POS_LOG_EVERY-th Adam step, AFTER the mountain projection;
          frame 0 is the chain's starting layout.
  lbfgs — every POS_LOG_EVERY-th L-BFGS CLOSURE call, concatenated over the
          sweep chunks and ending on the projected optimum. These are closure
          evaluations, not accepted iterates: strong-Wolfe probes points it
          then rejects, so U is not monotonic along this path, and the layout
          is unprojected until that final frame. Expect a visible jump at the
          adam->lbfgs handover.

Both phases play for a share of the runtime proportional to their frame count,
so the pacing reflects where the optimizer actually spent its steps.

--monotonic keeps only the frames that improved on the best utility so far,
which is how you drop the rejected line-search probes. See _monotonic_mask for
why "so far" is per sweep chunk and not global.

Run from the v6 folder:

    cd TambOpt/detector_optimization_v6
    python plots/04_plot_trajectory_gif.py
    python plots/04_plot_trajectory_gif.py --phase adam --seconds 10
    python plots/04_plot_trajectory_gif.py --phase both --monotonic
    python plots/04_plot_trajectory_gif.py --opt-suffix _full_corpus --chain best
    python plots/04_plot_trajectory_gif.py --run-dir /path/to/one/scheme_dir -o out.gif
"""
import argparse
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_V6_DIR = os.path.dirname(_HERE)
if _V6_DIR not in sys.path:
    sys.path.insert(0, _V6_DIR)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation, PillowWriter

import modules_v6  # noqa: F401 — sys.path injection for v3 + v4
from modules_v4.tr_geometry import load_tr_mountain
from modules_v6.constants import (
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES, OPT_FOLDER,
)


def _find_run_dirs(opt_suffix):
    """Scheme dirs holding a trajectory.pt, as {scheme: dir}."""
    pattern = OPT_FOLDER + "_lbfgs_ensemble" + opt_suffix + "_*"
    out = {}
    for d in sorted(glob.glob(pattern)):
        if os.path.exists(os.path.join(d, "trajectory.pt")):
            out[os.path.basename(d).split("_")[-1]] = d
    return out


def _pick_chain(traj, which):
    """Chain index: 'best' = highest final U, 'ref' = the ensemble reference."""
    if which == "ref":
        return int(traj.get("ref_idx", 0))
    if which == "best":
        return int(np.argmax(np.asarray(traj["refined_U"])))
    return int(which)


def _frame_utilities(run_dir, k, traj):
    """Per-frame utility for chain k, rebuilt from optimize_log.json.

    trajectory.pt keeps only the final scalar refined_U, but the optimizer's
    per-step U sits next to it in optimize_log.json, and the frame->step map is
    exact because both are strided by the same pos_log_every:

      adam  — frame 0 is the pre-optimization init, which has no U (NaN);
              frame i>0 is the i-th logged epoch, reproducing the "every Nth
              step, and always the last one" rule the optimizer records
              positions with (04_optimize_lbfgs_ensemble.py:213).
      lbfgs — `iter` restarts at 0 on every sweep chunk, so the flat log splits
              on those resets. Inside a chunk, closure j is logged when
              j % every == 0; each chunk then ends on a mountain-projected
              handover frame whose U is never logged (NaN).

    Returns {phase: (U, chunk_id)} aligned 1:1 with traj[phase][k], dropping any
    phase whose count disagrees with the trajectory rather than silently
    mis-pairing layouts with utilities.
    """
    path = os.path.join(run_dir, "optimize_log.json")
    if not os.path.exists(path):
        print(f"  [warn] no optimize_log.json in {run_dir} — no per-frame U")
        return {}
    with open(path) as f:
        d = json.load(f)
    every = int(traj["pos_log_every"])
    out = {}

    alog = d.get("adam_logs", [])
    if k < len(alog) and alog[k]:
        n_ep = len(alog[k])
        logged = [e for e in range(n_ep) if (e + 1) % every == 0 or e == n_ep - 1]
        U = np.array([np.nan] + [alog[k][e]["U"] for e in logged], dtype=float)
        out["adam"] = (U, np.zeros(U.size, dtype=int))

    llog = d.get("lbfgs_logs", [])
    if k < len(llog) and llog[k]:
        ent = llog[k]
        bounds = ([0]
                  + [i for i in range(1, len(ent)) if ent[i]["iter"] <= ent[i - 1]["iter"]]
                  + [len(ent)])
        U, chunk = [], []
        for c in range(len(bounds) - 1):
            seg = ent[bounds[c]:bounds[c + 1]]
            for j in range(0, len(seg), every):
                U.append(seg[j]["U"]); chunk.append(c)
            U.append(np.nan); chunk.append(c)       # projected handover frame
        out["lbfgs"] = (np.array(U, dtype=float), np.array(chunk, dtype=int))

    for name in list(out):
        t = traj[name][k]
        n = 0 if t is None else int(t.shape[0])
        if out[name][0].size != n:
            print(f"  [warn] {name}: {out[name][0].size} logged utilities vs "
                  f"{n} frames — dropping this phase's U")
            out.pop(name)
    return out


def _monotonic_mask(U, chunk):
    """Keep frames whose U beats the best so far WITHIN their own sweep chunk.

    Per chunk, not global: every L-BFGS chunk scores on its own fixed batch of
    LBFGS_BATCH_PRIMARIES primaries, so U is not comparable across chunks. A
    global running max latches onto whichever chunk happened to draw an easy
    batch and then discards every later chunk, including the layouts that are
    actually better. Within a chunk the batch is fixed, so the comparison is
    the one the line search itself is making.

    Each chunk's first frame survives as its baseline, and the final frame
    overall is always kept so the GIF ends on the returned optimum. Unscored
    handover frames (NaN) drop out.
    """
    keep = np.zeros(U.size, dtype=bool)
    for c in np.unique(chunk):
        best = -np.inf
        for n, i in enumerate(np.nonzero(chunk == c)[0]):
            if n == 0 or (np.isfinite(U[i]) and U[i] > best):
                keep[i] = True
            if np.isfinite(U[i]):
                best = max(best, U[i])
    keep[-1] = True
    return keep


def _phase_frames(traj, k, phase, utils=None, monotonic=False):
    """(n_frames, 2*n_det) for the requested phase(s), plus per-frame label, U
    and the index each kept frame had in the unfiltered trajectory (so step
    numbers stay honest under --monotonic)."""
    parts, tags, us, origs = [], [], [], []
    for name in (("adam", "lbfgs") if phase == "both" else (phase,)):
        t = traj[name][k]
        if t is None or t.numel() == 0:
            print(f"  [warn] chain {k} has no '{name}' trajectory (pre-logging "
                  "checkpoint?) — skipping that phase")
            continue
        t = t.float()
        U = utils.get(name, (None, None))[0] if utils else None
        if U is None:
            U = np.full(t.shape[0], np.nan)
        orig = np.arange(t.shape[0])
        if monotonic:
            if not utils or name not in utils:
                print(f"  [warn] {name}: no per-frame U — keeping every frame")
            else:
                m = _monotonic_mask(*utils[name])
                t, U, orig = t[torch.from_numpy(m)], U[m], orig[m]
                print(f"  [monotonic] {name}: kept {int(m.sum())}/{m.size} frames")
        parts.append(t); tags += [name] * t.shape[0]
        us.append(U); origs.append(orig)
    if not parts:
        raise SystemExit(f"chain {k}: no trajectory data for phase='{phase}'")
    return (torch.cat(parts, dim=0).numpy(), tags,
            np.concatenate(us), np.concatenate(origs))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opt-suffix", default="_full_corpus",
                    help="suffix used by the run (matches 04's --opt_suffix; "
                         "default: _full_corpus)")
    ap.add_argument("--run-dir", nargs="+", default=None,
                    help="explicit scheme dir(s) containing trajectory.pt "
                         "(overrides --opt-suffix)")
    ap.add_argument("--phase", choices=("adam", "lbfgs", "both"), default="adam",
                    help="which phase to animate (default: adam)")
    ap.add_argument("--chain", default="best",
                    help="'best' (highest final U), 'ref', or an integer index")
    ap.add_argument("--monotonic", action="store_true",
                    help="keep only frames that improved on the best utility so "
                         "far within their sweep chunk (drops rejected "
                         "line-search probes). Needs optimize_log.json beside "
                         "trajectory.pt. Note the Adam phase is scored on a "
                         "random minibatch per step, so its improvements are "
                         "partly sampling noise.")
    ap.add_argument("--seconds", type=float, default=15.0, help="GIF duration")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--trail", type=int, default=0,
                    help="draw the last N frames as a fading trail (0 = off)")
    ap.add_argument("-o", "--output", default=None,
                    help="output .gif (default: <run dir parent>/layout_trajectory.gif)")
    args = ap.parse_args()

    if args.run_dir:
        dirs = {os.path.basename(d.rstrip("/")).split("_")[-1]: d for d in args.run_dir}
    else:
        dirs = _find_run_dirs(args.opt_suffix)
    if not dirs:
        raise SystemExit(
            f"no trajectory.pt found under {OPT_FOLDER}_lbfgs_ensemble{args.opt_suffix}_*\n"
            "Run 04_optimize_lbfgs_ensemble.py first (trajectory logging is written "
            "at the end of each scheme), or pass --run-dir / --opt-suffix.")
    print(f"schemes : {list(dirs)}")

    out = args.output or os.path.join(os.path.dirname(list(dirs.values())[0]),
                                      "layout_trajectory.gif")

    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                                east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX,
                                n_planes=N_PLANES)
    cen = np.asarray(mountain.centroids_ENU)
    tri = mtri.Triangulation(cen[:, 0], cen[:, 1])

    panels = []
    for scheme, d in dirs.items():
        traj = torch.load(os.path.join(d, "trajectory.pt"),
                          map_location="cpu", weights_only=False)
        k = _pick_chain(traj, args.chain)
        utils = _frame_utilities(d, k, traj)
        flat, tags, U_frame, orig = _phase_frames(traj, k, args.phase, utils,
                                                  args.monotonic)
        nd = int(traj["n_det"])
        xy = np.stack([flat[:, :nd], flat[:, nd:]], axis=-1)      # (F, n_det, 2)
        panels.append(dict(scheme=scheme, k=k, xy=xy, tags=tags,
                           U_frame=U_frame, orig=orig,
                           U=float(np.asarray(traj["refined_U"])[k]),
                           every=int(traj["pos_log_every"])))
        print(f"  {scheme:8s} chain {k}  frames={xy.shape[0]}  "
              f"n_det={nd}  final U={panels[-1]['U']:+.3f}")

    # Resample every panel onto a common frame count so they stay in step even
    # when their phase lengths differ (L-BFGS closure counts vary per chain).
    n_out = max(1, int(round(args.seconds * args.fps)))
    for P in panels:
        src = np.linspace(0, P["xy"].shape[0] - 1, n_out).round().astype(int)
        P["idx"], P["xy_a"] = src, P["xy"][src]
        P["tag_a"] = [P["tags"][i] for i in src]

    n_col = len(panels)
    fig, axes = plt.subplots(1, n_col, figsize=(5.75 * n_col, 5.2), dpi=100,
                             squeeze=False)
    axes = axes[0]
    fig.suptitle(f"Stage-4 layout optimization — {args.phase} phase", fontsize=12)

    arts = []
    for ax, P in zip(axes, panels):
        ax.tricontourf(tri, cen[:, 2], levels=24, cmap="Greys", alpha=0.55)
        ax.scatter(P["xy"][0][:, 0], P["xy"][0][:, 1], s=16, marker="x",
                   c="#898781", linewidths=0.8, label="start", zorder=2)
        trail = [ax.scatter([], [], s=12, c="#2a78d6",
                            alpha=0.10 + 0.25 * (j + 1) / max(args.trail, 1),
                            edgecolors="none", zorder=3)
                 for j in range(args.trail)]
        sc = ax.scatter(P["xy_a"][0][:, 0], P["xy_a"][0][:, 1], s=26, c="#2a78d6",
                        edgecolors="white", linewidths=0.4, label="detectors",
                        zorder=4)
        ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
        ax.set_xlim(cen[:, 0].min() - 60, cen[:, 0].max() + 60)
        ax.set_ylim(cen[:, 1].min() - 60, cen[:, 1].max() + 60)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        arts.append((sc, trail, ax.set_title("", fontsize=10), P))

    fig.tight_layout(rect=(0, 0, 1, 0.94))

    def update(f):
        changed = []
        for sc, trail, ttl, P in arts:
            sc.set_offsets(P["xy_a"][f])
            for j, tr in enumerate(trail):
                b = f - (len(trail) - j)
                tr.set_offsets(P["xy_a"][b] if b >= 0 else np.empty((0, 2)))
                changed.append(tr)
            # Frame index in the phase's OWN unfiltered trajectory -> optimizer
            # step number, so --monotonic doesn't renumber the steps it kept.
            src_i = P["idx"][f]
            step = int(P["orig"][src_i]) * P["every"]
            u = P["U_frame"][src_i]
            u_txt = f"U={u:+.2f}" if np.isfinite(u) else "U=n/a"
            ttl.set_text(f"{P['scheme']}  chain {P['k']}   {P['tag_a'][f]}  "
                         f"~step {step}   {u_txt}   final U={P['U']:+.2f}")
            changed += [sc, ttl]
        return changed

    print(f"[render] {n_out} frames @ {args.fps} fps -> {n_out/args.fps:.1f}s")
    anim = FuncAnimation(fig, update, frames=n_out, blit=False)
    anim.save(out, writer=PillowWriter(fps=args.fps))
    print(f"[save] {out}  ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
