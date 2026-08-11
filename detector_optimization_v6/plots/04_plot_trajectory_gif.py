"""Animate the stage-4 detector layout from `trajectory.pt`.

One panel per scheme, in the (East, North) plane the optimizer works in, plus a
second GIF zoomed on the first --zoom-epochs Adam epochs.

Two phases are stored per chain. `adam` frames are mountain-projected every
step. `lbfgs` frames are closure calls — line-search probes included, so U is
not monotonic along them — and are projected only at the end of each sweep
chunk, which is why detectors are seen outside the valid region there.

    python plots/04_plot_trajectory_gif.py
    python plots/04_plot_trajectory_gif.py --monotonic --seconds 20 -o out.gif
    python plots/04_plot_trajectory_gif.py --run-dir DIR_A DIR_B
"""
import argparse
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(_HERE))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

import modules_v6  # noqa: F401 — sys.path injection for v3 + v4
from modules_v4.tr_geometry import load_tr_mountain
from modules_v6.constants import (
    GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
    EAST_ENTRY, LAYER_EAST_DX, N_PLANES, OPT_FOLDER,
)
from modules_v6.tr_geometry_ne import _ne_max_gap   # the snap tolerance itself

DET_C = "#e2691f"        # warm: reads on the grey mesh AND the blue valid band
ADAM_SMOOTH = 11         # frames to average U over before judging improvement
BRIDGE_CAP = 60          # most transit frames inserted into one gap


def _run_dirs(suffix):
    """{scheme: dir} for scheme dirs holding a trajectory.pt."""
    out = {}
    for d in sorted(glob.glob(OPT_FOLDER + "_lbfgs_ensemble" + suffix + "_*")):
        if os.path.exists(os.path.join(d, "trajectory.pt")):
            out[os.path.basename(d).split("_")[-1]] = d
    return out


def _pick_chain(traj, which):
    if which == "ref":
        return int(traj.get("ref_idx", 0))
    if which == "best":
        return int(np.argmax(np.asarray(traj["refined_U"])))
    return int(which)


def _utilities(run_dir, k, traj):
    """{phase: (U, chunk, step)} aligned 1:1 with traj[phase][k].

    trajectory.pt stores only the final scalar U, but optimize_log.json holds
    the per-step values and both are strided by the same pos_log_every, so the
    frame->step map is exact. `step` is carried explicitly because the L-BFGS
    handover frames make it drift from frame_index * pos_log_every.

    Unscored frames are NaN: the Adam init, and each chunk's projected handover
    — except the last, which IS scored and equals the chain's refined_U.
    """
    path = os.path.join(run_dir, "optimize_log.json")
    if not os.path.exists(path):
        print(f"  [warn] no optimize_log.json in {run_dir} — no per-frame U")
        return {}
    d = json.load(open(path))
    every = int(traj["pos_log_every"])
    out = {}

    alog = d.get("adam_logs", [])
    if k < len(alog) and alog[k]:
        n = len(alog[k])
        hit = [e for e in range(n) if (e + 1) % every == 0 or e == n - 1]
        out["adam"] = (np.array([np.nan] + [alog[k][e]["U"] for e in hit]),
                       np.zeros(len(hit) + 1, dtype=int),
                       np.array([0] + [e + 1 for e in hit]))

    llog = d.get("lbfgs_logs", [])
    if k < len(llog) and llog[k]:
        e = llog[k]
        # `iter` restarts at 0 on every chunk (one LBFGS.step per chunk).
        bounds = ([0] + [i for i in range(1, len(e)) if e[i]["iter"] <= e[i - 1]["iter"]]
                  + [len(e)])
        U, chunk, step = [], [], []
        for c in range(len(bounds) - 1):
            seg = e[bounds[c]:bounds[c + 1]]
            for j in range(0, len(seg), every):
                U.append(seg[j]["U"]); chunk.append(c); step.append(bounds[c] + j)
            U.append(np.nan); chunk.append(c); step.append(bounds[c + 1])
        U = np.array(U)
        U[-1] = float(np.asarray(traj["refined_U"])[k])
        out["lbfgs"] = (U, np.array(chunk), np.array(step))

    for name in list(out):                       # never mis-pair U with layouts
        t = traj[name][k]
        n = 0 if t is None else int(t.shape[0])
        if out[name][0].size != n:
            print(f"  [warn] {name}: {out[name][0].size} utilities vs {n} frames "
                  "— dropping this phase's U")
            out.pop(name)
    return out


def _smooth(U, w):
    """Centred running mean, NaNs passed through.

    Adam scores each step on a fresh random minibatch, so a running max on the
    raw U keeps the lucky batches (21 of 1001 frames) rather than the steps
    that improved the layout. Used to judge improvement, never to display.
    """
    if w <= 1:
        return U
    out = np.full(U.size, np.nan)
    for i in np.nonzero(np.isfinite(U))[0]:
        seg = U[max(0, i - w // 2):i + w // 2 + 1]
        out[i] = np.nanmean(seg)
    return out


def _keep_increasing(U, chunk, score):
    """Mask of frames that improved on the best score so far.

    Single-chunk phases (Adam) are a plain running max. L-BFGS needs two
    levels, because its U is not comparable across sweep chunks — each scores
    on its own fixed batch and overfits it, so chaining within-chunk gains
    reports U~245 for a layout truly worth ~36. Instead:

      across chunks — a chunk's FIRST closure scores the incoming layout on a
                      fresh batch, so that series is like-for-like; keep a
                      chunk only if it beats the best such entry so far.
      within a kept chunk — the batch is fixed, so a running max over the
                      chunk's own frames is the line search's own comparison.

    The last frame is always kept, so the GIF ends on the returned layout.
    """
    keep = np.zeros(U.size, dtype=bool)
    best_entry = -np.inf
    for c in np.unique(chunk):
        idx = np.nonzero(chunk == c)[0]
        fin = idx[np.isfinite(score[idx])]
        if not fin.size:
            continue
        entry = score[fin[0]]
        if c != chunk[0] and entry <= best_entry:
            best_entry = max(best_entry, entry)
            continue
        best_entry, best = max(best_entry, entry), -np.inf
        for n, i in enumerate(idx):
            if n == 0 or (np.isfinite(score[i]) and score[i] > best):
                keep[i] = True
            if np.isfinite(score[i]):
                best = max(best, score[i])
    keep[-1] = True
    return keep


def _bridge(xy, keep, max_jump, cen, hard):
    """Fill gaps where the kept path teleports; returns (positions, src, transit).

    --monotonic keeps ~2% of frames and the layout does not sit still in the
    rest: kept-path jumps move all 100 detectors ~700 m in one frame. Gaps
    longer than `max_jump` are filled first from the real dropped layouts
    (sampled evenly in ARC length, since the path wanders), then by linear
    interpolation for whatever hop is still too long — the log alone cannot be
    continuous, as consecutive logged L-BFGS frames sit 402 m apart at p90.

    `hard` marks the projected chunk handovers. Interpolation never crosses
    one: the snap is an instantaneous ~183 m yank, and smoothing it into a
    glide invents motion, reading as detectors drifting gently home.
    """
    # A strong-Wolfe probe can fling the layout 4.8e7 m; such frames are
    # off-canvas and useless as filler, so they never get picked.
    span = np.ptp(cen[:, :2], axis=0)
    lo, hi = cen[:, :2].min(0) - 0.1 * span, cen[:, :2].max(0) + 0.1 * span
    ok = (np.isfinite(xy).all((1, 2))
          & (xy >= lo).all((1, 2)) & (xy <= hi).all((1, 2)))
    hop = np.linalg.norm(np.diff(xy, axis=0), axis=2).max(1)
    hop[~(ok[:-1] & ok[1:])] = 0.0
    arc = np.concatenate([[0.0], np.cumsum(hop)])

    pos, src, transit = [xy[keep[0]]], [int(keep[0])], [False]
    for a, b in zip(keep[:-1], keep[1:]):
        a, b, budget, picks = int(a), int(b), BRIDGE_CAP, []
        if b - a > 1 and np.linalg.norm(xy[b] - xy[a], axis=1).max() > max_jump:
            mid = np.arange(a + 1, b)[ok[a + 1:b]]
            n = min(budget, int(np.ceil((arc[b] - arc[a]) / max_jump)) - 1)
            if mid.size and n > 0:
                tgt = arc[a] + (arc[b] - arc[a]) * np.arange(1, n + 1) / (n + 1)
                picks = np.unique(mid[np.searchsorted(arc[mid], tgt)
                                      .clip(0, mid.size - 1)])
                budget -= picks.size
        chain = [a] + [int(p) for p in picks] + [b]
        for u, v in zip(chain[:-1], chain[1:]):
            d = np.linalg.norm(xy[v] - xy[u], axis=1).max()
            n = min(budget, int(np.ceil(d / max_jump)) - 1) if d > max_jump else 0
            if hard is not None and hard[u + 1:v + 1].any():
                n = 0                                  # a snap: show the cut
            for w in np.arange(1, n + 1) / (n + 1):
                pos.append((1 - w) * xy[u] + w * xy[v]); src.append(u); transit.append(True)
            budget -= max(n, 0)
            pos.append(xy[v]); src.append(v); transit.append(v != b)
    return np.stack(pos), np.array(src), np.array(transit, dtype=bool)


def _panel(scheme, traj, k, utils, args, cen, step_limit=0):
    """One panel: frames, per-frame metadata, and the labels the title needs."""
    P = dict(scheme=scheme, k=k, U=float(np.asarray(traj["refined_U"])[k]),
             xy=[], tags=[], u=[], step=[], chunk=[], transit=[], n_chunks={})
    nd = int(traj["n_det"])
    for name in (("adam", "lbfgs") if args.phase == "both" else (args.phase,)):
        t = traj[name][k]
        if t is None or t.numel() == 0:
            print(f"  [warn] chain {k} has no '{name}' trajectory — skipped")
            continue
        if name in utils:
            U, chunk, step = (a.copy() for a in utils[name])
        else:
            U = np.full(t.shape[0], np.nan)
            chunk = np.zeros(t.shape[0], dtype=int)
            step = np.arange(t.shape[0]) * int(traj["pos_log_every"])
        P["n_chunks"][name] = int(np.unique(chunk).size)

        # Indices into the FULL trajectory throughout, so _bridge can reach
        # back into frames the filters dropped.
        sel = np.arange(t.shape[0])
        if step_limit:
            # Truncate BEFORE filtering, so a zoom is judged against its own
            # window rather than emptied by later, better frames.
            sel = sel[step[sel] <= step_limit]
            if not sel.size:
                print(f"  [warn] {name}: nothing at or before step {step_limit}")
                continue
        if args.monotonic and name in utils:
            score = _smooth(U, ADAM_SMOOTH) if name == "adam" else U
            m = _keep_increasing(U[sel], chunk[sel], score[sel])
            print(f"  [monotonic] {name}: kept {int(m.sum())}/{sel.size} frames"
                  + (f" over {np.unique(chunk[sel][m]).size}/{P['n_chunks'][name]}"
                     " chunks" if name == "lbfgs" else ""))
            sel = sel[m]

        a = t.float().numpy()
        xy = np.stack([a[:, :nd], a[:, nd:]], axis=-1)
        if args.max_jump and sel.size > 1:
            hard = None
            if name == "lbfgs":                  # last frame of each chunk is
                hard = np.zeros(t.shape[0], dtype=bool)   # the projected snap
                hard[np.nonzero(np.diff(chunk))[0]] = True
                hard[-1] = True
            xy, sel, tr = _bridge(xy, sel, args.max_jump, cen, hard)
            print(f"  [bridge] {name}: +{int(tr.sum())} transit frames "
                  f"(hops under {args.max_jump:g} m)")
        else:
            xy, tr = xy[sel], np.zeros(sel.size, dtype=bool)
        P["xy"].append(xy); P["tags"] += [name] * len(sel)
        P["u"].append(U[sel]); P["step"].append(step[sel])
        P["chunk"].append(chunk[sel]); P["transit"].append(tr)

    if not P["xy"]:
        raise SystemExit(f"chain {k}: no data for phase='{args.phase}'")
    for key in ("xy", "u", "step", "chunk", "transit"):
        P[key] = np.concatenate(P[key], axis=0)
    print(f"  {scheme:8s} chain {k}  frames={len(P['xy'])}  "
          f"final U={P['U']:+.3f}")
    return P


def _resample(panels, n_out, min_per_chunk=0):
    """Common n_out-frame timeline, allocated per phase.

    Runtime is split between phases by their mean frame count, then each panel
    is resampled WITHIN each phase — so panels cross the adam->lbfgs handover
    together even when they keep very different numbers of frames. Nearest-
    index sampling upsamples too, so a short panel holds frames rather than
    losing any.

    `min_per_chunk` floors how many frames each L-BFGS sweep chunk gets. Plain
    proportional sampling starves the short chunks — at 300 output frames the
    median chunk got 1 frame and 36 of 167 got none, so nothing WITHIN a chunk
    was ever visible. Each chunk takes the floor first, then the remaining
    budget is shared out by chunk length.
    """
    phases = [p for p in ("adam", "lbfgs") if any(p in P["tags"] for P in panels)]
    n = {p: np.mean([P["tags"].count(p) for P in panels]) for p in phases}
    slots = {p: max(1, int(round(n_out * n[p] / (sum(n.values()) or 1)))) for p in phases}
    slots[max(slots, key=slots.get)] += n_out - sum(slots.values())

    def _pick(idx, k):
        return idx[np.linspace(0, idx.size - 1, max(k, 1)).round().astype(int)]

    for P in panels:
        tags, src = np.asarray(P["tags"]), []
        for p in phases:
            idx = np.nonzero(tags == p)[0]
            if not idx.size:
                src.append(np.array([int(src[-1][-1]) if src else 0])); continue
            if p != "lbfgs" or not min_per_chunk:
                src.append(_pick(idx, slots[p])); continue
            ck = P["chunk"][idx]
            uniq, length = np.unique(ck, return_counts=True)
            spare = slots[p] - min_per_chunk * uniq.size
            if spare < 0:
                print(f"  [warn] {slots[p]} lbfgs slots cannot give {uniq.size} "
                      f"chunks {min_per_chunk} each — raise --seconds/--fps")
                per = np.full(uniq.size, max(1, slots[p] // uniq.size))
            else:
                exact = spare * length / length.sum()
                per = min_per_chunk + np.floor(exact).astype(int)
                rem = int(slots[p] - per.sum())
                if rem > 0:
                    order = np.argsort(-(exact - np.floor(exact)))
                    per[order[:rem]] += 1
            src.append(np.concatenate([_pick(idx[ck == c], k)
                                       for c, k in zip(uniq, per)]))
        out = np.concatenate(src)
        if out.size != n_out:
            out = out[np.linspace(0, out.size - 1, n_out).round().astype(int)]
        P["idx"] = out


def _writer(path, fps):
    """Writer chosen by extension. MP4 for anything but .gif.

    GIF stores frame delays in centiseconds, so only 100/n fps is exact — a
    60 fps GIF is really 50, and 2 minutes of it is ~165 MB. ffmpeg is not on
    this cluster, so the H.264 path leans on the binary bundled with
    imageio-ffmpeg when matplotlib cannot find one itself.
    """
    if path.lower().endswith(".gif"):
        exact = 100 / round(100 / fps) if fps else fps
        if abs(exact - fps) > 0.01:
            print(f"  [warn] GIF cannot do {fps} fps (centisecond delays); it "
                  f"will play at {exact:.0f}. Use an .mp4 output for {fps}")
        return PillowWriter(fps=fps)
    try:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    return FFMpegWriter(fps=fps, codec="libx264", extra_args=[
        "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "veryfast",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])       # yuv420p needs even dims


def _render(panels, out, cen, tri, title, n_out, fps, region, min_per_chunk=0):
    _resample(panels, n_out, min_per_chunk)
    fig, axes = plt.subplots(1, len(panels), figsize=(5.75 * len(panels), 5.6),
                             dpi=100, squeeze=False)
    fig.suptitle(title, fontsize=12)

    arts = []
    for ax, P in zip(axes[0], panels):
        if region is not None:
            XX, YY, D, lo, hi, gap = region
            ax.contourf(XX, YY, D, levels=[0, gap], colors=["#bcd8ef"], alpha=.55)
            ax.contour(XX, YY, D, levels=[gap], colors=["#2a78d6"], linewidths=.9,
                       linestyles="--")
            ax.plot([], [], ls="--", c="#2a78d6", lw=.9, label=f"valid region (≤{gap:.0f} m)")
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        else:
            ax.set_xlim(cen[:, 0].min() - 60, cen[:, 0].max() + 60)
            ax.set_ylim(cen[:, 1].min() - 60, cen[:, 1].max() + 60)
        ax.tricontourf(tri, cen[:, 2], levels=24, cmap="Greys", alpha=.55, zorder=2)
        ax.scatter(*P["xy"][0].T, s=16, marker="x", c="#5c5a55", lw=.8,
                   label="start", zorder=3)
        sc = ax.scatter(*P["xy"][P["idx"][0]].T, s=28, c=DET_C, edgecolors="white",
                        lw=.5, label="detectors", zorder=5)
        ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]"); ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8, framealpha=.9)
        arts.append((sc, ax.set_title("", fontsize=10), P))

    fig.tight_layout(rect=(0, 0, 1, 0.94))

    def update(f):
        for sc, ttl, P in arts:
            # Index into the panel's OWN frame list, so filtering and zooming
            # never renumber the optimizer steps they kept.
            i = P["idx"][f]
            sc.set_offsets(P["xy"][i])
            tag = P["tags"][i]
            where = (f"epoch {P['step'][i]}" if tag == "adam" else
                     f"closure {P['step'][i]}  chunk {P['chunk'][i] + 1}/{P['n_chunks']['lbfgs']}")
            u = P["u"][i]
            ttl.set_text(f"{P['scheme']}  chain {P['k']}   final U={P['U']:+.2f}\n"
                         f"{tag}  {where}   "
                         f"{f'U={u:+.2f}' if np.isfinite(u) else 'U=n/a'}"
                         + ("   (transit)" if P["transit"][i] else ""))
        return [a for art in arts for a in art[:2]]

    print(f"[render] {n_out} frames @ {fps} fps -> {n_out / fps:.1f}s")
    FuncAnimation(fig, update, frames=n_out, blit=False).save(
        out, writer=_writer(out, fps))
    plt.close(fig)
    print(f"[save] {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opt-suffix", default="_full_corpus")
    ap.add_argument("--run-dir", nargs="+", help="scheme dir(s); overrides --opt-suffix")
    ap.add_argument("--phase", choices=("adam", "lbfgs", "both"), default="both")
    ap.add_argument("--chain", default="best", help="'best', 'ref', or an index")
    ap.add_argument("--monotonic", action="store_true",
                    help="keep only frames that improved on the best utility so "
                         "far (drops rejected line-search probes)")
    ap.add_argument("--max-jump", type=float, default=0.0,
                    help="metres a detector may move between frames before the "
                         "gap is filled in with the layouts the filters dropped "
                         "(0 = off, which lets --monotonic teleport ~700 m)")
    ap.add_argument("--no-region", dest="region", action="store_false",
                    help="omit the shaded band the projection admits")
    ap.add_argument("--only", choices=("both", "full", "zoom"), default="both",
                    help="the full render is minutes long; 'zoom' redoes the short one")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--zoom-seconds", type=float, default=20.0)
    ap.add_argument("--zoom-epochs", type=int, default=500,
                    help="second GIF covering the first N Adam epochs (0 = skip)")
    ap.add_argument("--fps", type=int, default=60,
                    help="GIF stores frame delays in centiseconds, so only "
                         "100/n is exact: 50, 33, 25, 20. 60 is written as 50")
    ap.add_argument("--min-per-chunk", type=int, default=10,
                    help="floor on output frames per L-BFGS sweep chunk, so "
                         "motion WITHIN a chunk is visible (0 = purely "
                         "proportional, which starves the short chunks)")
    ap.add_argument("-o", "--output",
                    help="output path; .gif or .mp4 (mp4 for long/high-fps runs)")
    args = ap.parse_args()

    if args.run_dir:
        dirs = {os.path.basename(d.rstrip("/")).split("_")[-1]: d for d in args.run_dir}
    else:
        dirs = _run_dirs(args.opt_suffix)
    if not dirs:
        raise SystemExit(f"no trajectory.pt under {OPT_FOLDER}_lbfgs_ensemble"
                         f"{args.opt_suffix}_* — run 04 first, or pass --run-dir")
    print(f"schemes : {list(dirs)}")
    out = args.output or os.path.join(os.path.dirname(list(dirs.values())[0]),
                                      "layout_trajectory.mp4")

    mountain = load_tr_mountain(GEOMETRY_PATH_RESOLVED, GEOMETRY_GROUP, DET_KEY,
                                east_entry=EAST_ENTRY, layer_east_dx=LAYER_EAST_DX,
                                n_planes=N_PLANES)
    cen = np.asarray(mountain.centroids_ENU)
    tri = mtri.Triangulation(cen[:, 0], cen[:, 1])

    region = None
    if args.region:
        # project_to_mountain_ne leaves a detector alone within max_gap of ANY
        # centroid and snaps it onto the nearest past that, so the union of
        # those disks is where the optimizer may sit — and, since every stage-1
        # layout strategy ends with the same call, where the surrogate was
        # trained. The mesh drawn under it is only the surface, which is why
        # detectors look "off the mountain" while still being legal.
        gap = _ne_max_gap(mountain)
        lo, hi = cen[:, :2].min(0) - 1.35 * gap, cen[:, :2].max(0) + 1.35 * gap
        XX, YY = np.meshgrid(np.linspace(lo[0], hi[0], 460),
                             np.linspace(lo[1], hi[1], 460))
        from scipy.spatial import cKDTree
        D = cKDTree(cen[:, :2]).query(np.stack([XX.ravel(), YY.ravel()], 1))[0]
        region = (XX, YY, D.reshape(XX.shape), lo, hi, gap)
        print(f"[region] valid band = within {gap:.1f} m of a centroid")

    loaded = {}
    for scheme, d in dirs.items():
        traj = torch.load(os.path.join(d, "trajectory.pt"), map_location="cpu",
                          weights_only=False)
        k = _pick_chain(traj, args.chain)
        loaded[scheme] = (traj, k, _utilities(d, k, traj))

    want = max(1, int(round(args.seconds * args.fps)))
    if args.only in ("both", "full"):
        print(f"[full] phase={args.phase}{'  monotonic' if args.monotonic else ''}")
        panels = [_panel(s, *v, args, cen) for s, v in loaded.items()]
        # Never resample below the frame count, or the transit frames --max-jump
        # just inserted get dropped again; capped at 3x so an unfiltered run
        # cannot become a multi-minute, multi-hundred-MB GIF.
        have = max(len(P["xy"]) for P in panels)
        n_out = int(np.clip(have, want, 3 * want)) if args.max_jump else want
        if args.min_per_chunk:      # the floor sets its own minimum budget
            n_c = max(P["n_chunks"].get("lbfgs", 0) for P in panels)
            frac = np.mean([P["tags"].count("lbfgs") / len(P["tags"]) for P in panels])
            n_out = max(n_out, int(np.ceil(args.min_per_chunk * n_c / max(frac, 1e-9))))
        if args.max_jump and have > 3 * want:
            print(f"  [warn] {have} frames > 3x the {want} requested — some hops "
                  f"will exceed {args.max_jump:g} m; raise --seconds")
        _render(panels, out, cen, tri,
                f"Stage-4 layout optimization — {args.phase} phase"
                + ("  (increasing-U frames only)" if args.monotonic else ""),
                n_out, args.fps, region, args.min_per_chunk)

    if args.zoom_epochs > 0 and args.only in ("both", "zoom"):
        print(f"[zoom] adam phase, first {args.zoom_epochs} epochs")
        zoom_args = argparse.Namespace(**{**vars(args), "phase": "adam"})
        panels = [_panel(s, *v, zoom_args, cen, args.zoom_epochs)
                  for s, v in loaded.items()]
        n_out = max(int(round((args.zoom_seconds or args.seconds) * args.fps)),
                    max(len(P["xy"]) for P in panels))
        stem, ext = os.path.splitext(out)
        _render(panels, f"{stem}_first{args.zoom_epochs}ep{ext or '.mp4'}", cen, tri,
                f"Stage-4 layout optimization — adam phase, first "
                f"{args.zoom_epochs} epochs"
                + ("  (increasing-U frames only)" if args.monotonic else ""),
                n_out, args.fps, region)


if __name__ == "__main__":
    main()
