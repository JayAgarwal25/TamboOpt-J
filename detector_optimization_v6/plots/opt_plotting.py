"""Shared plotting for the v6/04 detector-layout optimizers.

Single home for every figure the three 04 optimizers
(`04_optimize_lbfgs_ensemble.py`, `04_optimize_differential_evolution.py`,
`04_optimize_differential_evolution_pop.py`) draw: the layout ensemble + density
heatmap (with the East→Up `project_ne_to_up` projection so detectors sit on the
mountain profile) and the per-optimizer convergence-curve / utility-component
panels. Keeping one implementation means a plotting fix (e.g. the East-on-Up
projection bug) is fixed everywhere at once.

The 04 scripts load this module by path (their filenames start with a digit, so
they cannot be imported by name) and bind the names they need. The non-plotting
core (objective, alignment, model loading, the cosine diagnostic) lives in
`modules_v6/opt_core.py`.

`plot_ensemble` / `plot_density_heatmap` default to the DE wording ("run", "K");
the population script passes `member_word="member"`/`count_word="pop"` and L-BFGS
passes `title_kind="L-BFGS ensemble"`, so each optimizer reproduces its own
figure text from the one implementation.
"""
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_V6 = os.path.dirname(_HERE)                       # detector_optimization_v6/
if _V6 not in sys.path:
    sys.path.insert(0, _V6)

import numpy as np
import torch

import modules_v6  # noqa: F401 — sys.path injection for v3 + v4
from modules_v6.opt_core import consecutive_cos_distance, LAYOUT_THRESHOLD


# Type scale for the layout figures. They are drawn on a 14-inch-wide canvas
# and then read at slide/document size, where matplotlib's 10 pt default
# shrinks to nothing. Diagnostic grids (per-chain curve panels) keep their own
# smaller sizes — those are read full-screen.
FS_TITLE  = 20
FS_LABEL  = 17
FS_TICK   = 14
FS_LEGEND = 14


# ── Layout figures ────────────────────────────────────────────────────────────
@torch.no_grad()
def project_ne_to_up(surface, north, east):
    """Map detector (North, East) → Up via the differentiable mountain surface
    Up = g(North, East) (modules_v6.tr_surface_map_ne.SurfaceUpMap).

    The optimizers work in the North–East plane, so their layouts carry East, not
    Up. To draw them in the (North, Up) cross section, project each detector's East
    through the surface to recover the height it sits at. Returns a numpy array
    shaped like `north`."""
    dev = surface.grid_up.device
    shp = np.asarray(north).shape
    n = torch.as_tensor(np.asarray(north).reshape(-1), dtype=torch.float32, device=dev)
    e = torch.as_tensor(np.asarray(east ).reshape(-1), dtype=torch.float32, device=dev)
    return surface(n, e).detach().cpu().numpy().reshape(shp)


def mountain_nue(mountain) -> np.ndarray:
    """The mountain surface as (n, 3) columns [North, Up, East], the order this
    module's plotting bodies are written in.

    Built from the canonical `centroids_ENU` ([East, North, Up]) plus, when
    present, `vertices_ENU` — the unique triangle corner points of the detector
    region. The vertices are the real surface corners (denser and truer than the
    face centroids), and they are what `SurfaceUpMap` now fits, so including them
    makes the drawn slope match the surface the detectors are actually projected
    onto. Replaces the old `mountain.centroids_NUE` compat view."""
    cen = np.asarray(mountain.centroids_ENU)                  # (n_tri, 3) [E, N, U]
    verts = getattr(mountain, "vertices_ENU", None)
    enu = cen if verts is None or not len(verts) else np.concatenate(
        [cen, np.asarray(verts)], axis=0)
    return enu[:, [1, 2, 0]]                                  # -> [North, Up, East]


def mountain_enu(mountain) -> np.ndarray:
    """The same surface points as `mountain_nue`, in canonical [East, North, Up]."""
    cen = np.asarray(mountain.centroids_ENU)
    verts = getattr(mountain, "vertices_ENU", None)
    return cen if verts is None or not len(verts) else np.concatenate(
        [cen, np.asarray(verts)], axis=0)


def cliff_frame(mountain):
    """Fit the detector region's plane and return its face-on basis.

    The malata detector region is a dipping ramp, not a North-South wall: it has
    real extent in BOTH horizontal axes. Drawing it as a (North, Up) cross
    section collapses East, so the face is seen edge-on and renders as a diagonal
    line — detectors separated in East pile onto the same spot. This builds the
    plane the ramp actually lies in so it can be drawn face-on instead.

    The plane is the least-squares (SVD) fit through the surface points; the fit
    is justified because the ramp is planar to ~0.2% of its variance (see the
    `resid_std` return). The in-plane basis is the geological one:

        normal  outward plane normal, Up component forced positive
        strike  horizontal in-plane axis (perpendicular to the dip) = Up x normal
        updip   in-plane axis pointing up the slope = normal x strike

    Returns (origin, strike, updip, normal, resid_std) with `origin` the surface
    centroid [E,N,U], the three axes unit vectors in [E,N,U], and `resid_std` the
    std of the out-of-plane residual [m] — the roughness the face view flattens."""
    P = mountain_enu(mountain)
    origin = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - origin, full_matrices=False)

    normal = Vt[2]
    if normal[2] < 0.0:
        normal = -normal                       # outward / upward-facing

    strike = np.cross(np.array([0.0, 0.0, 1.0]), normal)
    sn = np.linalg.norm(strike)
    # Degenerate only for a perfectly horizontal plane (no strike direction);
    # fall back to East so the frame stays well defined.
    strike = np.array([1.0, 0.0, 0.0]) if sn < 1e-9 else strike / sn

    updip = np.cross(normal, strike)
    updip /= np.linalg.norm(updip)
    if updip[2] < 0.0:
        updip = -updip                         # point up the slope

    resid_std = float(((P - origin) @ normal).std())
    return origin, strike, updip, normal, resid_std


def dip_deg(normal) -> float:
    """Dip of the fitted plane from horizontal [deg]."""
    return float(np.degrees(np.arccos(abs(float(normal[2])))))


def enu_to_face(pts_enu, frame):
    """Project [E,N,U] points onto the cliff plane → (strike, updip) in-plane
    coordinates [m], measured from the surface centroid."""
    origin, strike, updip = frame[0], frame[1], frame[2]
    d = np.asarray(pts_enu, dtype=float) - origin
    return d @ strike, d @ updip


@torch.no_grad()
def project_en_to_face(surface, east, north, frame):
    """Detector (East, North) → cliff-face (strike, updip).

    Lifts each detector onto the mountain with Up = g(North, East) (the same
    differentiable `SurfaceUpMap` the optimiser is scored through), then projects
    the resulting 3-D point into the fitted cliff plane. Returns two arrays shaped
    like `east`."""
    e = np.asarray(east, dtype=float)
    n = np.asarray(north, dtype=float)
    up = project_ne_to_up(surface, n, e)
    pts = np.stack([e.reshape(-1), n.reshape(-1), np.asarray(up).reshape(-1)], axis=-1)
    s, u = enu_to_face(pts, frame)
    return s.reshape(e.shape), u.reshape(e.shape)


def cloud_to_face(cloud, frame, east_entry=None, layer_east_dx=None):
    """A PLACED shower cloud → cliff-face (strike, updip).

    Thin composition of `fnn_surrogate_ne.cloud_to_enu` (undo the kernel's
    North/Up/z_cont packing) with `enu_to_face`, so a cloud can be overlaid on the
    same axes as the mountain and the detectors."""
    from modules_v6.fnn_surrogate_ne import cloud_to_enu
    from modules_v6.constants import EAST_ENTRY, LAYER_EAST_DX
    pts = cloud_to_enu(cloud,
                       east_entry=EAST_ENTRY if east_entry is None else east_entry,
                       layer_east_dx=LAYER_EAST_DX if layer_east_dx is None else layer_east_dx)
    return enu_to_face(pts, frame)


def det_counts(E, log1p_space: bool = False):
    """Detector labels → RAW counts, the space LAYOUT_THRESHOLD lives in.

    Two different things are called "E" in this codebase and they are NOT in the
    same space — mixing them up double-transforms the values:

    * ``fnn_surrogate_ne.compute_labels_batch`` returns **raw counts**. The log1p
      is applied afterwards, by 01_build_dataset_northeast.py, when writing E.pt.
      Pass these through unchanged (``log1p_space=False``, the default).
    * the trained **surrogate** predicts in log1p space, because it was fitted to
      E.pt. That is why ``opt_core.utility_of_xy`` calls
      ``reconstructability(torch.expm1(E_pred_det), ...)``. Pass those with
      ``log1p_space=True``.

    Non-finite entries are zeroed, mirroring the ``torch.nan_to_num`` the builder
    applies to the kernel output (fnn_surrogate_ne), so a single bad detector
    cannot poison a colour scale."""
    c = np.asarray(E, dtype=float)
    if log1p_space:
        c = np.expm1(c)
    return np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)


def face_limits(ax, mtn_s, mtn_u, pad: float = 400.0):
    """Frame an axis on the mountain face (+pad), equal aspect.

    A shower rod is ~11.5 km long while the array is ~1.4 km, so autoscaling to
    the cloud shrinks the detectors to a dot."""
    ax.set_xlim(float(np.min(mtn_s)) - pad, float(np.max(mtn_s)) + pad)
    ax.set_ylim(float(np.min(mtn_u)) - pad, float(np.max(mtn_u)) + pad)
    ax.set_aspect("equal")


def draw_detectors_on_face(ax, E, e_det, n_det, surface, frame,
                           layout_threshold=None, legend=True,
                           log1p_space: bool = False, dyn_range: float = 1e6,
                           cmap: str = "plasma"):
    """Scatter detectors on the cliff face, coloured by RAW counts (log scale).

    Colouring by magnitude rather than drawing a boolean "triggered" mask matters:
    the kernel's Gaussian (SIGMA_SPATIAL) leaves a numerically nonzero tail on
    essentially ALL detectors, so an ``E > 0`` cut lights the whole array up and
    hides the spatial falloff. Detectors clearing `layout_threshold` (the
    optimiser's own cut, in counts) get a ring.

    `log1p_space` says which space `E` is in — see `det_counts`. `dyn_range` caps
    how many decades the colour scale spans below the peak: real layouts mix a few
    strong detectors with a far tail, and an unbounded LogNorm over ~60 decades
    fails outright with "Invalid vmin or vmax".

    Returns (mappable_or_None, n_with_signal, n_over_threshold)."""
    from matplotlib.colors import LogNorm
    if layout_threshold is None:
        layout_threshold = LAYOUT_THRESHOLD
    ds, du = project_en_to_face(surface, np.asarray(e_det), np.asarray(n_det), frame)
    c = det_counts(E, log1p_space=log1p_space)
    live, over = c > 0.001, c > layout_threshold

    # "dimgray" rather than a mid gray value: needs to read as clearly distinct
    # from the mountain's light neutral background, not just a darker shade of
    # the same gray family.
    ax.scatter(ds[~live], du[~live], s=14, c="dimgray", marker="x", lw=0.9, zorder=2,
               label="no signal" if legend else None)
    sc = None
    if live.any():
        hi = float(c[live].max())
        lo = max(float(c[live].min()), hi / dyn_range)
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0:
            lo, hi = 1e-12, 1.0
        if hi <= lo:                      # single value / all-equal → give LogNorm room
            hi = lo * 10.0
        # `cmap` defaults to plasma rather than inferno on purpose: inferno's low
        # end is pure BLACK, so the weakest detectors render as bold black dots
        # indistinguishable from the strongest — and from a black shower cloud.
        # plasma bottoms out at dark blue-violet, keeping the ramp readable. The
        # thin white edge separates markers from the cloud behind them.
        sc = ax.scatter(ds[live], du[live], c=np.clip(c[live], lo, hi), s=42,
                        cmap=cmap, edgecolor="white", linewidths=0.5,
                        norm=LogNorm(vmin=lo, vmax=hi), zorder=3,
                        label="signal" if legend else None)
    if over.any():
        ax.scatter(ds[over], du[over], s=150, facecolors="none", edgecolors="cyan",
                   lw=1.4, label=f"> {layout_threshold:g} counts" if legend else None)
    return sc, int(live.sum()), int(over.sum())


def plot_ensemble(aligned_xy: np.ndarray,
                  mean_xy: np.ndarray,
                  std_xy: np.ndarray,
                  best_x, best_y,
                  mountain, path: str, surface=None,
                  member_word: str = "run",
                  title_kind: str = "DE ensemble",
                  count_word: str = "K"):
    """Cliff-face ensemble: every aligned run/member (faint) + per-group mean +
    1σ ellipses.

    With `surface` (a SurfaceUpMap) each detector is lifted onto the mountain via
    Up = g(North, East) and then projected FACE-ON into the fitted cliff plane, so
    the axes are the ramp's own (along-strike, up-dip) coordinates and both
    horizontal extents survive. Mean/std are recomputed in that plane. Without
    `surface` the native (North, East) map view is drawn.
    `member_word`/`title_kind`/`count_word` set the legend + title wording so each
    optimizer keeps its own labels."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        from matplotlib.collections import PatchCollection

        # Layouts arrive as (East, North) to match the ENU h5 convention.
        sub = ""
        if surface is not None:
            frame = cliff_frame(mountain)
            s, u = project_en_to_face(surface, aligned_xy[..., 0], aligned_xy[..., 1], frame)
            aligned_xy = np.stack([s, u], axis=-1)              # (K, n_det, 2) = (strike, updip)
            mean_xy = aligned_xy.mean(axis=0)
            std_xy  = aligned_xy.std(axis=0)
            best_x, best_y = project_en_to_face(
                surface, np.asarray(best_x), np.asarray(best_y), frame)
            mtn_x, mtn_y = enu_to_face(mountain_enu(mountain), frame)
            xlab, ylab = "along-strike [m]", "up-dip [m]"
            xlet, ylet = "s", "u"
            sub = (f"cliff face — dip {dip_deg(frame[3]):.1f}°, "
                   f"out-of-plane σ={frame[4]:.0f} m")
        else:
            # Body below is written x=North, y=East; reorder once here.
            aligned_xy = np.stack([aligned_xy[..., 1], aligned_xy[..., 0]], axis=-1)
            mean_xy    = np.stack([mean_xy[..., 1],    mean_xy[..., 0]],    axis=-1)
            std_xy     = np.stack([std_xy[..., 1],     std_xy[..., 0]],     axis=-1)
            best_x, best_y = best_y, best_x
            mtn = mountain_nue(mountain)                         # (n, 3) [N, Up, E]
            mtn_x, mtn_y = mtn[:, 0], mtn[:, 2]
            xlab, ylab = "North [m]", "East [m]"
            xlet, ylet = "N", "E"

        K = aligned_xy.shape[0]
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(mtn_x, mtn_y,
                   s=2, c="lightgray", alpha=0.6, label="mountain")

        colors = plt.cm.tab10(np.linspace(0, 1, max(K, 1)))
        for k in range(K):
            ax.scatter(aligned_xy[k, :, 0], aligned_xy[k, :, 1], s=8,
                       color=colors[k % 10], alpha=0.35, edgecolors="none",
                       label=f"{member_word} {k}" if k < 10 else None)

        ellipses = [
            Ellipse(xy=(float(mx), float(my)),
                    width=2.0 * float(sx), height=2.0 * float(sy))
            for (mx, my), (sx, sy) in zip(mean_xy, std_xy)
        ]
        ax.add_collection(PatchCollection(
            ellipses, facecolor="C1", edgecolor="C1", alpha=0.25, linewidths=0.6,
        ))
        ax.scatter(best_x, best_y, s=26, c="C3",
                   edgecolors="black", linewidths=0.4, alpha=0.95,
                   label=f"best  (σ̄{xlet}={std_xy[:,0].mean():.1f} m, "
                         f"σ̄{ylet}={std_xy[:,1].mean():.1f} m)")

        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        ax.set_aspect("equal")
        title = f"{title_kind} ({count_word}={K}) — aligned best + 1σ ellipses"
        ax.set_title(title + (f"\n{sub}" if sub else ""))
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] ensemble skipped ({exc!r})")


def plot_density_heatmap(aligned_xy: np.ndarray,
                         best_x, best_y,
                         mountain, path: str,
                         bins: int = 60, surface=None, vmax=None,
                         member_word: str = "run",
                         count_word: str = "K", count_suffix: str = " runs"):
    """2D density of detector placements across the ensemble, on the cliff face.

    With `surface` each detector is lifted onto the mountain via Up = g(North,
    East) and projected FACE-ON into the fitted cliff plane, so the axes are the
    ramp's own (along-strike, up-dip) coordinates; without it the native
    (North, East) map view is drawn. `vmax` pins the colorbar to [0, vmax] so faint
    structure isn't squashed by a few hot cells; None auto-scales.
    `member_word`/`count_word`/`count_suffix` set the colorbar + title wording per
    optimizer."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Layouts arrive as (East, North) to match the ENU h5 convention.
        sub = ""
        if surface is not None:
            frame = cliff_frame(mountain)
            s, u = project_en_to_face(surface, aligned_xy[..., 0], aligned_xy[..., 1], frame)
            aligned_xy = np.stack([s, u], axis=-1)               # (K, n_det, 2)
            best_x, best_y = project_en_to_face(
                surface, np.asarray(best_x), np.asarray(best_y), frame)
            mtn_x, mtn_y = enu_to_face(mountain_enu(mountain), frame)
            xlab, ylab = "along-strike [m]", "up-dip [m]"
            sub = (f"cliff face — dip {dip_deg(frame[3]):.1f}°, "
                   f"out-of-plane σ={frame[4]:.0f} m")
        else:
            # Body below is written x=North, y=East; reorder once here.
            aligned_xy = np.stack([aligned_xy[..., 1], aligned_xy[..., 0]], axis=-1)
            best_x, best_y = best_y, best_x
            cen = mountain_nue(mountain)                          # (n, 3) [N, Up, E]
            mtn_x, mtn_y = cen[:, 0], cen[:, 2]
            xlab, ylab = "North [m]", "East [m]"

        K, n_det, _ = aligned_xy.shape
        pts = aligned_xy.reshape(-1, 2)                          # (K*n_det, 2)

        allx = np.concatenate([mtn_x, pts[:, 0]])
        ally = np.concatenate([mtn_y, pts[:, 1]])
        extent = [float(allx.min()), float(allx.max()),
                  float(ally.min()), float(ally.max())]

        rng = [[extent[0], extent[1]], [extent[2], extent[3]]]
        H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=bins, range=rng)
        H = H / max(K, 1)
        try:
            from scipy.ndimage import gaussian_filter
            H = gaussian_filter(H, sigma=1.0)
        except Exception:
            pass

        # Mask cells the mountain does not actually occupy, so the heatmap is
        # drawn only over real surface rather than the whole bounding box.
        occ, _, _ = np.histogram2d(mtn_x, mtn_y, bins=bins, range=rng)
        det_occ, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=bins, range=rng)
        mask = (occ > 0) | (det_occ > 0)
        try:
            from scipy.ndimage import (binary_dilation, binary_fill_holes,
                                       binary_erosion)
            mask = binary_dilation(mask, iterations=2)
            mask = binary_fill_holes(mask)
            mask = binary_erosion(mask, iterations=1, border_value=1)
        except Exception:
            pass
        H = np.ma.masked_array(H, mask=~mask)

        data_ar = (extent[3] - extent[2]) / (extent[1] - extent[0])
        fig_w = 14.0
        fig, ax = plt.subplots(figsize=(fig_w, max(fig_w * data_ar + 1.2, 3.0)))
        cmap = plt.cm.magma.copy()
        cmap.set_bad(alpha=0.0)
        im = ax.imshow(H.T, origin="lower", extent=extent, aspect="equal",
                       cmap=cmap, interpolation="bilinear", zorder=0,
                       vmin=(0.0 if vmax is not None else None), vmax=vmax)
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        cax = make_axes_locatable(ax).append_axes("right", size="2.5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax)
        # Type sized for the figure: this is a 14-inch-wide canvas, so the
        # matplotlib defaults (10 pt labels, 8 pt legend) render unreadably
        # small once the PNG is scaled into a slide or a doc.
        cbar.set_label(f"detector density (count per {member_word} per cell)",
                       fontsize=FS_LABEL)
        cbar.ax.tick_params(labelsize=FS_TICK)

        ax.scatter(np.asarray(best_x), np.asarray(best_y), s=32, c="cyan",
                   edgecolors="black", linewidths=0.4, alpha=0.95, zorder=3,
                   label="best-U layout")
        ax.set_xlabel(xlab, fontsize=FS_LABEL)
        ax.set_ylabel(ylab, fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        title = (f"detector placement density ({count_word}={K}{count_suffix}, "
                 f"{bins}×{bins} bins) + best-U layout")
        ax.set_title(title + (f"\n{sub}" if sub else ""), fontsize=FS_TITLE)
        ax.legend(loc="upper right", fontsize=FS_LEGEND)
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] density heatmap skipped ({exc!r})")


# ── Convergence curves ────────────────────────────────────────────────────────
def plot_curves_de(de_logs, path: str):
    """DE (perturbed-restart ensemble): per-run best-U over DE generations, one
    line per run. Single optimiser → no phase divider / gradient panel."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        K = max(len(de_logs), 1)
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, K))
        for k in range(K):
            lg = de_logs[k] if k < len(de_logs) else []
            u = [e["U"] for e in lg]
            if u:
                best = max(u)
                ax.plot(np.arange(1, len(u) + 1), u, color=colors[k], alpha=0.85,
                        linewidth=1.0, label=f"chain {k}  best={best:.2f}")
        ax.set_xlabel("DE generation")
        ax.set_ylabel("U (composite)")
        ax.set_title(f"Differential evolution: best-U per generation, K={K}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, bbox_to_anchor=(1.04, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] curves skipped ({exc!r})")


def plot_components_de(de_logs, path: str):
    """DE ensemble: one subfigure per chain — weighted utility sub-parts (θ, φ, E)
    over the DE generations plus the overall U (bold black)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        K = max(len(de_logs), 1)
        ncol = min(K, 5)
        nrow = math.ceil(K / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.5 * nrow),
                                 squeeze=False, sharex=False)
        axes_flat = axes.flatten()
        parts = [("θ", "u_theta", "C0"), ("φ", "u_phi", "C1"), ("E", "u_e", "C2")]

        for k in range(K):
            ax = axes_flat[k]
            lg = de_logs[k] if k < len(de_logs) else []
            x = np.arange(1, len(lg) + 1)
            for label, key, col in parts:
                if lg:
                    ax.plot(x, [e[key] for e in lg], color=col, linewidth=1.0,
                            alpha=0.85, label=label)
            if lg:
                ax.plot(x, [e["U"] for e in lg], color="black", linewidth=1.8,
                        label="U (overall)")
            ax.set_title(f"chain {k}", fontsize=10)
            ax.set_xlabel("DE generation"); ax.set_ylabel("utility")
            ax.grid(alpha=0.3); ax.legend(fontsize=7)

        for j in range(K, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle("per-chain utility decomposition "
                     "(weighted θ/φ/E sub-parts + overall U; DE generations)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] utility components skipped ({exc!r})")


def plot_curves_de_pop(de_log, path: str, pop_size: int):
    """DE population (single run over the whole population): best-U over DE
    generations (single trajectory)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        u = [e["U"] for e in de_log]
        if not u:
            print("[plot] curves skipped (empty log)")
            return
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        ax.plot(np.arange(1, len(u) + 1), u, color="C0", linewidth=1.2,
                label=f"best={max(u):.2f}")
        ax.set_xlabel("DE generation")
        ax.set_ylabel("U (composite)")
        ax.set_title(f"Differential evolution: best-U per generation (pop={pop_size})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] curves skipped ({exc!r})")


def plot_components_de_pop(de_log, path: str):
    """DE population: weighted utility sub-parts (θ, φ, E) + overall U over the DE
    generations (single panel)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not de_log:
            print("[plot] utility components skipped (empty log)")
            return
        x = np.arange(1, len(de_log) + 1)
        fig, ax = plt.subplots(figsize=(9, 5))
        for label, key, col in [("θ", "u_theta", "C0"), ("φ", "u_phi", "C1"), ("E", "u_e", "C2")]:
            ax.plot(x, [e[key] for e in de_log], color=col, linewidth=1.0,
                    alpha=0.85, label=label)
        ax.plot(x, [e["U"] for e in de_log], color="black", linewidth=1.8,
                label="U (overall)")
        ax.set_xlabel("DE generation")
        ax.set_ylabel("utility")
        ax.set_title("utility decomposition (weighted θ/φ/E sub-parts + overall U; DE generations)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] utility components skipped ({exc!r})")


def _clip_u_axis(ax, series):
    """Floor the U axis at 0 and report how many points that hides.

    L-BFGS logs CLOSURE calls, so every strong-Wolfe probe is in there — and a
    probe that lands hundreds of km off the mesh pays an off-mesh penalty of
    ~1e5-1e6 U. Those points are rejected by the line search (U is back to its
    pre-excursion value within a handful of closures), but on a shared linear
    axis they compress the +0..+37 band everyone actually reads into one flat
    line. Autoscaling on the positive data only and letting the probes clip
    below the axis is the honest fix: nothing real lives under 0.
    """
    finite = np.concatenate([np.asarray(s, dtype=float) for s in series if len(s)]) \
        if any(len(s) for s in series) else np.zeros(0)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 0
    top = float(finite.max())
    ax.set_ylim(0.0, top * 1.05 if top > 0 else 1.0)
    return int((finite < 0).sum())


def plot_curves_lbfgs(adam_logs, lbfgs_logs, adam_grads, lbfgs_grads, path: str,
                      grad_cos_window: int = 10, cos_smoothed=None):
    """L-BFGS ensemble, two panels: (1) combined Adam→L-BFGS U trajectory, one line
    per run with the SAME color across both phases (Adam solid, L-BFGS dashed) and
    a vertical divider at the switch; (2) consecutive-step gradient cosine
    distance (W=`grad_cos_window`-step vector-averaged; raw drawn faint).

    Panel 1's y axis starts at 0 — see `_clip_u_axis`.

    `cos_smoothed` is for re-plotting from a finished run's optimize_log.json,
    which keeps the already-smoothed cosine series but not the raw gradients:
    pass dict(adam=[per-run series], lbfgs=[...]) with `adam_grads`/`lbfgs_grads`
    empty and panel 2 is drawn from it, without the faint raw underlay."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        K = max(len(adam_logs), len(lbfgs_logs), 1)
        fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
        colors = plt.cm.tab10(np.linspace(0, 1, K))

        # Panel 1 — combined Adam + L-BFGS U, one continuous line per run.
        adam_switch = max((len(lg) for lg in adam_logs), default=0)
        all_u = []
        for k in range(K):
            a_lg = adam_logs[k] if k < len(adam_logs) else []
            l_lg = lbfgs_logs[k] if k < len(lbfgs_logs) else []
            a_u = [e["U"] for e in a_lg]
            l_u = [e["U"] for e in l_lg]
            all_u += [a_u, l_u]
            best = max(a_u + l_u) if (a_u or l_u) else float("nan")
            if a_u:
                axes[0].plot(np.arange(1, len(a_u) + 1), a_u, color=colors[k],
                             alpha=0.85, linewidth=1.0, linestyle="-",
                             label=f"chain {k}  best={best:.2f}")
            if l_u:
                xl = np.arange(adam_switch + 1, adam_switch + 1 + len(l_u))
                axes[0].plot(xl, l_u, color=colors[k], alpha=0.85, linewidth=1.0,
                             linestyle="--",
                             label=None if a_u else f"chain {k}  best={best:.2f}")
        if adam_switch:
            axes[0].axvline(adam_switch + 0.5, color="gray", linestyle=":",
                            alpha=0.7, label="Adam→L-BFGS")
        n_clipped = _clip_u_axis(axes[0], all_u)
        axes[0].set_xlabel("optimizer step (Adam epochs → L-BFGS calls)")
        axes[0].set_ylabel("U (composite)")
        axes[0].set_title(f"optimization: Adam (solid) + L-BFGS (dashed), K={K}"
                          + (f"\ny clipped at 0 — {n_clipped} rejected line-search "
                             f"probes below" if n_clipped else ""),
                          fontsize=11)
        axes[0].grid(alpha=0.3); axes[0].legend(fontsize=7, bbox_to_anchor=(1.04, 1), loc="upper left",)

        # Panel 2 — consecutive-step gradient cosine distance, one line per run.
        # From raw gradient histories when the caller has them (live run), else
        # from the pre-smoothed series a finished run wrote to its log.
        from_raw = bool(adam_grads)
        if from_raw:
            adam_len = max((len(consecutive_cos_distance(g, 1)) for g in adam_grads),
                           default=0)
            runs = [(adam_grads[k], lbfgs_grads[k] if lbfgs_grads else None)
                    for k in range(len(adam_grads))]
        else:
            a_cos = list((cos_smoothed or {}).get("adam") or [])
            l_cos = list((cos_smoothed or {}).get("lbfgs") or [])
            adam_len = max((len(c) for c in a_cos), default=0)
            runs = [(a_cos[k] if k < len(a_cos) else None,
                     l_cos[k] if k < len(l_cos) else None)
                    for k in range(max(len(a_cos), len(l_cos)))]
        any_line = False
        for k, (a_series, l_series) in enumerate(runs):
            for series, x0, dashed in ((a_series, 0, False),
                                       (l_series, adam_len, True)):
                if series is None or not len(series):
                    continue
                if from_raw:
                    raw = consecutive_cos_distance(series, 1)
                    sm = consecutive_cos_distance(series, grad_cos_window)
                    if len(raw):
                        axes[1].plot(np.arange(x0 + 1, x0 + 1 + len(raw)), raw,
                                     color=colors[k % K], alpha=0.1, linewidth=0.7,
                                     linestyle="--" if dashed else "-")
                        any_line = True
                    off = x0 + (len(raw) - len(sm)) // 2 + 1
                else:
                    sm, off = np.asarray(series, dtype=float), x0 + 1
                if len(sm):
                    axes[1].plot(np.arange(off, off + len(sm)), sm,
                                 color=colors[k % K], alpha=0.9, linewidth=1.6,
                                 linestyle="--" if dashed else "-",
                                 label=f"run {k}" if not dashed else None)
                    any_line = True
        if adam_len and any(l is not None for _, l in runs):
            axes[1].axvline(adam_len + 0.5, color="gray", linestyle=":", alpha=0.6,
                            label="Adam→L-BFGS")
        axes[1].set_xlabel("optimizer step")
        axes[1].set_ylabel("cos distance (consecutive grads)")
        axes[1].set_title(f"per-run gradient-direction turn "
                          f"(W={grad_cos_window}-step vector avg"
                          + ("; raw faint)" if from_raw else "; from log)"),
                          fontsize=11)
        axes[1].grid(alpha=0.3)
        if any_line:
            axes[1].legend(fontsize=7)
        else:
            axes[1].text(0.5, 0.5, "no gradient history", ha="center", va="center",
                         transform=axes[1].transAxes, fontsize=10)

        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] curves skipped ({exc!r})")


def plot_components_lbfgs(adam_logs, lbfgs_logs, path: str):
    """L-BFGS ensemble: one subfigure per chain — weighted utility sub-parts
    (θ, φ, E) over the combined Adam→L-BFGS trajectory plus the overall U (bold
    black). Adam phase solid, L-BFGS phase dashed; a vertical divider marks the
    switch."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        K = max(len(adam_logs), len(lbfgs_logs), 1)
        ncol = min(K, 5)
        nrow = math.ceil(K / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.5 * nrow),
                                 squeeze=False, sharex=False)
        axes_flat = axes.flatten()

        # The logged u_* are ALREADY the weighted contributions (W_x * u_x / W_DIV;
        # see utility_of_xy) and sum to U — plot them as-is (weight 1.0).
        parts = [
            ("θ", "u_theta", 1.0, "C0"),
            ("φ", "u_phi",   1.0, "C1"),
            ("E", "u_e",     1.0, "C2"),
        ]
        adam_switch = max((len(lg) for lg in adam_logs), default=0)

        for k in range(K):
            ax = axes_flat[k]
            a_lg = adam_logs[k] if k < len(adam_logs) else []
            l_lg = lbfgs_logs[k] if k < len(lbfgs_logs) else []
            xa = np.arange(1, len(a_lg) + 1)
            xl = np.arange(adam_switch + 1, adam_switch + 1 + len(l_lg))
            ax_series = []

            for label, key, w, col in parts:
                if a_lg:
                    ya = [e[key] * w for e in a_lg]
                    ax_series.append(ya)
                    ax.plot(xa, ya, color=col,
                            linewidth=1.0, linestyle="-", alpha=0.85, label=label)
                if l_lg:
                    yl = [e[key] * w for e in l_lg]
                    ax_series.append(yl)
                    ax.plot(xl, yl, color=col,
                            linewidth=1.0, linestyle="--", alpha=0.85,
                            label=None if a_lg else label)
            if a_lg:
                ya = [e["U"] for e in a_lg]
                ax_series.append(ya)
                ax.plot(xa, ya, color="black",
                        linewidth=1.8, linestyle="-", label="U (overall)")
            if l_lg:
                yl = [e["U"] for e in l_lg]
                ax_series.append(yl)
                ax.plot(xl, yl, color="black",
                        linewidth=1.8, linestyle="--",
                        label=None if a_lg else "U (overall)")
            if adam_switch:
                ax.axvline(adam_switch + 0.5, color="gray", linestyle=":", alpha=0.6)
            n_clipped = _clip_u_axis(ax, ax_series)
            ax.set_title(f"chain {k}" + (f"  ({n_clipped} probes below axis)"
                                         if n_clipped else ""), fontsize=10)
            ax.set_xlabel("optimizer step"); ax.set_ylabel("utility")
            ax.grid(alpha=0.3); ax.legend(fontsize=7)

        for j in range(K, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle("per-chain utility decomposition "
                     "(weighted θ/φ/E sub-parts + overall U; Adam solid, L-BFGS dashed)",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[plot] wrote {path}")
    except Exception as exc:
        print(f"[plot] utility components skipped ({exc!r})")
