"""Shared figure functions lifted out of the malata/tau geometry notebooks.

Each function here is the exact plotting code that used to be inline in one
notebook cell (`malata_tau_geometry.ipynb` / `explore_trigger_counts.ipynb`),
parametrized by `figsize` and a few font sizes so the same function renders
both the notebook's on-screen version (defaults match what the cell used to
produce) and a paper-sized version (05_paper_figures.py passes a smaller
figsize + larger fontsize so the text reads correctly once LaTeX scales the
figure back down to its `\\includegraphics` width).

Every legend here is drawn ONCE, at the bottom of the figure via
`fig.legend(...)` (outside every axes' bbox), never per-axes: the plotted
artists still carry their `label=` kwargs (so `get_legend_handles_labels()`
can find them), but nothing calls `ax.legend()` — an in-axes legend sits on
top of the plotted data, which is the opposite of what a legend is for.
`savefig_paper`'s `bbox_inches="tight"` expands the saved canvas to include
the bottom legend rather than clipping it.

Setup/data-loading that a LATER, untouched cell in the same notebook also
depends on (e.g. `rod()` in section 9, `panel()` in explore_trigger_counts'
detector-patterns cell) is deliberately NOT moved in here — it stays inline in
the notebook and is passed to these functions as an argument, so nothing that
another cell needs disappears.

Does not call matplotlib.use(...) — the caller's backend (inline in a
notebook, Agg in a script) is left untouched.
"""
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def tidy_3d_axes(ax, fs_tick=None, fs_label=None, nbins=4):
    """Fewer, evenly-spaced tick numbers on all three 3D axes, with
    tick/label padding that scales with the figure's own font size instead
    of matplotlib's small fixed default. 3D tick labels crowd each other and
    the axis label next to them much faster than 2D ones do -- a fixed pad
    that looked fine at screen size stays that same fixed size once fonts
    are scaled 2-4x for print, so the (now much bigger) numbers run into
    whatever is next to them.

    `fs_tick`/`fs_label` default to the CURRENT rcParams so this still works
    inside a bare `paper_rc` rc_context: mplot3d's z axis doesn't reliably
    pick up the `xtick.labelsize`/`ytick.labelsize` rcParams the way the x/y
    axes do, so every axis is set explicitly here rather than left to
    inherit -- silently keeping z ticks at matplotlib's unscaled default is
    exactly the earlier "legend bigger than title" bug in a different spot.

    The z axis specifically needs much more labelpad than x/y: mplot3d
    offsets the z label by a fixed distance from the axis LINE, not from the
    tick numbers next to it, so at any nontrivial tick fontsize the numbers'
    own width eats into that fixed offset and the label collides with (or at
    print scale, prints on top of / off the edge of the canvas past) the
    tick numbers. x/y labels sit below their (horizontal) tick text and
    don't have this problem.
    """
    if fs_tick is None:
        fs_tick = matplotlib.rcParams.get("xtick.labelsize", 10)
        if not isinstance(fs_tick, (int, float)):
            fs_tick = matplotlib.rcParams["font.size"]
    if fs_label is None:
        fs_label = matplotlib.rcParams.get("axes.labelsize", 12)
        if not isinstance(fs_label, (int, float)):
            fs_label = matplotlib.rcParams["font.size"]
    pad = max(3.0, 0.5 * fs_tick)
    labelpad = max(4.0, 1.0 * fs_label) + 0.4 * fs_tick
    # z is the outlier: unlike x/y, pushing it out much further than this
    # doesn't separate it from the tick numbers -- matplotlib's 3D tight-bbox
    # calculation for the z axis stops tracking it and it gets cropped/lost
    # off the saved canvas entirely instead. Same value as x/y is the largest
    # that reliably stays visible.
    z_labelpad = labelpad
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=nbins))
    for a in ("x", "y", "z"):
        ax.tick_params(axis=a, labelsize=fs_tick, pad=pad)
    ax.xaxis.labelpad = labelpad
    ax.yaxis.labelpad = labelpad
    ax.zaxis.labelpad = z_labelpad


def plot_mountain_3d(tris, tri_up, layouts, det_up, mE, mN, mU, *,
                     figsize=(13, 9), fs_title=14, fs_label=12, fs_tick=10,
                     fs_legend=10, view=(28, -125)):
    """malata_tau_geometry.ipynb section 1b: mountain surface + grid detectors.

    `tris`/`tri_up` are the detector-region triangles (ENU) and their mean Up,
    already rebuilt by the caller (h5py + `_ecef_to_enu`) — kept there rather
    than recomputed here because section 6 (surface normals) reuses the same
    `tris` array as a notebook global."""
    norm = matplotlib.colors.Normalize(tri_up.min(), tri_up.max())
    poly = Poly3DCollection(tris, facecolors=matplotlib.cm.terrain(norm(tri_up)),
                            edgecolors="k", linewidths=0.1, alpha=0.95)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(poly)
    Eg, Ng = layouts["grid"]
    ax.scatter(Eg, Ng, det_up(Ng, Eg) + 110, c="black", s=22, marker="^",
              depthshade=False, label="grid detectors")
    ax.set_xlim(mE.min(), mE.max()); ax.set_ylim(mN.min(), mN.max())
    ax.set_zlim(mU.min(), mU.max())
    ax.set_box_aspect((np.ptp(mE), np.ptp(mN), np.ptp(mU)))
    ax.set_xlabel("East [m]", fontsize=fs_label)
    ax.set_ylabel("North [m]", fontsize=fs_label)
    ax.set_zlabel("Up [m]", fontsize=fs_label)
    tidy_3d_axes(ax, fs_tick, fs_label)
    ax.set_title("Malata mountain (3D)", fontsize=fs_title)
    ax.view_init(elev=view[0], azim=view[1])
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap="terrain"); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label("Up [m]", fontsize=fs_label)
    cbar.ax.tick_params(labelsize=fs_tick)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center",
              fontsize=fs_legend, ncol=1, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    return fig


def plot_tau_arrival_directions(tau_wholesky_path, on_terr, tE_e, tN_n, tU_u,
                                mE, mN, mU, tri_up, add_terrain, *,
                                figsize=(30, 13), fs_title=14, fs_label=12,
                                fs_tick=10, fs_legend=9, r_lim=5000.0,
                                view=(22, -155), seed=0):
    """malata_tau_geometry.ipynb section 2b: tau travel-direction vector field.

    Self-contained (reloads `direction` from `tau_wholesky_path` itself, same
    as the source cell) — no other cell in the notebook depends on values this
    one used to leave behind: the two-panel variant right after it reloads its
    own copy of `tdir`/`dE`/`dN`/`dU`/`idx`/`q` independently."""
    with h5py.File(tau_wholesky_path, "r") as f:
        tdir = f["direction"][...]
    dE, dN, dU = tdir[0], tdir[1], tdir[2]

    idx = np.where(on_terr & (np.abs(tE_e) <= r_lim) & (np.abs(tN_n) <= r_lim))[0]
    qrng = np.random.default_rng(seed)
    q = qrng.choice(idx, size=min(500, idx.size), replace=False)

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(1, 1, 1, projection="3d")
    add_terrain(ax1, alpha=0.45, r_lim=r_lim)
    ax1.quiver(tE_e[q], tN_n[q], tU_u[q], dE[q], dN[q], dU[q],
              length=300.0, normalize=True, linewidth=0.7, arrow_length_ratio=0.45)

    # Wireframe box spanning the mountain's E/N/U extent.
    x0, x1 = mE.min(), mE.max()
    y0, y1 = mN.min(), mN.max()
    z0, z1 = mU.min(), mU.max()
    corners = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    edges = [
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5),
        (2, 3), (2, 6), (3, 7), (4, 5), (4, 6),
        (5, 7), (6, 7),
    ]
    for k, (i, j) in enumerate(edges):
        ax1.plot(*zip(corners[i], corners[j]), c="k", lw=0.8, alpha=0.6,
                label="region of interest" if k == 0 else None)

    ax1.set_xlabel("East [m]", fontsize=fs_label)
    ax1.set_ylabel("North [m]", fontsize=fs_label)
    ax1.set_zlabel("Up [m]", fontsize=fs_label)
    tidy_3d_axes(ax1, fs_tick=fs_tick, fs_label=fs_label)
    ax1.set_title(f"Tau travel directions (n={q.size})", fontsize=fs_title)
    ax1.view_init(elev=view[0], azim=view[1])

    norm = matplotlib.colors.Normalize(tri_up.min(), tri_up.max())
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap="terrain"); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, shrink=0.3, pad=0.08)
    cbar.set_label("Up-component", fontsize=fs_label)
    fig.legend(*ax1.get_legend_handles_labels(), loc="lower center",
              fontsize=fs_legend, ncol=1, bbox_to_anchor=(0.5, -0.02))
    return fig


def plot_showers_overlaid(rod, add_terrain, tE_e, tN_n, tU_u, inband,
                          mE, mN, mU, LOC_R, *,
                          figsize=(15, 6.8), fs_title=13, fs_label=11,
                          fs_tick=9, fs_legend=8, n_pick=40, seed=0):
    """malata_tau_geometry.ipynb section 9: real showers over the local terrain.

    `rod(i)` places shower `i` via the pipeline's own `place_clouds_enu` and
    `add_terrain` draws the local mesh patch — both kept as parameters (not
    redefined here) because section 9b reuses the exact same `rod`/`add_terrain`
    closures the notebook cell defines, so they must stay notebook globals
    there."""
    rng = np.random.default_rng(seed)
    sel = np.where(inband & (np.hypot(tE_e, tN_n) < LOC_R))[0]
    pick = rng.choice(sel, size=min(n_pick, sel.size), replace=False)

    fig = plt.figure(figsize=figsize)
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    add_terrain(ax0, alpha=0.55)
    for i in pick:
        P = rod(i); ax0.plot(P[:, 0], P[:, 1], P[:, 2], lw=0.9, alpha=0.8)
    ax0.scatter(tE_e[pick], tN_n[pick], tU_u[pick], c="red", s=18,
               depthshade=False, label="decay vertex")
    ax0.scatter(mE, mN, mU, c="k", s=4, depthshade=False, label="obs region (1.4 km)")
    ax0.set_xlabel("East [m]", fontsize=fs_label)
    ax0.set_ylabel("North [m]", fontsize=fs_label)
    ax0.set_zlabel("Up [m]", fontsize=fs_label)
    tidy_3d_axes(ax0, fs_tick=fs_tick, fs_label=fs_label)
    ax0.set_box_aspect((1, 1, 0.42))
    ax0.set_title(f"{pick.size} real showers (±{LOC_R/1e3:.0f} km)", fontsize=fs_title)
    ax0.view_init(elev=24, azim=-125)

    ax1 = fig.add_subplot(1, 2, 2)
    for i in pick:
        P = rod(i); ax1.plot(P[:, 0] / 1e3, P[:, 1] / 1e3, lw=0.9, alpha=0.75)
    ax1.scatter(tE_e[pick] / 1e3, tN_n[pick] / 1e3, c="red", s=18, zorder=5,
               label="decay vertex")
    ax1.plot(np.array([mE.min(), mE.max(), mE.max(), mE.min(), mE.min()]) / 1e3,
            np.array([mN.min(), mN.min(), mN.max(), mN.max(), mN.min()]) / 1e3,
            "k-", lw=2.2, zorder=6, label="obs footprint (1.4 km)")
    ax1.set_xlabel("East [km]", fontsize=fs_label)
    ax1.set_ylabel("North [km]", fontsize=fs_label)
    ax1.set_aspect("equal")
    ax1.set_title("Map view", fontsize=fs_title)

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=fs_legend,
              ncol=len(labels), bbox_to_anchor=(0.5, -0.02))
    return fig


def plot_placement_closure(mE_np, mN_np, mU_np, DET, P_dec, D_dir, L_ROD,
                           points, energs, axis_dist, df_scan, axis_crosses,
                           cloud_to_enu, *,
                           figsize=(18, 10.5), fs_suptitle=12,
                           fs_panel_title=9, fs_legend=8, max_candidates=9):
    """explore_trigger_counts.ipynb: placement-closure figure (2x3 panels).

    Everything numeric (axis crossing test, axis->detector distance, the
    df_scan columns) is computed by the caller and passed in — this only draws
    the panels, matching the source cell's `plot_panel`/figure-assembly tail.

    All 6 panels share the same 4-entry legend (mountain / detectors / tau
    axis / decay vertex / placed cloud); it is drawn once, at the bottom of
    the whole figure, not per panel."""
    def plot_panel(ax, s):
        ax.scatter(mE_np, mN_np, mU_np, s=3, c="#DDD6C9", alpha=0.6,
                  depthshade=False, label="mountain")
        ax.scatter(DET[:, 0], DET[:, 1], DET[:, 2], s=12, c="dimgray",
                  marker="^", depthshade=False, label="detectors")

        axis = P_dec[s][None, :] + np.linspace(0, L_ROD, 2)[:, None] * D_dir[s][None, :]
        ax.plot(axis[:, 0], axis[:, 1], axis[:, 2], c="red", lw=1.0, label="tau axis")
        ax.scatter(*P_dec[s], c="red", s=80, marker="*", depthshade=False,
                  label="decay vertex")

        P_cloud = cloud_to_enu(points[s])
        ax.scatter(P_cloud[::8, 0], P_cloud[::8, 1], P_cloud[::8, 2], s=2,
                  c="tab:green", alpha=0.45, depthshade=False, label="placed cloud")

        ax.set_xlabel("East [m]", fontsize=fs_panel_title)
        ax.set_ylabel("North [m]", fontsize=fs_panel_title)
        ax.set_zlabel("Up [m]", fontsize=fs_panel_title)
        tidy_3d_axes(ax, fs_tick=fs_panel_title, fs_label=fs_panel_title)

        ax.set_title(f"#{s}  E={float(energs[s]):.1e} GeV\n"
                    f"d={axis_dist[s]:.0f} m, n={int(df_scan.n_trig[s])}",
                    fontsize=fs_panel_title)
        ax.view_init(elev=25, azim=-125)

    PICK = np.where(axis_crosses)[0][:max_candidates]
    fig, axes = plt.subplots(2, 3, figsize=figsize, subplot_kw=dict(projection="3d"),
                             gridspec_kw={"wspace": 0.4})
    for ax, s in zip(axes.flat, PICK):
        plot_panel(ax, s)

    fig.suptitle("Placement closure", fontsize=fs_suptitle)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=fs_legend,
              ncol=len(labels), bbox_to_anchor=(0.5, -0.02))
    return fig


def plot_detector_patterns(panel, df, df_show, layouts_cache, points,
                           example_strategy, layout_threshold, *,
                           figsize_per_panel=(7.0, 6.2), fs_title=9,
                           fs_legend=7, cols_per=3, rows=(0, 3, 4, 5)):
    """explore_trigger_counts.ipynb: detector patterns for representative
    (shower, layout) pairs.

    `panel(ax, E_arr, e_det, n_det_xy, cloud=..., s=...)` is the notebook's own
    3D-ENU drawing helper — kept as a parameter (not redefined here) because a
    later cell ("same shower, different reinitializations") calls the exact
    same closure, so it must stay a notebook global there.

    Every panel shares the same legend (mountain / shower cloud / no signal /
    signal); it is drawn once, at the bottom of the whole figure.

    `rows` indexes `df_show`, which is built as energy-major over
    (Elow, Emid, Ehigh) x (vert, horiz):

        0 Elow/vert   1 Elow/horiz   2 Emid/vert
        3 Emid/horiz  4 Ehigh/vert   5 Ehigh/horiz

    The default picks four spanning both energy and zenith, filling a 2x2."""
    df_show_tmp = df_show.copy().iloc[list(rows)]
    # Grid sized for what's actually drawn (len(rows)), not len(df_show) --
    # sizing it for the full candidate set left a whole blank row of hidden
    # axes (a visible gap) whenever fewer than df_show's rows were selected.
    rows_per = (len(df_show_tmp) + cols_per - 1) // cols_per
    fig, axes = plt.subplots(
        rows_per, cols_per,
        figsize=(figsize_per_panel[0] * cols_per, figsize_per_panel[1] * rows_per),
        squeeze=False, subplot_kw=dict(projection="3d"),
        # Extra horizontal room: each panel's z-axis label now needs enough
        # labelpad to clear its own tick numbers (see tidy_3d_axes), and the
        # default spacing put that label right on top of the next panel's
        # colorbar.
        gridspec_kw={"wspace": 0.35})

    legend_ax = None
    for ax, (_, row) in zip(axes.flat, df_show_tmp.iterrows()):
        s = int(row["shower"])
        seeds = df[(df["shower"] == s) & (df["strategy"] == example_strategy)].sort_values("n_trig")
        pick_seed = int(seeds.iloc[len(seeds) // 2]["seed"])
        E_arr, e_det, n_det_xy = layouts_cache[(s, example_strategy, pick_seed)]

        sc, n_live, n_over = panel(ax, E_arr, e_det, n_det_xy, cloud=points[s], s=s)
        if sc is not None:
            plt.colorbar(sc, ax=ax, label="counts", shrink=0.55, pad=0.04)
        ax.set_title(f"#{s} ({row['tag']})\n"
                    f"E={row['E']:.1e}, θ={row['theta_deg']:.0f}°, "
                    f"φ={row['phi_deg']:.0f}°, {n_live}/100", fontsize=fs_title)
        if legend_ax is None and ax.get_legend_handles_labels()[1]:
            legend_ax = ax

    for ax in axes.flat[len(df_show_tmp):]:
        ax.set_visible(False)
    if legend_ax is not None:
        handles, labels = legend_ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", fontsize=fs_legend,
                  ncol=len(labels), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=1.2)
    return fig
