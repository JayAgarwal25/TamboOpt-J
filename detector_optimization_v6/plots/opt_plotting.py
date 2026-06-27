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
from modules_v6.opt_core import consecutive_cos_distance


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


def plot_ensemble(aligned_xy: np.ndarray,
                  mean_xy: np.ndarray,
                  std_xy: np.ndarray,
                  best_x, best_y,
                  mountain, path: str, surface=None,
                  member_word: str = "run",
                  title_kind: str = "DE ensemble",
                  count_word: str = "K"):
    """Mountain top-down ensemble: every aligned run/member (faint) + per-group
    mean + 1σ ellipses.

    With `surface` (a SurfaceUpMap) the detector East is projected to Up =
    g(North, East) and the plot is the (North, Up) cross section; mean/std are
    recomputed in that plane. Without it the native (North, East) plane is drawn.
    `member_word`/`title_kind`/`count_word` set the legend + title wording so each
    optimizer keeps its own labels."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        from matplotlib.collections import PatchCollection

        if surface is not None:
            up = project_ne_to_up(surface, aligned_xy[..., 0], aligned_xy[..., 1])
            aligned_xy = np.stack([aligned_xy[..., 0], up], axis=-1)   # (K, n_det, 2)=(N,Up)
            mean_xy = aligned_xy.mean(axis=0)
            std_xy  = aligned_xy.std(axis=0)
            best_y  = project_ne_to_up(surface, np.asarray(best_x), np.asarray(best_y))
            mtn_y, ylab, ylet = mountain.centroids_NUE[:, 1], "Up [m]", "Up"
        else:
            mtn_y, ylab, ylet = mountain.centroids_NUE[:, 2], "East [m]", "E"

        K = aligned_xy.shape[0]
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(mountain.centroids_NUE[:, 0], mtn_y,
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
                   label=f"best  (σ̄N={std_xy[:,0].mean():.1f} m, "
                         f"σ̄{ylet}={std_xy[:,1].mean():.1f} m)")

        ax.set_xlabel("North [m]"); ax.set_ylabel(ylab)
        ax.set_aspect("equal")
        ax.set_title(f"{title_kind} ({count_word}={K}) — aligned best + 1σ ellipses")
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
    """Mountain top-down 2D density of detector placements across the ensemble.

    With `surface` the detector East is projected to Up = g(North, East) so the
    plot is the (North, Up) cross section; without it the native (North, East)
    plane is drawn. `vmax` pins the colorbar to [0, vmax] so faint structure isn't
    squashed by a few hot cells; None auto-scales. `member_word`/`count_word`/
    `count_suffix` set the colorbar + title wording per optimizer."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if surface is not None:
            up = project_ne_to_up(surface, aligned_xy[..., 0], aligned_xy[..., 1])
            aligned_xy = np.stack([aligned_xy[..., 0], up], axis=-1)
            best_y = project_ne_to_up(surface, np.asarray(best_x), np.asarray(best_y))
            mtn_col, ylab = 1, "Up [m]"
        else:
            mtn_col, ylab = 2, "East [m]"

        K, n_det, _ = aligned_xy.shape
        pts = aligned_xy.reshape(-1, 2)                          # (K*n_det, 2)

        cen = getattr(mountain, "centroids_NUE", None)
        if cen is not None:
            allx = np.concatenate([cen[:, 0], pts[:, 0]])
            ally = np.concatenate([cen[:, mtn_col], pts[:, 1]])
        else:
            allx, ally = pts[:, 0], pts[:, 1]
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

        if cen is not None:
            occ, _, _ = np.histogram2d(cen[:, 0], cen[:, mtn_col], bins=bins, range=rng)
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
        cbar.set_label(f"detector density (count per {member_word} per cell)")

        ax.scatter(np.asarray(best_x), np.asarray(best_y), s=22, c="cyan",
                   edgecolors="black", linewidths=0.4, alpha=0.95, zorder=3,
                   label="best-U layout")
        ax.set_xlabel("North [m]"); ax.set_ylabel(ylab)
        ax.set_title(f"detector placement density ({count_word}={K}{count_suffix}, "
                     f"{bins}×{bins} bins) + best-U layout")
        ax.legend(loc="upper right", fontsize=8)
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


def plot_curves_lbfgs(adam_logs, lbfgs_logs, adam_grads, lbfgs_grads, path: str,
                      grad_cos_window: int = 10):
    """L-BFGS ensemble, two panels: (1) combined Adam→L-BFGS U trajectory, one line
    per run with the SAME color across both phases (Adam solid, L-BFGS dashed) and
    a vertical divider at the switch; (2) consecutive-step gradient cosine
    distance (W=`grad_cos_window`-step vector-averaged; raw drawn faint)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        K = max(len(adam_logs), 1)
        fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
        colors = plt.cm.tab10(np.linspace(0, 1, K))

        # Panel 1 — combined Adam + L-BFGS U, one continuous line per run.
        adam_switch = max((len(lg) for lg in adam_logs), default=0)
        for k in range(K):
            a_lg = adam_logs[k] if k < len(adam_logs) else []
            l_lg = lbfgs_logs[k] if k < len(lbfgs_logs) else []
            a_u = [e["U"] for e in a_lg]
            l_u = [e["U"] for e in l_lg]
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
        axes[0].set_xlabel("optimizer step (Adam epochs → L-BFGS calls)")
        axes[0].set_ylabel("U (composite)")
        axes[0].set_title(f"optimization: Adam (solid) + L-BFGS (dashed), K={K}")
        axes[0].grid(alpha=0.3); axes[0].legend(fontsize=7, bbox_to_anchor=(1.04, 1), loc="upper left",)

        # Panel 2 — consecutive-step gradient cosine distance, one line per run.
        adam_len = max((len(consecutive_cos_distance(g, 1)) for g in (adam_grads or [])),
                       default=0)
        any_line = False
        for k in range(len(adam_grads or [])):
            for grads, x0, dashed in (
                (adam_grads[k], 0, False),
                (lbfgs_grads[k] if lbfgs_grads else None, adam_len, True),
            ):
                if grads is None:
                    continue
                raw  = consecutive_cos_distance(grads, 1)
                if len(raw):
                    axes[1].plot(np.arange(x0 + 1, x0 + 1 + len(raw)), raw,
                                 color=colors[k % K], alpha=0.1, linewidth=0.7,
                                 linestyle="--" if dashed else "-")
                    any_line = True
                sm = consecutive_cos_distance(grads, grad_cos_window)
                if len(sm):
                    off = x0 + (len(raw) - len(sm)) // 2 + 1
                    axes[1].plot(np.arange(off, off + len(sm)), sm,
                                 color=colors[k % K], alpha=0.9, linewidth=1.6,
                                 linestyle="--" if dashed else "-",
                                 label=f"run {k}" if not dashed else None)
        if adam_len and lbfgs_grads:
            axes[1].axvline(adam_len + 0.5, color="gray", linestyle=":", alpha=0.6,
                            label="Adam→L-BFGS")
        axes[1].set_xlabel("optimizer step")
        axes[1].set_ylabel("cos distance (consecutive grads)")
        axes[1].set_title(f"per-run gradient-direction turn "
                          f"(W={grad_cos_window}-step vector avg; raw faint)")
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

            for label, key, w, col in parts:
                if a_lg:
                    ax.plot(xa, [e[key] * w for e in a_lg], color=col,
                            linewidth=1.0, linestyle="-", alpha=0.85, label=label)
                if l_lg:
                    ax.plot(xl, [e[key] * w for e in l_lg], color=col,
                            linewidth=1.0, linestyle="--", alpha=0.85,
                            label=None if a_lg else label)
            if a_lg:
                ax.plot(xa, [e["U"] for e in a_lg], color="black",
                        linewidth=1.8, linestyle="-", label="U (overall)")
            if l_lg:
                ax.plot(xl, [e["U"] for e in l_lg], color="black",
                        linewidth=1.8, linestyle="--",
                        label=None if a_lg else "U (overall)")
            if adam_switch:
                ax.axvline(adam_switch + 0.5, color="gray", linestyle=":", alpha=0.6)
            ax.set_title(f"chain {k}", fontsize=10)
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
