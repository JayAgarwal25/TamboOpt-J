"""Font-size math for dropping this repo's figures into an Elsevier LaTeX paper.

Every figure in this codebase is drawn OVERSIZED on purpose — see
`opt_plotting.py`'s own FS_* comment: a 14-inch canvas read on screen at full
size needs ~14-20 pt titles, not matplotlib's 10 pt default. `\\includegraphics`
then shrinks that same PNG down to a fraction of `\\textwidth` for print, and
the text shrinks by the exact same linear factor as the artwork.

`paper_fontsize` inverts that shrink: given the figure's DRAWN width (the
figsize width already tuned for on-screen legibility — unchanged here) and the
`\\textwidth` fraction it will be placed at, it returns the matplotlib fontsize
that lands at `target_pt` once printed. Figsize/layout are never touched, only
fontsize — so the legend placement, subplot spacing, and colorbar sizing every
figure was already tuned for stays intact; only the type scale moves.

Edit TEXTWIDTH_PT below to match the Elsevier document this feeds. Get the
real value by putting `\\the\\textwidth` in the .tex body, compiling, and
reading it off the log/output (a fractional-width figure — `0.5\\textwidth`
— still scales off the FULL \\textwidth, not the half).
"""

PT_PER_IN = 72.27   # TeX/LaTeX point (NOT the 72 pt/in of a "printer's point")

# elsarticle two-column ("5p"/"3p") default. Preprint/"1p" single-column mode
# uses \textwidth=384.1pt instead -- swap this if that's the class option in
# use. Override per-call via the `textwidth_pt` argument if you don't want to
# edit this file (e.g. from a one-off script).
TEXTWIDTH_PT = 469.75

# Apparent size [pt] each figure element should read at in the printed PDF.
# Elsevier body text is ~9-10 pt; figure type is conventionally a notch below
# that so it doesn't compete with the prose. Strictly decreasing on purpose --
# legend text is auxiliary and must never be the biggest thing on the page,
# which it silently became when axis labels/ticks had no explicit fontsize
# (they rode matplotlib's fixed ~10 pt default while everything else here got
# scaled up 2-4x to survive the print shrink). `paper_rc` below applies this
# same hierarchy to those previously-unscaled elements too.
TARGET_TITLE_PT       = 9.5   # fig.suptitle
TARGET_PANEL_TITLE_PT = 8.5   # per-axes ax.set_title
TARGET_LABEL_PT       = 7.5   # ax.set_xlabel / set_ylabel / set_zlabel
TARGET_LEGEND_PT      = 6.5   # legend entries -- below label, above tick
TARGET_TICK_PT        = 6.0   # tick numbers

# Multi-panel grids (2x3 / 1x3 of small 3D subplots): each panel only gets
# ~1/3 of the figure's final on-page width, so the same absolute target sizes
# above overflow a single panel's title/label text. Same hierarchy, smaller.
GRID_TITLE_PT       = 7.0
GRID_PANEL_TITLE_PT = 4.8
GRID_LABEL_PT       = 5.0
GRID_LEGEND_PT      = 5.5     # legend is now figure-level, shared by all panels
GRID_TICK_PT        = 4.0

DEFAULT_DPI = 400   # print-quality raster; a vector save (pdf/svg) ignores this


def textwidth_in(frac: float = 1.0, textwidth_pt: float = TEXTWIDTH_PT) -> float:
    """On-page width [inches] once placed via `\\includegraphics[width=frac\\textwidth]`."""
    return frac * textwidth_pt / PT_PER_IN


def paper_fontsize(target_pt: float, drawn_width_in: float, frac: float = 1.0,
                   textwidth_pt: float = TEXTWIDTH_PT) -> float:
    """matplotlib fontsize so `target_pt` is the apparent size in print.

    `drawn_width_in` is the figure's OWN figsize width (inches) at the size it
    is actually being rendered — e.g. 13.0 for `plot_mountain_3d`'s default
    figsize=(13, 9), or 7.0*cols_per for `plot_detector_patterns`. Do not pass
    the paper's column width here; that's `frac`/`textwidth_pt`'s job.
    """
    return target_pt * drawn_width_in / textwidth_in(frac, textwidth_pt)


def paper_fontsizes(drawn_width_in: float, frac: float = 1.0,
                    textwidth_pt: float = TEXTWIDTH_PT,
                    title_pt: float = TARGET_TITLE_PT,
                    label_pt: float = TARGET_LABEL_PT,
                    tick_pt: float = TARGET_TICK_PT,
                    legend_pt: float = TARGET_LEGEND_PT) -> dict:
    """The four common FS_* values at once, as a dict — matches the
    `fs_title`/`fs_label`/`fs_tick`/`fs_legend` kwarg names `geometry_plots.py`
    and `opt_plotting.py`'s FS_* constants already use."""
    args = (drawn_width_in, frac, textwidth_pt)
    return dict(
        fs_title=paper_fontsize(title_pt, *args),
        fs_label=paper_fontsize(label_pt, *args),
        fs_tick=paper_fontsize(tick_pt, *args),
        fs_legend=paper_fontsize(legend_pt, *args),
    )


def paper_rc(drawn_width_in, frac: float = 1.0, textwidth_pt: float = TEXTWIDTH_PT,
            title_pt: float = TARGET_TITLE_PT, panel_title_pt: float = TARGET_PANEL_TITLE_PT,
            label_pt: float = TARGET_LABEL_PT, legend_pt: float = TARGET_LEGEND_PT,
            tick_pt: float = TARGET_TICK_PT) -> dict:
    """rcParams dict for `matplotlib.rc_context(rc=paper_rc(...))` around a
    figure-drawing call. Fills in the same paper-scaled hierarchy for every
    element that ISN'T given an explicit fontsize at the call site (axis
    xlabel/ylabel/zlabel, tick numbers, and any bare `ax.legend()`/
    `fig.legend()`) -- explicit fontsize kwargs elsewhere always win over
    rcParams, so this only closes the gap, it never fights the fs_* args
    already threaded through geometry_plots.py / opt_plotting.py."""
    args = (drawn_width_in, frac, textwidth_pt)
    return {
        "figure.titlesize": paper_fontsize(title_pt, *args),
        "axes.titlesize": paper_fontsize(panel_title_pt, *args),
        "axes.labelsize": paper_fontsize(label_pt, *args),
        "legend.fontsize": paper_fontsize(legend_pt, *args),
        "xtick.labelsize": paper_fontsize(tick_pt, *args),
        "ytick.labelsize": paper_fontsize(tick_pt, *args),
    }


def savefig_paper(fig, path, dpi: int = DEFAULT_DPI, **kwargs):
    """`fig.savefig` with paper-appropriate defaults (tight bbox, print dpi).
    `bbox_inches="tight"` is what lets a `fig.legend(...)` placed below the
    axes (outside their bbox) expand the saved canvas instead of overlapping
    the bottom row of subplots.

    `pad_inches` defaults generous (not matplotlib's usual ~0.1) because
    `Axes3D.get_tightbbox()` doesn't reliably measure a 3D z-axis label's
    true rendered extent -- the "tight" box it hands back can undershoot
    where that label actually lands, and undershoot cuts it off outright
    rather than just leaving too little whitespace. A caller with no 3D
    content can pass a smaller `pad_inches` explicitly if the extra margin
    is unwanted."""
    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("pad_inches", 1.5)
    fig.savefig(path, dpi=dpi, **kwargs)
    print(f"[paper] wrote {path}")


def savefig_paper_formats(fig, path_no_ext, formats=("png",), dpi: int = DEFAULT_DPI, **kwargs):
    """`savefig_paper` for one or more extensions off the same basename, e.g.
    `savefig_paper_formats(fig, ".../mountain_3d", formats=("png", "pdf"))`
    writes both `mountain_3d.png` and `mountain_3d.pdf`. PDF is vector (dpi is
    ignored for it) except where the figure rasterizes internally (hexbin)."""
    for ext in formats:
        savefig_paper(fig, f"{path_no_ext}.{ext}", dpi=dpi, **kwargs)
