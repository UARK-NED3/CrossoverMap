"""Shared figure style and output paths."""
import os

import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def out_path(filename):
    os.makedirs(RESULTS, exist_ok=True)
    return os.path.join(RESULTS, filename)

PLOT_BOX_AR = 1.452

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 16,
    "axes.linewidth": 1.1,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
}


ANNOT_FS = 14.5


def trim_width(fig, pad=0.3):
    fig.canvas.draw()
    r = fig.canvas.get_renderer(); dpi = fig.dpi
    W, H = fig.get_size_inches()
    boxes = [ax.get_tightbbox(r) for ax in fig.axes]
    x0 = min(b.x0 for b in boxes) / dpi
    x1 = max(b.x1 for b in boxes) / dpi
    lw = max([lg.get_window_extent(r).width / dpi for lg in fig.legends] or [0.0])
    content = max(x1 - x0, lw)
    new_w = content + 2.0 * pad
    shift = pad + (content - (x1 - x0)) / 2.0 - x0
    pos = [(ax, ax.get_position()) for ax in fig.axes]
    fig.set_size_inches(new_w, H, forward=True)
    for ax, p in pos:
        ax.set_position([(p.x0 * W + shift) / new_w, p.y0, p.width * W / new_w, p.height])
    return new_w


def apply_rc():
    plt.rcParams.update(RC)


def lock_box(ax, ar=PLOT_BOX_AR):
    ax.set_box_aspect(1.0 / ar)


def legend_below(fig, ax, ncol=2, row_height=0.52, pad=0.34, top_pad=0.0, **kw):
    import math

    axes = [ax] if not isinstance(ax, (list, tuple)) else list(ax)
    handles, labels = [], []
    for a in axes:
        h, l = a.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    if not labels:
        return None

    rows = int(math.ceil(len(labels) / float(ncol)))
    extra = rows * row_height + pad
    tighten(fig, top_pad=top_pad, bottom_pad=extra)

    opts = dict(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.004),
                ncol=ncol, columnspacing=1.7, handlelength=2.4,
                handletextpad=0.6, labelspacing=0.45)
    opts.update(kw)
    return fig.legend(handles, labels, **opts)


def tighten(fig, top_pad=0.0, bottom_pad=0.0):
    grow = top_pad + bottom_pad
    w, h = fig.get_size_inches()
    if grow:
        fig.set_size_inches(w, h + grow, forward=True)
    fig.tight_layout(rect=(0.0, bottom_pad / (h + grow) if grow else 0.0,
                           1.0, 1.0 - (top_pad / (h + grow) if grow else 0.0)))

SMC_C = "#286247"
MMC_C = "#BE7960"
ACCENT_C = "#A6B4C4"
ACCENT_DEEP_C = "#4F6D8F"
PEACH_LIGHT_C = "#E7B8A4"
SAGE_C = "#84B39A"
CONTOUR_C = "#5C2E20"

PRESENTATION = os.environ.get("CROSSOVER_PALETTE", "").strip().lower() == "redblue"
if PRESENTATION:
    SMC_C = "#1f77b4"
    MMC_C = "#d62728"
    ACCENT_C = "#98df8a"
    ACCENT_DEEP_C = "#2ca02c"
    PEACH_LIGHT_C = "#f4a582"
    SAGE_C = "#92c5de"
    CONTOUR_C = "#67000d"


def ratio_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    if PRESENTATION:
        return plt.get_cmap("RdBu")
    return LinearSegmentedColormap.from_list(
        "wedding", ["#8F4F3A", "#B86F55", "#C68E79", "#EBCBBD", "#F7F7F7", "#BCD7C9", "#5E9A7D", SMC_C, "#163A2A"], N=256)


def ratio_alt_cmap():
    if PRESENTATION:
        return plt.get_cmap("PuOr")
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "wedding_alt", ["#8F4F3A", "#B86F55", "#C68E79", "#EBCBBD", "#F7F7F7", "#C9D3DE", ACCENT_C, ACCENT_DEEP_C, "#2C3E52"], N=256)
