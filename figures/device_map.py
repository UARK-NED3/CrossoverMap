"""Fig. 1: published SMC and MMC devices by channel Reynolds number and hydraulic diameter."""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import style

OUT = os.path.join(os.path.dirname(HERE), "results", "fig01_device_map.png")
DATA = os.path.join(os.path.dirname(HERE), "data", "literature", "arch_devices.csv")
ONSET = 4610.0
NU = 0.8e-6
SHOW_VELOCITY = False
COL = {"SMC": "#286247", "MMC": "#BE7960"}
CMP_STYLE = dict(ls=(0, (2.2, 1.4)), lw=3.0)
OPEN = dict(mfc="white", mew=2.2)
REF = {"Tuckerman 1981": "[14]", "Hwang 1987": "[18]", "Qu & Mudawar 2002": "[15]", "Knight 1992": "[17]",
       "Ortega 2022": "[9]", "Pinkus 2026": "[6]", "Ye 2019": "[41]", "Lin 2022": "[42]", "Osman 2021": "[43]",
       "Chen 2022": "[44]", "Li 2025": "[45]", "Harpole 1991": "[23]", "Escher 2010": "[26]",
       "Ryu 2002/2003": "[16], [33]", "Cetegen 2010": "[46]", "van Erp 2020": "[4]", "Boteler 2012": "[37]",
       "Kisitu & Ortega 2023": "[47]", "Bazmi 2026": "[5]", "Collins 2019": "[20]", "Yang 2026": "[10]",
       "Jung 2020": "[27]", "Kermani 2009": "[28]", "Wu 2024": "[38]", "Andhare 2016": "[30]", "Lin 2021": "[39]",
       "Zhou 2025": "[36]", "Kang 2026": "[32]", "Jung 2019": "[13]", "Yuruker 2019": "[35]"}
cite = lambda lab: "%s %s" % (lab, REF[lab])
MARK = {"exp": "o", "cfd": "s", "model": "^"}
FS = 16.5
XLIM, YLIM = (0.008, 1.8e5), (0.014, 12.0)
PLACE = {
    "Harpole1991": (1.08, None), "Escher2010-a": (0.90, None), "Escher2010-b": (1.0, None),
    "Yang2026": (1.0, None), "Jung2020-D8": (1.0, None), "Kermani2009-36": (1.0, None), "Wu2024": (1.0, None),
    "Jung2020-D7": (1.0, None), "Kermani2009-67": (1.0, None), "Andhare2016": (1.0, None), "Lin2021": (0.97, None),
    "Zhou2025": (1.03, None), "Kang2026": (1.0, None), "Jung2019": (0.97, None), "Jung2020-D4": (1.08, None),
    "Yuruker2019": (1.0, None),

    "vanErp2020-25": (1.08, "right"), "vanErp2020-50": (0.90, None), "vanErp2020-100": (0.92, "left"),
    "Ryu2003": (0.96, "right"), "Tuckerman1981": (1.10, "right"), "Boteler2012": (0.96, "skip"),
    "Cetegen2010-P": (1.0, "right"), "Cetegen2010-H": (1.0, "right"),
    "Kisitu": (0.93, "left"), "QuMudawar2002": (0.97, "left"), "Ortega2022": (1.08, "left"), "Knight1992": (1.0, "right-down"),
    "Collins2019": (0.90, "right"), "Bazmi2026-B": (1.0, "left"), "Bazmi2026-O": (1.0, "skip"),
    "Pinkus2026": (1.0, "right"), "Ye2019": (1.0, "right"),
    "Lin2022": (1.03, "left"), "Hwang1987-D": (1.0, "below"),

    "Osman2021": (1.0, "left"), "Chen2022": (0.94, "right"), "Li2025": (1.10, "right"),
}
CLASS_EDGES = (0.2, 3.0)
UPPER_BOUND = {"Ortega2022"}
SPLIT = {"Bazmi2026-B": "Bazmi2026-O"}


def ours():
    import regime_map as F
    rows = F.sweep()
    t, d = [], []
    for r in rows:
        if F.E_OK[0] <= r["lsd"] <= F.E_OK[1]:
            tt = np.asarray(r["traj"][1], float)
            tt = tt[np.isfinite(tt) & (tt * ONSET >= F.RE_FLOOR)]
            if len(tt):
                t += [tt.min(), tt.max()]; d.append(r["dh"])
    return min(t) * ONSET, max(t) * ONSET, min(d), max(d)


ROWS = {}


def draw(ax, r):
    nud, side = PLACE.get(r["key"], (1.0, "right"))
    if side == "skip":
        return
    dh = float(r["dh_mm"])
    assert 0.90 <= nud <= 1.10, (r["key"], nud)
    cls = lambda d: sum(d > e for e in CLASS_EDGES)
    assert cls(dh) == cls(dh * nud), ("nudge moves %s across a class line" % r["key"])
    x0, x1, y = float(r["re_lo"]), float(r["re_hi"]), dh * nud
    cmp_ = r["kind"] == "comparison"
    c = COL[r["verdict"]] if cmp_ else COL[r["arch"]]
    m = MARK[r["method"]]
    line = CMP_STYLE if cmp_ else dict(lw=3.0)
    face = OPEN if cmp_ else dict(mfc=c)
    if r["key"] in SPLIT:
        o = ROWS[SPLIT[r["key"]]]
        xm = (x0 * x1) ** 0.5
        ax.plot([x0, x1], [y, y], color="0.35", lw=1.2, zorder=3)
        for xe in (x0, x1):
            ax.plot([xe, xe], [y / 1.025, y * 1.025], color="0.35", lw=1.2, zorder=3)
        ax.plot(xm, y, m, ms=10, fillstyle="left", mfc=COL[r["verdict"]], mfcalt=COL[o["verdict"]], mec="0.25", mew=0.8,
                zorder=4)
    elif r["key"] in UPPER_BOUND:
        ax.annotate("", xy=(x1 / 3.2, y), xytext=(x1, y), zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2.6, mutation_scale=16, shrinkA=0, shrinkB=0))
        ax.plot(x1, y, m, ms=8, mfc=c, mec=c, zorder=4)
        x0 = x1 / 3.2
    else:
        if x1 > x0 * 1.02:
            ax.plot([x0, x1], [y, y], color=c, solid_capstyle="butt", dash_capstyle="butt", zorder=3, **line)
        for x in ((x0, x1) if x1 > x0 * 1.02 else (x0,)):
            ax.plot(x, y, m, ms=8, mec=c, zorder=4, **face)
    if side is None:
        if r["kind"] == "device" and r["arch"] == "MMC":
            LEADERS.append((cite(r["label"]), x0, y))
        return
    box = dict(boxstyle="square,pad=0.05", fc="white", ec="none") if r["key"] in SPLIT else None
    if side == "above":
        xt, ha, va, off = ((x0 * x1) ** 0.5, "center", "bottom", (0, 6))
    elif side == "below":
        xt, ha, va, off = ((x0 * x1) ** 0.5, "center", "top", (0, -7))
    elif side == "right-up":
        xt, ha, va, off = (x1, "left", "bottom", (6, 3))
    elif side == "right-down":
        xt, ha, va, off = (x1, "left", "top", (6, -3))
    elif side == "left-mid":
        xt, ha, va, off = (x0, "right", "center", (-7, -8))
    else:
        xt, ha, va, off = (x1, "left", "center", (7, 0)) if side == "right" else (x0, "right", "center", (-7, 0))
    ax.annotate(cite(r["label"]), (xt, y), xytext=off, textcoords="offset points", ha=ha, va=va, fontsize=FS, color="0.15",
                zorder=5, bbox=box)


FIG_H = 10.6
LEADERS = []
LEADER_X = 0.30
LEADER_STEP = 1.27


def leader_labels(ax):
    items = sorted(LEADERS, key=lambda t: t[2])
    ly = []
    for _, _, y in items:
        ly.append(max(y, ly[-1] * LEADER_STEP) if ly else y)
    for (lab, x0, y), yl in zip(items, ly):
        ax.plot([LEADER_X * 1.06, x0 / 1.08], [yl, y], color="0.6", lw=0.6, zorder=2)
        ax.text(LEADER_X, yl, lab, ha="right", va="center", fontsize=FS, color="0.15", zorder=5)


CLASSES = [(0.010, 0.200, "Microchannels\nmicroelectronics"),
           (0.200, 3.0, "Minichannels\ncompact HX,\nautomotive, aerospace"),
           (3.0, 100.0, "Conventional")]
CLASS_C = ("#ebebeb", "#dadada", "#c8c8c8")


def class_bar(axb):
    for (lo, hi, lab), c in zip(CLASSES, CLASS_C):
        axb.axhspan(lo, hi, color=c, lw=0)
        y0, y1 = max(lo, YLIM[0]), min(hi, YLIM[1])
        axb.text(0.5, (y0 * y1) ** 0.5, lab, rotation=90, ha="center", va="center", fontsize=FS, color="0.15",
                 linespacing=1.15)
        if YLIM[0] < lo < YLIM[1]:
            axb.axhline(lo, color="0.35", lw=0.8)
    axb.set_xlim(0, 1)
    axb.set_xticks([])


def main():
    style.apply_rc()
    fig, (axb, ax) = plt.subplots(1, 2, figsize=(11.8, FIG_H), sharey=True,
                                  gridspec_kw=dict(width_ratios=[0.105, 1.0], wspace=0.0))
    class_bar(axb)
    ax.axvspan(2300, 1e4, color="0.93", zorder=0)
    ax.axvspan(1e4, XLIM[1], color="0.86", zorder=0)
    ax.text(4800, YLIM[0] * 1.15, "transitional", ha="center", va="bottom", rotation=90, fontsize=FS, color="0.30")
    ax.text(3.3e4, YLIM[0] * 1.15, "turbulent", ha="center", va="bottom", fontsize=FS, color="0.30")
    if SHOW_VELOCITY:
        for V in (0.5, 2.0, 5.0):
            re = np.array(XLIM)
            ax.plot(re, re * NU / V * 1e3, color="0.45", ls="--", lw=1.1, zorder=1)
            rt = YLIM[1] / 1.25 * 1e-3 * V / NU
            ax.text(rt, YLIM[1] / 1.25, "%g m/s" % V, ha="right", va="bottom", fontsize=11.5, color="0.35",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8), zorder=2)
    r0, r1, d0, d1 = ours()
    ax.add_patch(Rectangle((r0, d0), r1 - r0, d1 - d0, fill=False, edgecolor="black", ls="--", lw=2.0, zorder=2))
    ax.text(r0, d1 * 1.04, "This work", ha="left", va="bottom", fontsize=FS + 1, color="black", zorder=6)
    rows = list(csv.DictReader(open(DATA, encoding="utf-8")))
    ROWS.update({r["key"]: r for r in rows})
    for r in rows:
        draw(ax, r)
    leader_labels(ax)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_xticks([1, 10, 100, 1e3, 1e4, 1e5])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "{:,.0f}".format(v)))
    ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))
    for a in (ax.xaxis, ax.yaxis):
        a.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Channel Reynolds number, $Re_\\mathrm{ch}$")
    axb.set_ylabel("Channel hydraulic diameter, $D_\\mathrm{h}$ (mm)")
    ax.tick_params(axis="y", which="both", labelleft=False)
    axb.set_title("channel\nclass", fontsize=FS, color="0.30")
    h = [Line2D([], [], color=COL["SMC"], lw=3, label="Straight channel (SMC)"),
         Line2D([], [], color=COL["MMC"], lw=3, label="Manifold (MMC)"),
         Line2D([], [], color=COL["MMC"], marker="o", ms=8, label="Comparison: manifold better", **CMP_STYLE, **OPEN),
         Line2D([], [], color=COL["SMC"], marker="o", ms=8, label="Comparison: straight channel better", **CMP_STYLE,
                **OPEN),
         Patch(fill=False, edgecolor="black", ls="--", lw=2.0, label="This work"),
         Line2D([], [], ls="none", marker="o", ms=9, mfc="0.4", mec="0.4", label="Experiment"),
         Line2D([], [], ls="none", marker="s", ms=9, mfc="0.4", mec="0.4", label="CFD"),
         Line2D([], [], ls="none", marker="^", ms=9, mfc="0.4", mec="0.4", label="Analytical")]
    if SHOW_VELOCITY:
        h.append(Line2D([], [], color="0.45", ls="--", lw=1.1, label="Constant channel velocity (water)"))
    fig.tight_layout()
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=FS, bbox_to_anchor=(0.53, 0.0),
               columnspacing=1.2, handlelength=2.0, handletextpad=0.5)
    fig.subplots_adjust(bottom=1.5 / FIG_H)
    fig.savefig(OUT, dpi=200)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
