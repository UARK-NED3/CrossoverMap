"""Fig. 3: SMC transition band from the CFD pressure-drop and thermal-resistance slopes."""
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import model, refit_cal as rc
import style

OUT = style.out_path("fig03_transition_band.png")

BLUE, RED, GREY = style.SMC_C, style.MMC_C, "0.45"

plt.rcParams.update({
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
})

geom, fluid = model.Geometry(), model.Fluid()
rows = rc.load(rc.CANON["smc"])
Q = np.array([r[0] for r in rows])
R = np.array([r[1] for r in rows])
P = np.array([r[2] for r in rows])


def Re_of(q):
    return model.model_SMC(geom, fluid, q / 60000.0, 100.0,
                           band=model.BAND_MEASURED)["Re"]


lq = np.log(Q)
slope_P = np.diff(np.log(P)) / np.diff(lq)
slope_R = np.diff(np.log(R)) / np.diff(lq)
Re_mid = np.array([Re_of(q) for q in np.sqrt(Q[:-1] * Q[1:])])

X_LO, X_HI = 300.0, 20000.0
fig, ax = plt.subplots(figsize=(9.0, 6.2))

ax.axvspan(model.RE_LAM_END, model.RE_TURB_START, color="0.55", alpha=0.28,
           lw=0, zorder=0)

for y, txt, x, ha, dy, va in (
        (1.75, "Blasius, $\Delta P \propto \mathrm{Re}^{1.75}$", X_LO * 1.12, "left",
         0.10, "bottom"),
        (1.0, "Laminar, $\Delta P \propto \mathrm{Re}$", X_HI * 0.94, "right",
         -0.10, "top")):
    ax.axhline(y, color=GREY, ls=":", lw=1.3, zorder=1)
    ax.text(x, y + dy, txt, ha=ha, va=va, fontsize=14, color=GREY)

ax.plot(Re_mid, slope_P, "o-", color=BLUE, ms=7, lw=1.9, mec="white", mew=1.0,
        zorder=3, label="$d\ln(\Delta P)\;/\;d\ln \mathrm{Re}$")
ax.plot(Re_mid, slope_R, "s-", color=RED, ms=6.5, lw=1.9, mec="white", mew=1.0,
        zorder=3, label="$d\ln R_\mathrm{th}\;/\;d\ln \mathrm{Re}$")

ax.set_xscale("log")
ax.set_xlim(X_LO, X_HI)
ax.set_xticks([300, 1000, 3000, 10000, 20000])
ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("Channel Reynolds number, $\mathrm{Re_{ch}}$")

ax.set_ylim(-3.6, 2.9)
ax.set_yticks([-3, -2, -1, 0, 1, 2])
ax.set_ylabel("Local logarithmic slope")

ax.text(np.sqrt(model.RE_LAM_END * model.RE_TURB_START), 2.80,
        "Transition\nband", ha="center", va="top", fontsize=15, color="0.40",
        linespacing=1.15)

ax.grid(alpha=0.20, which="major", lw=0.6)

style.lock_box(ax)
style.legend_below(fig, ax, ncol=2)
fig.savefig(OUT, dpi=300)
print("wrote %s" % OUT)
