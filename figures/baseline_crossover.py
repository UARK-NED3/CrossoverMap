"""Fig. 6(a): baseline thermal resistance versus pumping power, CFD and ROM."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

import helpers as cp
import style

OUT = style.out_path("fig06a_baseline_crossover.png")

SC_C, MC_C = style.SMC_C, style.MMC_C

GRID = np.linspace(0.5, 60.0, 250)

CFD_WPSTAR = 9.29
BAND_Q = (15.0, 18.0)
STRIP_Q = (10.0, 17.5, 20.0)
STRIP_TXT = ((1.9, 19.5), (8.0, 12.5), (14.0, 12.5))


def main(strips=True, legend_inside=False, out=None):
    Rs, _, Ws = cp.eval_model("SC", GRID, calibrated=True)
    Re, _, We = cp.eval_model("MC", GRID, calibrated=True, endfed=True)

    xo = cp._crossover(Ws, Rs, We, Re)
    Qs_c, Rs_c, dPs_c = cp.load_cfd("smc_transition_sst_medium.csv")
    Qe_c, Re_c, dPe_c = cp.load_cfd("mmc_endfed_sst_medium.csv")
    Ws_c, We_c = (Qs_c / 60000.0) * dPs_c, (Qe_c / 60000.0) * dPe_c
    xc = cp._crossover(Ws_c, Rs_c, We_c, Re_c)
    assert abs(xc[0] - CFD_WPSTAR) < 0.01, "CFD crossover moved: %.3f W" % xc[0]
    assert all(abs(Qs_c[np.argmin(abs(Qs_c - q))] - q) < 1e-6 for q in BAND_Q), "band flows not in the CFD sweep"
    Rs, Re, Rs_c, Re_c = Rs * 1e3, Re * 1e3, Rs_c * 1e3, Re_c * 1e3

    style.apply_rc()
    fig, ax = plt.subplots(figsize=(9.0, 6.2))

    XLIM = (1e-1, 1e2)
    XTICKS = [1e-1, 1e0, 1e1, 1e2]

    xin = lambda W: (W >= XLIM[0]) & (W <= XLIM[1])
    ax.plot(Ws_c[xin(Ws_c)], Rs_c[xin(Ws_c)], ls="none", marker="o", ms=6.5, mfc="white", mec=SC_C, mew=1.5,
            zorder=4, label="SMC — CFD")
    ax.plot(Ws, Rs, color=SC_C, lw=2.2, ls="-", zorder=3, label="SMC — ROM")
    ax.plot(We_c[xin(We_c)], Re_c[xin(We_c)], ls="none", marker="s", ms=6.0, mfc="white", mec=MC_C, mew=1.5,
            zorder=4, label="MMC — CFD")
    ax.plot(We, Re, color=MC_C, lw=2.2, ls="-", zorder=3, label="MMC — ROM")

    sbox = dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", lw=0.8)
    lead = dict(arrowstyle="-", color="0.40", lw=0.9, shrinkA=0, shrinkB=5)
    for q, lab, txy in zip(STRIP_Q, ("a", "b", "c"), STRIP_TXT) if strips else ():
        i = int(np.argmin(abs(Qs_c - q)))
        assert abs(Qs_c[i] - q) < 1e-6, "strip flow %.1f L/min not in the CFD sweep" % q
        ax.plot(Ws_c[i], Rs_c[i], ls="none", marker="o", ms=7.5, mfc=SC_C, mec="white", mew=1.0, zorder=8)
        ax.annotate(lab, xy=(Ws_c[i], Rs_c[i]), xytext=txy, textcoords="data", ha="center", va="center",
                    fontsize=13, color="0.15", bbox=sbox, arrowprops=lead, zorder=6)

    box = dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", lw=0.8)
    arrow = dict(arrowstyle="-|>", color="0.25", lw=1.1, shrinkA=2, shrinkB=6, mutation_scale=12)
    for (w, r), lab, txy, mk in (
            (xo, "ROM\n%.2f W" % xo[0], (2.6, 4.6), dict(marker="o", mfc="0.15", ms=7, zorder=7)),
            (xc, "CFD\n%.2f W" % xc[0], (30.0, 11.0), dict(marker="D", mfc="white", ms=9, zorder=6))):
        ax.plot(w, r * 1e3, ls="none", mec="0.15", mew=1.4, **mk)
        ax.annotate(lab, xy=(w, r * 1e3), xytext=txy, textcoords="data", ha="center", va="center",
                    fontsize=13, color="0.15", bbox=box, arrowprops=arrow, zorder=7)

    parts = []
    for W, R in ((Ws, Rs), (We, Re), (Ws_c, Rs_c), (We_c, Re_c)):
        m = (W >= XLIM[0]) & (W <= XLIM[1]) & np.isfinite(R)
        if m.any():
            parts.append(R[m])
    allr = np.concatenate(parts)
    ylo, yhi = allr.min(), allr.max()
    pad = 0.08 * (yhi - ylo)
    YLIM = (ylo - pad, yhi + pad)

    ax.set_xscale("log")
    ax.set_xlim(*XLIM)
    ax.set_xticks(XTICKS)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(*YLIM)
    ax.set_yticks([t for t in range(0, 100, 5) if YLIM[0] <= t <= YLIM[1]])
    ax.set_xlabel("Pumping power, $W_\\mathrm{p}$  [W]")
    ax.set_ylabel("Thermal resistance, $R_\\mathrm{th}$  [mK/W]")
    ax.grid(alpha=0.20, which="major", lw=0.6)

    style.lock_box(ax)
    if legend_inside:
        ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="0.75", fancybox=True)
        fig.tight_layout()
    else:
        style.legend_below(fig, ax, ncol=2)
    out = out or OUT
    fig.savefig(out, dpi=300)
    plt.close(fig)

    print("wrote %s" % out)


if __name__ == "__main__":
    main()
