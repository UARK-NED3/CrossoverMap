"""Fig. 9: the regime map with the classical transition onset."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import regime_map as F


def fig_regime_map_at(rows, ok, c, onset, onset_label, lims, overlay, overlay_label, out):
    fig, ax = plt.subplots(figsize=(9.0, F.MAP_HEIGHT))
    ax.set_anchor("W")
    good = [r for r, g_ in zip(rows, ok) if g_]
    im, Zp, lims, _ = F._pane_rT(ax, good, c, onset, onset_label, lims, note=False, tau=False)
    if overlay is not None:
        px = np.geomspace(F.PI_LEFT, lims[0][1], 100)
        ax.plot(px, overlay(px), color="0.10", lw=1.8, ls=(0, (5, 3)), zorder=5, label=overlay_label)
    ax.fill([], [], facecolor="0.93", edgecolor="0.55", lw=0.8,
            label="no data: flow > %.0f L/min or outside the sampled designs" % F.Q_CAP)
    F.style.legend_below(fig, ax, ncol=1, row_height=0.36, pad=0.05)
    F._cbar_rT(fig, F._cbar_axes(fig, ax, right_edge=F.MAP_RIGHT), im, Zp)
    F.style.trim_width(fig)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


if __name__ == "__main__":
    F.style.apply_rc()
    on46 = F.cp.band_for_results()[0]
    rows = F.sweep()
    ok = np.array([F.E_OK[0] <= r["lsd"] <= F.E_OK[1] for r in rows])
    tau = np.array([r["tau"] for r in rows]); Pi = np.array([r["Pi"] for r in rows])
    c46 = F.fit_boundary(tau[ok], Pi[ok])
    fig, ax = plt.subplots()
    _, _, lims, _ = F._pane_rT(ax, [r for r, g in zip(rows, ok) if g], c46, on46, "x", None, note=False, tau=False)
    plt.close(fig)
    r23, ok23, c23 = F.run_at_band(F.model.BAND_CLASSICAL)
    out = fig_regime_map_at(r23, ok23, c23, 2300.0, "classical onset", lims, None, None,
                  F.style.out_path("fig09_regime_map_classical_onset.png"))
    print("wrote %s" % out)
