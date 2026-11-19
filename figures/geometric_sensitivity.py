"""Fig. 7: crossover pumping power across the five geometric sweeps."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import contextlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

import model
import model_cal as mc
import manifold
import helpers as cp
import style

OUT = style.out_path("fig07_geometric_sensitivity.png")

CURVE_C = style.MMC_C
DH_LO, DH_HI = 0.5185, 2.0741
N_DH = 48
FLOW = np.linspace(0.5, 150.0, 2000)
CFD_WPSTAR = 9.29

_BASE = model.Geometry()
DH_BASE = _BASE.Dh * 1e3
L_PLATE_MM = _BASE.L_plate * 1e3


ARM_TAPER = 6.0
PITCH_MM = L_PLATE_MM / model.Geometry().N_seg


SLOT_PITCH = 0.6
TAPER = 6.0
ARM_AR_MAX = 4.0
N_ARMS_BUILT = model.Geometry().N_seg

ARM_BUILT_MM = manifold.ARM_H0 * 1e3


W_PLATE_BUILT = _BASE.W_plate * 1e3
L_PLATE_BUILT = L_PLATE_MM


def geom_of(dh_mm, slot_mm, n_arms, L_mm, W_mm):
    s = dh_mm / DH_BASE
    return model.Geometry(w_ch=_BASE.w_ch * s, H_ch=_BASE.H_ch * s,
                          w_fin=_BASE.w_fin * s, N_seg=n_arms,
                          w_slot=slot_mm * 1e-3,
                          L_plate=L_mm * 1e-3, W_plate=W_mm * 1e-3)


def arm_friction_pa(Q, slot_m, n_arms, h0_m, hL_m, fl, L_m):
    V = manifold.v_arm(Q, slot_m, n_arms, h0_m)
    if V <= 0.0:
        return 0.0
    Dh_arm = 2.0 * slot_m * h0_m / (slot_m + h0_m)
    Re = fl.rho * V * Dh_arm / fl.mu
    if Re <= 0.0:
        return 0.0
    C = 0.5 * (manifold.f_Re_rect(min(slot_m, h0_m) / max(slot_m, h0_m))
               + manifold.f_Re_rect(min(slot_m, hL_m) / max(slot_m, hL_m)))
    f_lam = C / Re
    f_turb = 0.316 * Re ** -0.25
    lam = manifold.dP_arm_viscous(Q, slot_m, n_arms, fl.mu, h0_m, hL_m, L=L_m)
    return lam * (f_turb / f_lam)


def crossover(dh_mm, h0_mm, n_arms, L_mm=None, W_mm=None, band=None,
              arm_friction=False):
    fl = model.Fluid()
    L_mm = L_PLATE_BUILT if L_mm is None else L_mm
    W_mm = W_PLATE_BUILT if W_mm is None else W_mm
    slot = SLOT_PITCH * (L_mm / n_arms)
    g = geom_of(dh_mm, slot, n_arms, L_mm, W_mm)
    if band is None:
        band = cp.band_for_results()
    orig = manifold.equivalent_K

    def wrapped(geom, Q, fluid, k_eff, _h0=h0_mm * 1e-3):
        V_ch = Q / (geom.N_seg * geom.N_ch * geom.A_ch)
        if V_ch <= 0.0:
            return 0.0
        _hL = _h0 / TAPER
        dp = manifold.dP_manifold(Q, geom.slot, geom.N_seg, fluid, k_eff,
                                  _h0, _hL, L=geom.W_plate)
        if arm_friction:
            dp += arm_friction_pa(Q, geom.slot, geom.N_seg, _h0, _hL,
                                  fluid, geom.W_plate)
        return dp / (0.5 * fluid.rho * V_ch ** 2)

    manifold.equivalent_K = wrapped
    try:
        Rs, Ws, Rm, Wm = [], [], [], []
        for q in FLOW:
            Q = q / 60000.0
            S = mc.cal_SMC(g, fl, Q, 2500.0, band=band)
            M = mc.cal_MMC(g, fl, Q, 2500.0, endfed=True)
            Rs.append(S["R_th"]); Ws.append(S["W_pump"])
            Rm.append(M["R_th"]); Wm.append(M["W_pump"])
    finally:
        manifold.equivalent_K = orig
    Rs, Ws, Rm, Wm = map(np.asarray, (Rs, Ws, Rm, Wm))
    x = cp._crossover(Ws, Rs, Wm, Rm)
    k = np.where(np.diff(np.sign(Rs - Rm)))[0]
    q_eq = float(np.interp(0.0, [Rs[k[0] + 1] - Rm[k[0] + 1], Rs[k[0]] - Rm[k[0]]],
                           [FLOW[k[0] + 1], FLOW[k[0]]])) if len(k) else np.nan
    return (x[0] if x else np.nan), q_eq


SWEEP_LABEL_FS = 25.0
SWEEP_TEXT_FS = 22.5

def main():
    style.apply_rc()
    plt.rcParams.update({"axes.labelsize": SWEEP_LABEL_FS, "xtick.labelsize": SWEEP_TEXT_FS,
                         "ytick.labelsize": SWEEP_TEXT_FS})

    dh = np.unique(np.concatenate([np.linspace(DH_LO, DH_HI, N_DH), [DH_BASE]]))
    arm = np.unique(np.concatenate([np.linspace(3.0, 14.0, 40), [ARM_BUILT_MM]]))
    narm = np.arange(5, 21)
    wid = np.unique(np.concatenate([np.linspace(20.0, 150.0, 40), [W_PLATE_BUILT]]))
    lng = np.unique(np.concatenate([np.linspace(25.0, 155.0, 27), [L_PLATE_BUILT]]))

    B_DH, B_ARM, B_N = DH_BASE, ARM_BUILT_MM, N_ARMS_BUILT
    _dh_res = [crossover(d, B_ARM, B_N) for d in dh]
    wp_dh = np.array([r[0] for r in _dh_res])
    wp_arm = np.array([crossover(B_DH, h, B_N)[0] for h in arm])
    wp_n = np.array([crossover(B_DH, B_ARM, int(n))[0] for n in narm])
    wp_w = np.array([crossover(B_DH, B_ARM, B_N, W_mm=w)[0] for w in wid])
    wp_l = np.array([crossover(B_DH, B_ARM, max(3, int(round(x / PITCH_MM))),
                               L_mm=x)[0] for x in lng])

    W_BASE = float(crossover(B_DH, B_ARM, B_N)[0])
    fig = plt.figure(figsize=(16.5, 9.1))
    gs = fig.add_gridspec(2, 6, hspace=0.08, wspace=1.35)
    axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]),
            fig.add_subplot(gs[0, 4:6]), fig.add_subplot(gs[1, 1:3]),
            fig.add_subplot(gs[1, 3:5])]

    lseg = L_PLATE_BUILT / narm.astype(float)
    B_LSEG = L_PLATE_BUILT / B_N
    PANES = [
        (axes[0], dh, wp_dh, "Hydraulic diameter, $D_\mathrm{h}$  [mm]", B_DH,
         "arm 12 mm,\n$L_\mathrm{pitch}$ = %g mm" % B_LSEG),
        (axes[1], arm, wp_arm, "Manifold height, $h_0$  [mm]", B_ARM,
         "$D_\mathrm{h}$ = %.2f mm,\n$L_\mathrm{pitch}$ = %g mm" % (B_DH, B_LSEG)),
        (axes[2], lseg, wp_n, "Manifold arm pitch, $L_\mathrm{pitch}$  [mm]",
         B_LSEG, "$D_\mathrm{h}$ = %.2f mm,\narm 12 mm" % B_DH),
        (axes[3], wid, wp_w, "Plate width  [mm]", W_PLATE_BUILT,
         "$D_\mathrm{h}$ = %.2f mm,\narm 12 mm" % B_DH),
        (axes[4], lng, wp_l, "Plate length  [mm]", L_PLATE_BUILT,
         "$L_\mathrm{pitch}$ = %g mm held,\nso $N_\mathrm{arms} = L/%g$" % (PITCH_MM, PITCH_MM)),
    ]
    NOTE_AT = {1: ((0.03, 0.955), "left", "top"), 4: ((0.97, 0.05), "right", "bottom")}
    for ax, x, y, xlab, xb, held in PANES:
        ok = np.isfinite(y)
        ax.plot(x[ok], y[ok], color=CURVE_C, lw=2.4, zorder=3)
        ax.plot([xb], [W_BASE], "o", ms=9, mfc="white", mec=CURVE_C, mew=2.2,
                zorder=5)
        ax.set_xlabel(xlab)
        ax.set_yscale("log")
        lo, hi = float(np.nanmin(y[ok])), float(np.nanmax(y[ok]))
        ax.set_ylim(lo / (2.6 if ax in (axes[2], axes[4]) else 1.45), hi * 1.45)
        ax.yaxis.set_major_locator(
            matplotlib.ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylabel("Crossover $W_\\mathrm{p}^{*}$  [W]")
        ax.grid(alpha=0.20, which="major", lw=0.6)
        (nx, ny), nha, nva = NOTE_AT.get(axes.index(ax), ((0.97, 0.955), "right", "top"))
        ax.annotate(held, xy=(nx, ny), xycoords="axes fraction",
                    ha=nha, va=nva, fontsize=SWEEP_TEXT_FS, color="0.30")

    axes[2].xaxis.set_major_locator(matplotlib.ticker.FixedLocator([2.5, 5.0, 7.5, 10.0]))
    axes[2].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))
    n_cap = ARM_AR_MAX * SLOT_PITCH * (L_PLATE_BUILT / narm) < B_ARM
    if False and n_cap.any():
        axes[2].annotate("arm deeper than 4:1 slot\nbelow $L_\\mathrm{pitch}$ = %g mm"
                         % (L_PLATE_BUILT / (narm[n_cap].min() - 1)),
                         xy=(0.03, 0.06), xycoords="axes fraction",
                         ha="left", va="bottom", fontsize=SWEEP_TEXT_FS, color="0.40")

    _dot = matplotlib.lines.Line2D([], [], ls="none", marker="o", ms=9,
                                   mfc="white", mec=CURVE_C, mew=2.2)
    fig.legend([_dot], ["Baseline design\n(as built):\n"
                        "$D_\\mathrm{h}$ = %.2f mm,\narm %g mm,\n"
                        "$L_\\mathrm{pitch}$ = %g mm,\n%g $\\times$ %g mm"
                        % (B_DH, B_ARM, B_LSEG, L_PLATE_BUILT, W_PLATE_BUILT)],
               loc="center left", bbox_to_anchor=(0.0, 0.0), frameon=False,
               fontsize=SWEEP_TEXT_FS, handlelength=1.0, handletextpad=0.6, borderpad=0.0)

    for ax, tag in zip(axes, "abcde"):
        ax.annotate("(%s)" % tag, xy=(0.0, 1.0), xycoords="axes fraction",
                    xytext=(0, 7), textcoords="offset points",
                    ha="left", va="bottom", fontsize=SWEEP_LABEL_FS)
        style.lock_box(ax)
    fig.canvas.draw()
    _pe = axes[4].get_position()
    fig.legends[-1].set_bbox_to_anchor((_pe.x1 + 0.008, 0.5 * (_pe.y0 + _pe.y1)), transform=fig.transFigure)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    style.apply_rc()

    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
