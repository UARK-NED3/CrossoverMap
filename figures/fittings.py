"""Fig. B.1: baseline crossover with inlet and outlet fitting losses."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

import geometric_sensitivity as F
import model
import model_cal as mc
import manifold
import helpers as cp
import style

OUT = style.out_path("figB1_fittings.png")
BORE = np.geomspace(4.0, 25.0, 60)
K_MAIN, K_BAND = 1.5, (0.5, 3.0)
NOMINAL = (("¼″", 6.35, "1/4in"), ("⅜″", 9.53, "3/8in"), ("½″", 12.7, "1/2in"),
           ("¾″", 19.05, "3/4in"))


def curves(dh_mm, h0_mm, n_arms, L_mm=None, W_mm=None, band=None):
    fl = model.Fluid()
    L_mm = F.L_PLATE_BUILT if L_mm is None else L_mm
    W_mm = F.W_PLATE_BUILT if W_mm is None else W_mm
    slot = F.SLOT_PITCH * (L_mm / n_arms)
    g = F.geom_of(dh_mm, slot, n_arms, L_mm, W_mm)
    orig = manifold.equivalent_K

    def wrapped(geom, Q, fluid, k_eff, _h0=h0_mm * 1e-3):
        V_ch = Q / (geom.N_seg * geom.N_ch * geom.A_ch)
        if V_ch <= 0.0:
            return 0.0
        dp = manifold.dP_manifold(Q, geom.slot, geom.N_seg, fluid, k_eff,
                                  _h0, _h0 / F.TAPER, L=geom.W_plate)
        return dp / (0.5 * fluid.rho * V_ch ** 2)

    manifold.equivalent_K = wrapped
    try:
        Ws, Rs, Wm, Rm, Re = [], [], [], [], []
        for q in F.FLOW:
            Q = q / 60000.0
            S = mc.cal_SMC(g, fl, Q, 2500.0, band=band)
            M = mc.cal_MMC(g, fl, Q, 2500.0, endfed=True)
            Ws.append(S["W_pump"]); Rs.append(S["R_th"]); Re.append(S["Re"])
            Wm.append(M["W_pump"]); Rm.append(M["R_th"])
    finally:
        manifold.equivalent_K = orig
    return tuple(map(np.asarray, (Ws, Rs, Wm, Rm, Re))) + (g.R_base,)


def crossing(c_fit, cv):
    Ws, Rs, Wm, Rm, Re, _ = cv
    Q = F.FLOW / 60000.0
    wf = c_fit * Q ** 3
    x = cp._crossover(Ws + wf, Rs, Wm + wf, Rm)
    if not x:
        return (np.nan,) * 4
    lq = np.log(Q)
    qs = float(np.exp(np.interp(np.log(x[0]), np.log(Ws + wf), lq)))
    qm = float(np.exp(np.interp(np.log(x[0]), np.log(Wm + wf), lq)))
    return x[0], qs * 60000.0, qm * 60000.0, float(np.interp(qs * 60000.0, F.FLOW, Re))


def main(out=None):
    out = out or OUT
    style.apply_rc()
    band = cp.band_for_results()
    fl = model.Fluid()
    cv = curves(F.DH_BASE, F.ARM_BUILT_MM, F.N_ARMS_BUILT, band=band)
    coef = lambda d_mm, K: K * fl.rho / (2.0 * (np.pi * (d_mm * 1e-3) ** 2 / 4.0) ** 2)
    base = crossing(0.0, cv)
    q_eq = float(F.crossover(F.DH_BASE, F.ARM_BUILT_MM, F.N_ARMS_BUILT)[1])
    re_eq = float(np.interp(q_eq, F.FLOW, cv[4]))
    w_baseline = float(F.crossover(F.DH_BASE, F.ARM_BUILT_MM, F.N_ARMS_BUILT)[0])
    assert abs(base[0] - w_baseline) < 1e-6 * w_baseline, (base[0], w_baseline)

    rows = {K: np.array([crossing(coef(d, K), cv) for d in BORE]) for K in (K_BAND[0], K_MAIN, K_BAND[1])}
    allk = np.stack([rows[k] for k in rows])

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(9.0, 12.0), sharex=True)
    for ax, j, ylab in ((axA, 0, "Crossover pumping power, plate + fittings  [W]"),
                        (axB, 3, "Crossover $Re_\\mathrm{SC}$")):
        ax.fill_between(BORE, allk[:, :, j].min(axis=0), allk[:, :, j].max(axis=0), color=F.CURVE_C,
                        alpha=0.15, lw=0, zorder=1, label="fitting loss coefficient $K$ = %g–%g" % K_BAND)
        ax.plot(BORE, rows[K_MAIN][:, j], color=F.CURVE_C, lw=2.4, zorder=4,
                label="$K$ = %g (entry 0.5 + exit 1.0)" % K_MAIN)
        ax.axhline(base[j], color="0.15", lw=1.4, ls="--", zorder=2, label="no fittings (the selection chart)")
        for name, d, _ in NOMINAL:
            ax.axvline(d, color="0.6", lw=1.0, ls=":", zorder=0)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.20, which="major", lw=0.6)
    for name, d, _ in NOMINAL:
        axA.annotate(name, xy=(d, 0.97), xycoords=("data", "axes fraction"), ha="right", va="top",
                     xytext=(-3, 0), textcoords="offset points", fontsize=11, color="0.30")
    axA.set_yscale("log")
    axA.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    axA.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))
    axA.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    pad = 0.12 * (base[3] - re_eq)
    axB.set_ylim(re_eq - 2 * pad, base[3] + 2 * pad)
    axB.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "{:,.0f}".format(v)))
    axB.set_xscale("log")
    axB.set_xlim(BORE[0], BORE[-1])
    axB.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([4, 5, 6, 8, 10, 15, 20, 25]))
    axB.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))
    axB.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axB.set_xlabel("Fitting bore, both devices  [mm]")
    axB.annotate("$D_\\mathrm{h}$ = %.2f mm, arm %g mm, $N_\\mathrm{arms}$ = %d, %g $\\times$ %g mm"
                 % (F.DH_BASE, F.ARM_BUILT_MM, F.N_ARMS_BUILT, F.L_PLATE_BUILT, F.W_PLATE_BUILT),
                 xy=(0.97, 0.04), xycoords="axes fraction", ha="right", va="bottom",
                 fontsize=11.5, color="0.30")
    for ax, lab in ((axA, "(a)"), (axB, "(b)")):
        ax.text(0.0, 1.02, lab, transform=ax.transAxes, ha="left", va="bottom", fontsize=20)
        ax.set_anchor("W")
        style.lock_box(ax)
    style.legend_below(fig, [axA, axB], ncol=1)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    wmid = rows[K_MAIN][:, 0]
    i_min = int(np.nanargmin(wmid))

    table = []
    for _, d, name in NOMINAL:
        for K in (K_BAND[0], K_MAIN, K_BAND[1]):
            w, qs, qm, re = crossing(coef(d, K), cv)
            share = coef(d, K) * (qm / 60000.0) ** 3 / w * 100.0
            table.append((name, d, K, w, qs, qm, qm / qs, re,
                          100.0 * (re / base[3] - 1.0), share))
    print("wrote %s" % out)
    return dict(out=out, bore=BORE, rows=rows, base=base, re_eq=re_eq, q_eq=q_eq, w_baseline=w_baseline, table=table,
                w_min=(float(wmid[i_min]), float(BORE[i_min])))


if __name__ == "__main__":
    main()
