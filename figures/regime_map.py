"""Figs. 8, 10, C.1 and D.1: the dimensionless regime map and its construction."""
import contextlib
import os
import sys

import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import model
import model_cal as mc
import manifold
import helpers as cp
import style

FL = model.Fluid()
B = model.Geometry()
DHB = B.Dh * 1e3

LS = (25.0, 50.0, 100.0, 200.0)
WS = (25.0, 50.0, 100.0)
DHS = (0.5185, 0.75, 1.037, 1.5, 2.0741)
LSEG = (2.5, 5.0, 10.0)
E_OK = (2.4, 9.6)
RE_FLOOR = 50.0
PI_LEFT = 0.25
N_ARMS_MIN = 3
ARMS_MM = (8.0, 12.0, 16.0)
ARM_TAPER = 6.0
ARM_BUILT_MM = manifold.ARM_H0 * 1e3
Q_CAP = 100.0


def geom(L, W, dh, n):
    s = dh / DHB
    return model.Geometry(L_plate=L * 1e-3, W_plate=W * 1e-3, w_ch=B.w_ch * s,
                          H_ch=B.H_ch * s, w_fin=B.w_fin * s, N_seg=n)


@contextlib.contextmanager
def arm_height(h_mm):
    if abs(h_mm - ARM_BUILT_MM) < 1e-9:
        yield
        return
    oK, oV = manifold.equivalent_K, manifold.v_arm
    h0 = h_mm * 1e-3

    def K(geom, Q, fluid, k_eff):
        V_ch = Q / (geom.N_seg * geom.N_ch * geom.A_ch)
        if V_ch <= 0.0:
            return 0.0
        return manifold.dP_manifold(Q, geom.slot, geom.N_seg, fluid, k_eff, h0, h0 / ARM_TAPER,
                                    L=geom.W_plate) / (0.5 * fluid.rho * V_ch ** 2)

    manifold.equivalent_K = K
    manifold.v_arm = lambda Q, w, n_seg, h=None: oV(Q, w, n_seg, h0 if h is None else h)
    try:
        yield
    finally:
        manifold.equivalent_K, manifold.v_arm = oK, oV


def case(L, W, dh, n, arm_mm=None):
    arm_mm = ARM_BUILT_MM if arm_mm is None else arm_mm
    with arm_height(arm_mm):
        r = _case(L, W, dh, n)
    if r:
        r["arm"] = arm_mm
    return r


def _case(L, W, dh, n):
    g = geom(L, W, dh, n)
    band = cp.band_for_results()
    qs = np.geomspace(0.01, 800.0, 700) * (L * W) / 2500.0
    S = [mc.cal_SMC(g, FL, q / 60000.0, 100.0, band=band) for q in qs]
    M = [mc.cal_MMC(g, FL, q / 60000.0, 100.0, endfed=True) for q in qs]
    Ws = np.array([r["W_pump"] for r in S]); Rs = np.array([r["R_th"] for r in S])
    Wm = np.array([r["W_pump"] for r in M]); Rm = np.array([r["R_th"] for r in M])
    x = cp._crossover(Ws, Rs, Wm, Rm)
    if not x:
        return None
    Wx = x[0]
    lq = np.log(qs)
    lo = max(Ws.min(), Wm.min()); hi = min(Ws.max(), Wm.max())
    wg = np.geomspace(lo * 1.01, hi / 1.01, 60)
    qsg = np.exp(np.interp(np.log(wg), np.log(Ws), lq))
    qmg = np.exp(np.interp(np.log(wg), np.log(Wm), lq))
    rsg = np.exp(np.interp(np.log(wg), np.log(Ws), np.log(Rs)))
    rmg = np.exp(np.interp(np.log(wg), np.log(Wm), np.log(Rm)))
    tau_t = FL.rho * (qsg / 60000.0 / (g.N_ch * g.A_ch)) * g.Dh / FL.mu / band[0]
    traj = (qmg / qsg, tau_t, rmg < rsg)
    q_sc = float(np.exp(np.interp(np.log(Wx), np.log(Ws), lq)))
    q_mc = float(np.exp(np.interp(np.log(Wx), np.log(Wm), lq)))
    Re = FL.rho * (q_sc / 60000.0 / (g.N_ch * g.A_ch)) * g.Dh / FL.mu
    d = (Rs - Rm) / Rm
    k = [i for i in range(len(d) - 1) if d[i] > 0.0 >= d[i + 1]]
    tau_eq = np.nan
    if k:
        i = k[0]
        q_eq = qs[i]
        tau_eq = FL.rho * (q_eq / 60000.0 / (g.N_ch * g.A_ch)) * g.Dh / FL.mu / band[0]
    o = np.argsort(Rm)
    lRm, lqm = np.log(Rm[o]), lq[o]
    keep = np.concatenate([[True], np.diff(lRm) > 0])
    lRm, lqm = lRm[keep], lqm[keep]
    lrs = np.log(rsg)
    reach = (lrs >= lRm[0]) & (lrs <= lRm[-1])
    q_match = np.exp(np.interp(lrs, lRm, lqm))
    W_match = np.exp(np.interp(np.log(q_match), lq, np.log(Wm)))
    rW = np.where(reach, W_match / wg, np.nan)
    varm = np.where(reach, manifold.v_arm(q_match / 60000.0, g.slot, g.N_seg), np.nan)
    Rb = g.R_base
    rT = (rmg - Rb) / (rsg - Rb)
    full = dict(traj=traj, rT=rT, rW=rW, reach=reach.astype(float))
    cap = np.maximum(qsg, qmg) <= Q_CAP
    traj = tuple(np.asarray(a)[cap] for a in traj)
    if np.isfinite(tau_eq) and q_eq > Q_CAP:
        tau_eq = np.nan
    return dict(L=L, W=W, dh=dh, n=n, lsd=(L / n) / dh, Wx=Wx, tau=Re / band[0],
                Pi=q_mc / q_sc, tau_eq=tau_eq, ld=L / dh, traj=traj,
                x_in=bool(max(q_sc, q_mc) <= Q_CAP),
                g=((rsg - rmg) / rsg)[cap], rT=rT[cap],
                rW=rW[cap], reach=reach.astype(float)[cap], varm=varm[cap], full=full)


def sweep(arms=None):
    out = []
    for h in (ARMS_MM if arms is None else arms):
        for L in LS:
            for W in WS:
                for dh in DHS:
                    for ls in LSEG:
                        n = int(round(L / ls))
                        if n < N_ARMS_MIN:
                            continue
                        r = case(L, W, dh, n, arm_mm=h)
                        if r:
                            out.append(r)
    return out


SC_C, MC_C = style.SMC_C, style.MMC_C
G_C = style.CONTOUR_C


def re_star0(c, band=None):
    if band is None:
        band = cp.band_for_results()
    return c[0] * band[0]


def boundary(pi, c):
    t0, P0, a = c
    return t0 * np.maximum(1.0, pi / P0) ** a


def fit_boundary(tau, Pi):
    best = None
    for P0 in np.geomspace(0.1, 10.0, 120):
        for t0 in np.linspace(0.5, 4.0, 141):
            x = np.log(np.maximum(1.0, Pi / P0)); y = np.log(tau / t0)
            m = x > 0
            if m.sum() < 5:
                continue
            a = np.sum(x[m] * y[m]) / np.sum(x[m] ** 2)
            sd = np.std(np.log(tau) - np.log(t0 * np.maximum(1.0, Pi / P0) ** a))
            if best is None or sd < best[0]:
                best = (sd, t0, P0, a)
    return best[1:]


G_XB = np.arange(-6, 17) * (1.85 / 16)
G_YB = np.linspace(-2.0, 2.45, 23)
G_SMOOTH = 0.3
G_REACH = 2.2
RT_LEVELS = (0.75, 0.5, 0.25)
X_PI = "$\\Pi$"
Y_RE = "$Re_\\mathrm{SMC} = Re_\\mathrm{MMC}\\,N_\\mathrm{arms}\\,\\Pi^{-1}$"
Y_TAU = False
Y_TAU_LAB = "$\\tau = Re_\\mathrm{SMC}\\,/\\,Re_\\mathrm{onset}$"


CB_RT = "$r_{\\Delta T}$"
SHOW_UNITY = False


def _unity(ax, xc, yR, Z):
    if not SHOW_UNITY:
        return
    ax.contour(xc, yR, Z.T, levels=[0.0], colors="0.15", linewidths=1.8, linestyles="-.", zorder=4)
    ax.plot([], [], color="0.15", lw=1.8, ls="-.", label="field's 1× contour (the colour map's tie)")
RT_DECADES = 1.0


def _g_points(sub, key="rT", xform=np.log10, full=False):
    src = [r["full"] for r in sub] if full else sub
    P = np.concatenate([r["traj"][0] for r in src])
    T = np.concatenate([r["traj"][1] for r in src])
    V = np.concatenate([np.asarray(r[key], dtype=float) for r in src])
    D = np.concatenate([np.full(len(r["traj"][0]), i) for i, r in enumerate(src)])
    with np.errstate(divide="ignore", invalid="ignore"):
        X = xform(V)
    ok = np.isfinite(X)
    return P[ok], T[ok], X[ok], D[ok]


def _cells(P, T):
    ix = np.digitize(np.log10(P), G_XB) - 1
    iy = np.digitize(np.log10(T), G_YB) - 1
    inside = (ix >= 0) & (ix < len(G_XB) - 1) & (iy >= 0) & (iy < len(G_YB) - 1)
    return ix, iy, inside


def _unit(lx, ly):
    return np.column_stack([lx / (G_XB[1] - G_XB[0]), ly / (G_YB[1] - G_YB[0])])


def _spline(sums, cnts):
    from scipy.interpolate import RBFInterpolator
    from scipy.spatial import cKDTree
    xc = 0.5 * (G_XB[1:] + G_XB[:-1]); yc = 0.5 * (G_YB[1:] + G_YB[:-1])
    ii, jj = np.nonzero(cnts > 0)
    X = _unit(xc[ii], yc[jj])
    f = RBFInterpolator(X, sums[ii, jj] / cnts[ii, jj], kernel="thin_plate_spline",
                        smoothing=G_SMOOTH)
    return f, cKDTree(X)


def _mask_tree(cnts):
    from scipy.spatial import cKDTree
    xc = 0.5 * (G_XB[1:] + G_XB[:-1]); yc = 0.5 * (G_YB[1:] + G_YB[:-1])
    ii, jj = np.nonzero(cnts > 0)
    return cKDTree(_unit(xc[ii], yc[jj]))


def _full_fit_capped_mask(sub, key="rT", xform=np.log10):
    P, T, G, _ = _g_points(sub, key, xform, full=True)
    S, C, _, _, _ = _sums(P, T, G)
    f, _ = _spline(S, C)
    P, T, G, _ = _g_points(sub, key, xform)
    _, Cc, _, _, _ = _sums(P, T, G)
    return f, _mask_tree(Cc)


def _eval(f, tree, lx, ly):
    U = _unit(lx, ly)
    z = f(U)
    far = tree.query(U)[0] > G_REACH
    z[far] = np.nan
    return z


def _sums(P, T, G):
    ix, iy, ins = _cells(P, T)
    shape = (len(G_XB) - 1, len(G_YB) - 1)
    S = np.zeros(shape); C = np.zeros(shape)
    np.add.at(S, (ix[ins], iy[ins]), G[ins]); np.add.at(C, (ix[ins], iy[ins]), 1)
    return S, C, ix, iy, ins


def g_field(sub, key="rT", xform=np.log10):
    f, tree = _full_fit_capped_mask(sub, key, xform)
    lx = np.linspace(G_XB[0], G_XB[-1], 150); ly = np.linspace(G_YB[0], G_YB[-1], 180)
    LX, LY = np.meshgrid(lx, ly, indexing="ij")
    Z = _eval(f, tree, LX.ravel(), LY.ravel()).reshape(LX.shape)
    return 10 ** lx, 10 ** ly, Z


def pi_coeffs(g, h0=None, hL=None):
    h0 = manifold.ARM_H0 if h0 is None else h0
    hL = manifold.ARM_HL if hL is None else hL
    s = g.slot
    A_ch = g.N_seg * g.N_ch * g.A_ch
    A_arm = (g.N_seg / 2.0) * s * h0
    a = FL.mu * model.fRe_fd(g.aspect) * g.L_seg / (2.0 * g.Dh ** 2 * A_ch)
    b = FL.rho * mc.CAL_K_EFF / (2.0 * A_arm ** 2)
    return a, b, A_arm


def pi_estimate(g, Q_sc, W_sc, h0=None, hL=None):
    a, b, _ = pi_coeffs(g, h0, hL)
    r = np.roots([b, a, 0.0, -W_sc])
    r = r[np.isreal(r)].real
    return r[r > 0].min() / Q_sc


def fig_operating_points(rows, ok, c, out=None):
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    P, T, Wn = [], [], []
    for r, good in zip(rows, ok):
        if not good:
            continue
        p, t, w = r["traj"]
        P.append(p); T.append(t); Wn.append(w)
    P, T, Wn = np.concatenate(P), np.concatenate(T), np.concatenate(Wn)
    onset = cp.band_for_results()[0]
    Y = T * onset
    ax.scatter(P[~Wn], Y[~Wn], s=5, color=SC_C, alpha=0.35, lw=0, zorder=2,
               label="straight channel wins")
    ax.scatter(P[Wn], Y[Wn], s=5, color=MC_C, alpha=0.35, lw=0, zorder=2,
               label="manifold wins")
    px = np.geomspace(PI_LEFT, P.max(), 100)
    ax.plot(px, re_star0(c) * np.maximum(1.0, px / c[1]) ** c[2],
            color="0.10", lw=2.6, zorder=4,
            label="$Re^{*} = %.0f\\,\\max(1,\\,\\Pi/%.2f)^{%.2f}$"
                  % (re_star0(c), c[1], c[2]))
    ax.axhline(onset, color="0.45", lw=1.0, ls=":", zorder=1,
               label="transition onset measured here ($Re$ = %.0f)" % onset)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(left=PI_LEFT)
    ax.set_ylim(bottom=RE_FLOOR)
    wr = (T < boundary(P, c)) != Wn
    ax.set_xlabel(X_PI)
    ax.set_ylabel(Y_RE)
    ax.grid(alpha=0.25, lw=0.6)
    style.lock_box(ax)
    leg = style.legend_below(fig, ax, ncol=2, markerscale=4.0)
    for h in getattr(leg, "legend_handles", None) or leg.legendHandles:
        h.set_alpha(1.0)
    out = out or style.out_path("figC1_operating_points.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    wrong = (T < boundary(P, c)) != Wn
    return out


def _cbar_axes(fig, ax, right_edge=0.845):
    fig.canvas.draw()
    pos = ax.get_position()
    if pos.x1 > right_edge:
        k = (right_edge - pos.x0) / pos.width
        cy = pos.y0 + 0.5 * pos.height
        ax.set_position([pos.x0, cy - 0.5 * pos.height * k,
                         pos.width * k, pos.height * k])
        pos = ax.get_position()
    return fig.add_axes([pos.x1 + 0.015, pos.y0, 0.020, pos.height])


def _cbar_rT(fig, cax, im, Zp):
    lo_, hi_ = np.nanmin(Zp) < -RT_DECADES, np.nanmax(Zp) > RT_DECADES
    cb = fig.colorbar(im, cax=cax, extend={(True, True): "both", (True, False): "min",
                                           (False, True): "max"}.get((lo_, hi_), "neither"))
    tk = np.log10([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
    cb.set_ticks(tk)
    cb.set_ticklabels(["%g×" % (10 ** t) for t in tk])
    cb.set_label(CB_RT)
    return cb


def _pane_rT(ax, good, c, onset, onset_label, lims=None, note=True, tau=False):
    from matplotlib.colors import TwoSlopeNorm
    xc, yc, Z = g_field(good)
    ysc = 1.0 if tau else onset
    yR = yc * ysc
    Zp = Z.T
    norm = TwoSlopeNorm(vmin=-RT_DECADES, vcenter=0.0, vmax=RT_DECADES)
    cmap = style.ratio_cmap().copy()
    cmap.set_bad("0.93")
    ax.set_facecolor("0.93")
    im = ax.pcolormesh(xc, yR, Zp, cmap=cmap, norm=norm, shading="auto", zorder=1)

    _unity(ax, xc, yR, Z)
    cs = ax.contour(xc, yR, Z.T, levels=np.log10(sorted(RT_LEVELS)), colors=G_C,
                    linewidths=1.4, linestyles="solid", zorder=4)
    floor = (lims[1][0] if lims else RE_FLOOR / onset * ysc)
    for lev, segs in zip(cs.levels, cs.allsegs):
        for s in ([max(segs, key=len)] if segs else []):
            x, y = s[np.argmax(s[:, 0])]
            at_floor = y <= 1.05 * floor
            ax.annotate("%g×" % round(10 ** lev, 2), xy=(x, y),
                        xytext=(4, 3 if at_floor else 0), textcoords="offset points",
                        ha="left", va="bottom" if at_floor else "center",
                        fontsize=style.ANNOT_FS, color=G_C, zorder=6)

    P = np.concatenate([r["traj"][0] for r in good])
    R0 = c[0] * ysc
    px = np.geomspace(PI_LEFT, P.max(), 100)
    ax.plot(px, R0 * np.maximum(1.0, px / c[1]) ** c[2], color="0.10", lw=2.6,
            zorder=5,
            label=("$\\tau^{*} = %.2f\\,\\max(1,\\,\\Pi/%.2f)^{%.2f}$" if tau
                   else "$Re^{*} = %.0f\\,\\max(1,\\,\\Pi/%.2f)^{%.2f}$") % (R0, c[1], c[2]))
    ax.axhline(ysc, color="0.30", lw=1.2, ls=":", zorder=2,
               label=("SMC transition onset ($\\tau$ = 1)" if tau else "%s ($Re$ = %.0f)" % (onset_label, onset)))

    ax.set_xscale("log"); ax.set_yscale("log")
    if lims is None:
        lims = ((PI_LEFT, float(P.max())),
                (RE_FLOOR / onset * ysc, float(np.nanmax(yR[np.isfinite(Zp).any(axis=1)]))))
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1])
    _xt = [t for t in (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0) if lims[0][0] <= t <= lims[0][1]]
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(_xt))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(X_PI)
    ax.set_ylabel(Y_TAU_LAB if tau else Y_RE)
    if tau:
        _tau_yticks(ax)
    if note:
        ax.annotate("field computed with transition onset $Re$ = %.0f" % onset,
                    xy=(0.98, 0.97), xycoords="axes fraction", ha="right", va="top",
                    fontsize=11.5, color="0.25", zorder=7)
    ax.grid(alpha=0.20, lw=0.6, zorder=3)
    style.lock_box(ax)
    return im, Zp, lims, R0


def g_spline(sub, key="rT", xform=np.log10):
    return _full_fit_capped_mask(sub, key, xform)


WINDOW_TOP = 5532.0


def fig_onset_sensitivity(A, B, lims):
    from matplotlib.colors import TwoSlopeNorm
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    lx = np.linspace(G_XB[0], G_XB[-1], 150)
    lR = np.linspace(np.log10(lims[1][0]), np.log10(lims[1][1]), 220)
    LX, LR = np.meshgrid(lx, lR, indexing="ij")
    Zs, Ds = [], []
    for rows, ok, c, onset, _ in (A, B):
        sub = [r for r, g_ in zip(rows, ok) if g_]
        f, tree = g_spline(sub)
        ly = (LR - np.log10(onset)).ravel()
        Zs.append(_eval(f, tree, LX.ravel(), ly).reshape(LX.shape))
        Ds.append(tree.query(_unit(LX.ravel(), ly))[0].reshape(LX.shape))
    D = Zs[0] - Zs[1]
    lim = 0.05 * np.ceil(np.nanpercentile(np.abs(D), 99.5) / 0.05)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    cmap = style.ratio_alt_cmap().copy()
    cmap.set_bad("0.93")
    ax.set_facecolor("0.93")
    xs, ys = 10 ** lx, 10 ** lR
    im = ax.pcolormesh(xs, ys, D.T, cmap=cmap, norm=norm, shading="auto", zorder=1)
    cs = ax.contour(xs, ys, D.T, levels=np.log10([0.8, 0.9, 1 / 0.9, 1.25]),
                    colors="0.20", linewidths=1.0, linestyles="-", zorder=3)
    ax.clabel(cs, fmt=lambda v: "%.2g×" % (10 ** v), fontsize=10, inline=True,
              inline_spacing=4)

    for (rows, ok, c, onset, name), ls in zip((A, B), ("-", (0, (5, 2.5)))):
        P = np.concatenate([r["traj"][0] for r, g_ in zip(rows, ok) if g_])
        px = np.geomspace(PI_LEFT, P.max(), 100)
        R0 = c[0] * onset
        ax.plot(px, R0 * np.maximum(1.0, px / c[1]) ** c[2], color="0.10", lw=2.2,
                ls=ls, zorder=5, label="$Re^{*}$ fitted at %s" % name)
    for k, y in enumerate((B[3], WINDOW_TOP)):
        ax.axhline(y, color="0.35", lw=1.1, ls=":", zorder=4,
                   label=("straight channel's regime differs between the two "
                          "assumptions ($Re$ %.0f-%.0f)" % (B[3], WINDOW_TOP))
                   if k == 0 else None)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1])
    ax.set_xlabel(X_PI)
    ax.set_ylabel(Y_RE)
    ax.grid(alpha=0.20, lw=0.6, zorder=2)
    style.lock_box(ax)
    style.legend_below(fig, ax, ncol=1)
    fig.canvas.draw()
    pos = ax.get_position()
    RIGHT_EDGE = 0.845
    if pos.x1 > RIGHT_EDGE:
        k = (RIGHT_EDGE - pos.x0) / pos.width
        cy = pos.y0 + 0.5 * pos.height
        ax.set_position([pos.x0, cy - 0.5 * pos.height * k,
                         pos.width * k, pos.height * k])
        pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.015, pos.y0, 0.020, pos.height])
    over = np.nanmax(np.abs(D)) > lim
    cb = fig.colorbar(im, cax=cax, extend="both" if over else "neither")
    tk = [t for t in np.log10([0.5, 0.67, 0.8, 0.9, 1.0, 1.11, 1.25, 1.5, 2.0])
          if abs(t) <= lim + 1e-9]
    cb.set_ticks(tk)
    cb.set_ticklabels(["%.2g×" % (10 ** t) for t in tk])
    cb.set_label("$r_{\\Delta T}$, onset %s field ÷ onset %s field" % (A[4], B[4]))
    out = style.out_path("figD1_onset_sensitivity.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)

    return out


def key_field(sub, key, xform):
    return g_field(sub, key, xform)


POWER_DECADES = 3.0
POWER_LEVELS = (0.1, 0.01, 0.001)


BASELINE_BUDGETS = (2.0, 4.0, 15.0)
MAP_RIGHT = 0.79


def device_curves(L, W, dh, n, arm_mm=None):
    arm_mm = ARM_BUILT_MM if arm_mm is None else arm_mm
    g = geom(L, W, dh, n)
    band = cp.band_for_results()
    q = np.geomspace(0.01, 800.0, 700) * (L * W) / 2500.0 / 60000.0
    with arm_height(arm_mm):
        S = [mc.cal_SMC(g, FL, x, 100.0, band=band) for x in q]
        M = [mc.cal_MMC(g, FL, x, 100.0, endfed=True) for x in q]
    arr = lambda rs, k: np.array([r[k] for r in rs])
    return q, arr(S, "W_pump"), arr(M, "W_pump"), arr(S, "R_th"), arr(M, "R_th"), arr(S, "Re")


def device_at(cur, W):
    q, Ws, Wm, _, _, Re = cur
    qs = np.exp(np.interp(np.log(W), np.log(Ws), np.log(q)))
    qm = np.exp(np.interp(np.log(W), np.log(Wm), np.log(q)))
    return float(qm / qs), float(np.exp(np.interp(np.log(qs), np.log(q), np.log(Re))))


def baseline_points(budgets=BASELINE_BUDGETS, arm_mm=None):
    cur = device_curves(50.0, 50.0, DHB, 10, arm_mm)
    wx = cp._crossover(cur[1], cur[3], cur[2], cur[4])[0]
    pts = [(w,) + device_at(cur, w) + (False,) for w in budgets] + [(wx,) + device_at(cur, wx) + (True,)]
    return sorted(pts), cur


def dh_path(W_fixed, dhs=None):
    dhs = np.geomspace(DHS[0], DHS[-1], 25) if dhs is None else dhs
    return [(d,) + device_at(device_curves(50.0, 50.0, d, 10), W_fixed) for d in dhs]


MAP_HEIGHT = 5.0


def fig_regime_map(rows, ok, c, onset, onset_label, lims, out=None):
    fig, ax = plt.subplots(figsize=(9.0, MAP_HEIGHT))
    ax.set_anchor("W")
    good = [r for r, g_ in zip(rows, ok) if g_]
    if Y_TAU and lims is not None:
        lims = (lims[0], (lims[1][0] / onset, lims[1][1] / onset))
    im, Zp, lims, _ = _pane_rT(ax, good, c, onset, onset_label, lims, note=False, tau=Y_TAU)
    ax.fill([], [], facecolor="0.93", edgecolor="0.55", lw=0.8,
            label="no data: flow > %.0f L/min or outside the sampled designs" % Q_CAP)
    style.legend_below(fig, ax, ncol=1, row_height=0.36, pad=0.05)
    _cbar_rT(fig, _cbar_axes(fig, ax, right_edge=MAP_RIGHT), im, Zp)
    style.trim_width(fig)
    out = out or style.out_path("fig08_regime_map.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


ZOOM_PI = (0.6, 4.5)
ZOOM_RE = (1500.0, 12000.0)
DH_MARKS = (0.5185, 0.75, 1.037, 1.5, 2.0741)
ZOOM_BUDGET = 4.0
ZOOM_ARMS = (4.0,) + ARMS_MM
ZOOM_PATH_N = 25


def r_vol(h_mm):
    return 1.0 + h_mm * 1e-3 / (B.t_base + B.H_ch)


def arm_path(W_fixed, arms=None):
    arms = np.linspace(ZOOM_ARMS[0], ZOOM_ARMS[-1], ZOOM_PATH_N) if arms is None else arms
    return [(h,) + device_at(device_curves(50.0, 50.0, DHB, 10, arm_mm=h), W_fixed) for h in arms]


def _zoom_field(ax, xc, yR, Z, norm, levels, fmt, c, onset, tau=False):
    s = 1.0 if tau else onset
    zr = (ZOOM_RE[0] / onset * s, ZOOM_RE[1] / onset * s)
    cmap = style.ratio_cmap().copy(); cmap.set_bad("0.93")
    ax.set_facecolor("0.93")
    im = ax.pcolormesh(xc, yR, Z.T, cmap=cmap, norm=norm, shading="auto", zorder=1)
    _unity(ax, xc, yR, Z)
    cs = ax.contour(xc, yR, Z.T, levels=np.log10(sorted(levels)), colors=G_C, linewidths=1.2,
                    linestyles="solid", zorder=3)
    for lev, segs in zip(cs.levels, cs.allsegs):
        pts = np.concatenate(segs) if segs else np.empty((0, 2))
        inw = pts[(pts[:, 0] > ZOOM_PI[0]) & (pts[:, 0] < ZOOM_PI[1] / 1.08) & (pts[:, 1] > zr[0]) & (pts[:, 1] < zr[1])]
        if len(inw):
            x, y = inw[np.argmax(inw[:, 0])]
            ax.annotate(fmt(lev), xy=(x, y), xytext=(-2, 4), textcoords="offset points",
                        ha="right", va="bottom", fontsize=ZOOM_CONTOUR_FS, color=G_C, zorder=6)
    px = np.geomspace(ZOOM_PI[0], ZOOM_PI[1], 200)
    ax.plot(px, boundary(px, c) * s, color="0.10", lw=2.6, zorder=5,
            label=("$\\tau^{*} = %.2f\\,\\max(1,\\,\\Pi/%.2f)^{%.2f}$" if tau
                   else "$Re^{*} = %.0f\\,\\max(1,\\,\\Pi/%.2f)^{%.2f}$") % (c[0] * s, c[1], c[2]))
    ax.axhline(s, color="0.30", lw=1.2, ls=":", zorder=2,
               label=("SMC transition onset ($\\tau$ = 1)" if tau
                      else "transition onset measured here ($Re$ = %.0f)" % onset))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*ZOOM_PI); ax.set_ylim(*zr)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.6, 1, 2, 3, 4]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%g" % v))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if tau:
        _tau_yticks(ax, (0.4, 0.5, 0.7, 1, 1.5, 2, 2.5))
    else:
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([2000, 3000, 5000, 7000, 10000]))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "{:,.0f}".format(v)))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(X_PI)
    ax.set_ylabel(Y_TAU_LAB if tau else Y_RE)
    ax.grid(alpha=0.20, lw=0.6, zorder=2)
    ax.set_anchor("W")
    style.lock_box(ax)
    return im


ZOOM_CONTOUR_FS = style.ANNOT_FS
ARM_LABEL_OFFSETS = {12.0: (0, -50, "center")}
_TAG = dict(fontsize=style.ANNOT_FS, color="0.10", zorder=9,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))


SYM_DH, SYM_WP, SYM_H0 = "$D_\\mathrm{h}$", "$W_\\mathrm{p}$", "$h_0$"


def _zoom_dh(ax, dpath, dmarks, bpts, crossover=True):
    ax.plot([p[1] for p in dpath], [p[2] for p in dpath], color="black", lw=1.5, ls="--", zorder=6,
            label="%s constant, %s swept" % (SYM_WP, SYM_DH))
    for d, pi, re in dmarks:
        if abs(d - DHB) < 1e-3:
            continue
        ax.plot(pi, re, ls="none", marker="o", ms=8, mfc="white", mec="black", mew=1.6, zorder=8)
        dx, dy, ha, va = ((8, 3, "left", "bottom") if d > 1.8 else (-6, -11, "right", "top") if d > DHB
                          else (-7, -4, "right", "top") if d < 0.6 else (7, -2, "left", "top"))
        ax.annotate("%s = %.2f mm" % (SYM_DH, d), xy=(pi, re), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va=va, **_TAG)
    pts = [p for p in bpts if crossover or not p[3]]
    ax.plot([p[1] for p in pts], [p[2] for p in pts], color="black", lw=1.2, zorder=6)
    for w, pi, re, isx in pts:
        ax.plot(pi, re, ls="none", marker="D" if isx else "o", ms=9 if isx else 8, mfc="white",
                mec="black", mew=1.6, zorder=8)
        lab = ("%s = %.3g W (crossover)" if isx else "%s = %g W") % (SYM_WP, w)
        if abs(w - ZOOM_BUDGET) < 1e-9:
            lab += ", %s = %.2f mm" % (SYM_DH, DHB)
        ax.annotate(lab, xy=(pi, re), xytext=(10, -10) if isx else (-9, 0), textcoords="offset points",
                    ha="left" if isx else "right", va="top" if isx else "center", **_TAG)


def _zoom_pane(ax, xc, yR, Z, norm, levels, fmt, c, onset, dpath, dmarks, bpts, crossover=True, tau=False):
    im = _zoom_field(ax, xc, yR, Z, norm, levels, fmt, c, onset, tau=tau)
    _zoom_dh(ax, dpath, dmarks, bpts, crossover)
    return im


def _zoom_arms(ax, apath, arm_pts):
    ax.plot([p[1] for p in apath], [p[2] for p in apath], color="black", lw=1.5, ls="-.", zorder=6,
            label="%s constant, %s swept" % (SYM_WP, SYM_H0))
    hi = max(arm_pts)
    for h, pts in sorted(arm_pts.items()):
        pts = [p for p in pts if not p[3]]
        ax.plot([p[1] for p in pts], [p[2] for p in pts], color="black", lw=1.2, zorder=6)
        for w, pi, re, _ in pts:
            ax.plot(pi, re, ls="none", marker="o", ms=8, mfc="white", mec="black", mew=1.6, zorder=8)
            if h == hi:
                ax.annotate("%s = %g W" % (SYM_WP, w), xy=(pi, re), xytext=(8, 2),
                            textcoords="offset points", ha="left", va="bottom", **_TAG)
        _, pi0, re0, _ = min(pts)
        dx, dy, ha = ARM_LABEL_OFFSETS.get(h, (4, -9, "left") if h == hi else (0, -11, "center"))
        ax.annotate("%s = %g mm\n$r_{\\forall}$ = %.1f×" % (SYM_H0, h, r_vol(h)), xy=(pi0, re0),
                    xytext=(dx, dy), textcoords="offset points", ha=ha, va="top", **_TAG)


def _stacked_cbar(fig, axA, axB, right_edge=MAP_RIGHT):
    fig.canvas.draw()
    a, b = axA.get_position(), axB.get_position()
    if a.x1 > right_edge:
        k = (right_edge - a.x0) / a.width
        for ax, p in ((axA, a), (axB, b)):
            cy = p.y0 + 0.5 * p.height
            ax.set_position([p.x0, cy - 0.5 * p.height * k, p.width * k, p.height * k])
        a, b = axA.get_position(), axB.get_position()
    cy = 0.5 * (b.y0 + a.y1)
    return fig.add_axes([a.x1 + 0.015, cy - 0.5 * a.height, 0.020, a.height])


SIDE_W, SIDE_H = 16.0, 5.6


ZOOM_H = 10.9


def fig_device_pair_sweeps(rows, ok, c, onset, lims, out=None, out_pair=None, side_by_side=False, write_pair=False):
    from matplotlib.colors import TwoSlopeNorm
    good = [r for r, g_ in zip(rows, ok) if g_]
    xc, yc, ZT = g_field(good)
    _, _, ZW = key_field(good, "rW", np.log10)
    yR = yc * onset
    dpath = dh_path(ZOOM_BUDGET)
    dmarks = dh_path(ZOOM_BUDGET, np.array(DH_MARKS))
    bpts, _ = baseline_points()
    apath = arm_path(ZOOM_BUDGET)
    arm_pts = {h: baseline_points(arm_mm=h)[0] for h in ZOOM_ARMS}
    k = 1.0 / onset if Y_TAU else 1.0
    yD = yc if Y_TAU else yR
    sc = lambda pts: [tuple(p[:2]) + (p[2] * k,) + tuple(p[3:]) for p in pts]
    dpath_d, dmarks_d, bpts_d, apath_d = sc(dpath), sc(dmarks), sc(bpts), sc(apath)
    arm_d = {h: sc(v) for h, v in arm_pts.items()}
    normT = TwoSlopeNorm(vmin=-RT_DECADES, vcenter=0.0, vmax=RT_DECADES)
    fmtT = lambda v: "%g×" % round(10 ** v, 2)

    def draw(pair, path):
        if pair:
            fig, (axA, axB) = plt.subplots(2, 1, figsize=(9.0, ZOOM_H), sharex=False)
            imA = _zoom_pane(axA, xc, yD, ZT, normT, RT_LEVELS, fmtT, c, onset, dpath_d, dmarks_d, bpts_d, tau=Y_TAU)
            imB = _zoom_pane(axB, xc, yD, ZW, TwoSlopeNorm(vmin=-POWER_DECADES, vcenter=0.0, vmax=POWER_DECADES),
                             POWER_LEVELS, lambda v: "%g×" % float("%.1g" % 10 ** v), c, onset,
                             dpath_d, dmarks_d, bpts_d, tau=Y_TAU)
        else:
            fig, (axA, axB) = (plt.subplots(1, 2, figsize=(SIDE_W, SIDE_H), sharex=False, sharey=True) if side_by_side
                               else plt.subplots(2, 1, figsize=(9.0, ZOOM_H), sharex=False))
            imA = _zoom_pane(axA, xc, yD, ZT, normT, RT_LEVELS, fmtT, c, onset, dpath_d, dmarks_d, bpts_d,
                             crossover=False, tau=Y_TAU)
            _zoom_field(axB, xc, yD, ZT, normT, RT_LEVELS, fmtT, c, onset, tau=Y_TAU)
            _zoom_arms(axB, apath_d, arm_d)
        for ax, lab in ((axA, "(a)"), (axB, "(b)")):
            ax.text(0.0, 1.02, lab, transform=ax.transAxes, ha="left", va="bottom", fontsize=20)
        if side_by_side and not pair:
            axB.set_ylabel("")
            axB.tick_params(axis="y", labelleft=False)
        axA.plot([], [], ls="-", color="black", lw=1.2, marker="o", mfc="white", mec="black", mew=1.6, ms=8,
                 label=("%s or %s constant, %s swept" % (SYM_H0, SYM_DH, SYM_WP)) if not pair
                 else "baseline plate as the pump budget grows")
        if pair:
            axA.plot([], [], ls="none", marker="D", mfc="white", mec="black", mew=1.6, ms=9, label="its crossover")
        if pair:
            style.legend_below(fig, [axA, axB], ncol=1)
            _cbar_rT(fig, _cbar_axes(fig, axA, right_edge=MAP_RIGHT), imA, ZT.T)
            _cbar_rW(fig, _cbar_axes(fig, axB, right_edge=MAP_RIGHT), imB)
        else:
            style.legend_below(fig, [axA, axB], ncol=3 if side_by_side else 1, row_height=0.36,
                              pad=0.25 if side_by_side else 0.05)
            _cbar_rT(fig, _side_cbar(fig, axA, axB) if side_by_side
                     else _stacked_cbar(fig, axA, axB, right_edge=MAP_RIGHT), imA, ZT.T)
            style.trim_width(fig)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=300)
        plt.close(fig)

    out = out or style.out_path("fig10_device_pair_sweeps.png")
    out_pair = out_pair
    draw(False, out)
    if write_pair:
        draw(True, out_pair)
    return out


def run_at_band(band):
    ob, of = model.BAND_MEASURED, cp.band_for_results
    model.BAND_MEASURED = band
    cp.band_for_results = lambda L_over_Dh=None: band
    try:
        rows = sweep()
        ok = np.array([E_OK[0] <= r["lsd"] <= E_OK[1] for r in rows])
        tau = np.array([r["tau"] for r in rows]); Pi = np.array([r["Pi"] for r in rows])
        c = fit_boundary(tau[ok], Pi[ok])
        return rows, ok, c
    finally:
        model.BAND_MEASURED = ob
        cp.band_for_results = of


if __name__ == "__main__":
    style.apply_rc()
    rows = sweep()
    ok = np.array([E_OK[0] <= r["lsd"] <= E_OK[1] for r in rows])
    tau = np.array([r["tau"] for r in rows]); Pi = np.array([r["Pi"] for r in rows])
    c = fit_boundary(tau[ok], Pi[ok])
    onset = cp.band_for_results()[0]
    print("wrote %s" % fig_operating_points(rows, ok, c))
    fig, ax = plt.subplots()
    _, _, lims, _ = _pane_rT(ax, [r for r, g in zip(rows, ok) if g], c, onset, "transition onset measured here")
    plt.close(fig)
    rows23, ok23, c23 = run_at_band(model.BAND_CLASSICAL)
    print("wrote %s" % fig_onset_sensitivity((rows, ok, c, onset, "4610"), (rows23, ok23, c23, 2300.0, "2300"), lims))
    print("wrote %s" % fig_regime_map(rows, ok, c, onset, "transition onset measured here", lims))
    print("wrote %s" % fig_device_pair_sweeps(rows, ok, c, onset, lims))
