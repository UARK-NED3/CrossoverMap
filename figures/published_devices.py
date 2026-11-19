"""Fig. 11: published devices placed on the regime map."""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np

import regime_map as F

plt = F.plt
FL, mc, model, style = F.FL, F.mc, F.model, F.style
li = lambda x, X, Y: np.exp(np.interp(np.log(x), np.log(X), np.log(Y)))
A18_ESCHER = os.path.join(F.ROOT, "data", "literature", "data_escher2010_fig7a.csv")
A18_COLLINS = os.path.join(F.ROOT, "data", "literature", "data_collins2019_fig6_fig8.csv")
A18_VANERP = os.path.join(F.ROOT, "data", "literature", "data_vanerp2020_equalWp.csv")
DESIGNS = {"T&P 1981 (laminar)": (57e-6, 365e-6, 57e-6, 88, 11e-6, 2.27, 691),
           "Knight 1992 (turbulent)": (377e-6, 365e-6, 81e-6, 22, 59.2e-6, 12.2, 8459)}
MK = dict(ms=8.5, mec="k", mew=1.1, zorder=9)
COLLINS_QPR = 482.0 / 1298.0
COLLINS_ONSET = 600.0
OUT = style.out_path("fig11_published_devices.png")

ESCHER = {1: (108, 9.5, 92, 298, 100), 2: (107, 12.5, 93, 298, 100), 4: (56, 12.5, 44, 264, 50)}


def reverse(gs, gm, Q, W, band):
    qs = np.geomspace(Q.min() / 50, Q.max() * 3, 3000)
    S = [mc.cal_SMC(gs, FL, q, 100.0, band=band) for q in qs]
    Ws = np.array([r["W_pump"] for r in S]); Rs = np.array([r["R_th"] for r in S])
    qsc = li(W, Ws, qs)
    Rb = gs.R_base
    rT = np.array([(mc.cal_MMC(gm, FL, q, 100.0, endfed=True)["R_th"] - Rb) / (li(p, qs, Rs) - Rb)
                   for q, p in zip(Q, qsc)])
    return dict(Pi=Q / qsc, Re=FL.rho * (qsc / (gs.N_ch * gs.A_ch)) * gs.Dh / FL.mu, rT=rT, lsd=gm.L_seg / gm.Dh)


def escher(band):
    data = {}
    with open(A18_ESCHER, encoding="utf-8") as f:
        for r in csv.reader(l for l in f if not l.startswith("#")):
            try:
                width, K, q, dp = float(r[0]), float(r[1]), float(r[2]), float(r[3])
            except ValueError:
                continue
            data.setdefault((width, K), []).append((q, dp * 1e5))
    out = {}
    for chip, (w, K, fin, h, nom) in ESCHER.items():
        kw = dict(L_plate=20e-3, W_plate=20e-3, w_ch=w * 1e-6, H_ch=h * 1e-6, w_fin=fin * 1e-6, k_fin=148.0,
                  k_base=148.0, t_base=(525 - h) * 1e-6, N_ch_set=int(20000 // (w + fin)))
        pts = sorted(data[(float(nom), K)])
        Q = np.array([p[0] for p in pts]) / 60000.0
        dP = np.array([p[1] for p in pts])
        out[chip] = reverse(model.Geometry(**kw), model.Geometry(N_seg=int(round(2 * K)), **kw), Q, Q * dP, band)
    return out


def collins():
    D = {}
    with open(A18_COLLINS, encoding="utf-8") as f:
        for r in csv.reader(l for l in f if not l.startswith("#")):
            try:
                D.setdefault((r[0], r[1]), []).append((float(r[2]), float(r[3])))
            except (ValueError, IndexError):
                continue
    arr = lambda k: (np.array([p[0] for p in sorted(D[k])]), np.array([p[1] for p in sorted(D[k])]))
    ReS_p, dPS = arr(("Fig6_dP", "SMC")); ReM_p, dPM = arr(("Fig6_dP", "MMC"))
    ReS_t, RS = arr(("Fig8_Rtot", "SMC")); ReM_t, RM = arr(("Fig8_Rtot", "MMC"))
    W = lambda Re, Rep, dP: (Re * COLLINS_QPR / 6e7) * li(Re, Rep, dP) * 1e3
    Rs_at = lambda Re: li(Re, ReS_t, RS) if ReS_t[0] <= Re <= ReS_t[-1] else \
        RS[0] * (Re / ReS_t[0]) ** (np.log(RS[1] / RS[0]) / np.log(ReS_t[1] / ReS_t[0]))
    fine = np.linspace(ReS_p[0], ReS_p[-1], 4000)
    Wf = np.array([W(r, ReS_p, dPS) for r in fine])
    Pi, Re, ratio = [], [], []
    for re_m, r_m in zip(ReM_t, RM):
        wm = W(re_m, ReM_p, dPM) if re_m <= ReM_p[-1] else W(ReM_p[-1], ReM_p, dPM) * (re_m / ReM_p[-1]) ** 2.7
        re_s = np.interp(wm, Wf, fine) if wm >= Wf[0] else ReS_p[0] * np.sqrt(wm / Wf[0])
        Pi.append(re_m / re_s); Re.append(re_s); ratio.append(r_m / Rs_at(re_s))
    return dict(Pi=np.array(Pi), Re=np.array(Re), R_ratio=np.array(ratio))


def vanerp():
    D = {}
    with open(A18_VANERP, encoding="utf-8") as f:
        for r in csv.reader(l for l in f if not l.startswith("#")):
            try:
                row = (float(r[2]), float(r[3]), float(r[6]))
            except (ValueError, IndexError):
                continue
            D.setdefault(r[0], []).append(row)
    return {k: dict(Pi=np.array([p[0] for p in v]), Re=np.array([p[1] for p in v]), ratio=np.array([p[2] for p in v]))
            for k, v in D.items()}


RV_SWEEP = (2.0, 3.0, 5.0)


def designs():
    out = {}
    for name, (w, H, fin, N, Q, W, Re) in DESIGNS.items():
        kw = dict(L_plate=10e-3, W_plate=10e-3, w_ch=w, H_ch=H, w_fin=fin, k_fin=148.0, k_base=148.0, t_base=0.2e-3,
                  N_ch_set=N)
        g = model.Geometry(**kw)
        gm = model.Geometry(N_seg=max(2, int(round(10e-3 / (4.8 * g.Dh)))), **kw)
        h0s = [(rv - 1.0) * H for rv in RV_SWEEP]
        out[name] = dict(Pi=np.array([F.pi_estimate(gm, Q, W, h0, h0 / F.ARM_TAPER) for h0 in h0s]),
                         Re=np.full(len(RV_SWEEP), float(Re)), rv=np.array(RV_SWEEP))
    return out


A18_MODELCMP = os.path.join(F.ROOT, "data", "literature", "data_model_comparisons_equalWp.csv")


def modelcmp():
    if not os.path.exists(A18_MODELCMP):
        return {}
    D = {}
    with open(A18_MODELCMP, encoding="utf-8") as f:
        for r in csv.DictReader(l for l in f if not l.startswith("#")):
            D.setdefault(r["label"], []).append((float(r["Pi"]), float(r["Re_SMC"])))
    return {k: dict(Pi=np.array([p[0] for p in v]), Re=np.array([p[1] for p in v])) for k, v in D.items()}


def main(rows=None, ok=None, c=None, out=None):
    style.apply_rc()
    onset = F.cp.band_for_results()[0]
    band = F.cp.band_for_results()
    if rows is None:
        rows = F.sweep()
        ok = np.array([F.E_OK[0] <= r["lsd"] <= F.E_OK[1] for r in rows])
        c = F.fit_boundary(np.array([r["tau"] for r in rows])[ok], np.array([r["Pi"] for r in rows])[ok])
    good = [r for r, g in zip(rows, ok) if g]
    E, Cl, V, Dz = escher(band), collins(), vanerp(), designs()
    fT, tT = F._full_fit_capped_mask(good, "rT", np.log10)
    chart = lambda P, R: 10 ** F._eval(fT, tT, np.log10(np.asarray(P, float)), np.log10(np.asarray(R, float) / onset))

    fig, ax = plt.subplots(figsize=(9.0, F.MAP_HEIGHT))
    ax.set_anchor("W")
    im, Zp, lims, _ = F._pane_rT(ax, good, c, onset, "transition onset measured here", None, note=False, tau=F.Y_TAU)
    from matplotlib.contour import ContourSet
    XL, YL = (0.25, 12.0), (50.0, 30000.0)
    ax.set_xlim(*XL); ax.set_ylim(*YL)
    for t in list(ax.texts):
        t.remove()
    cs = next(cc for cc in ax.collections if isinstance(cc, ContourSet))
    for lev, segs in zip(cs.levels, cs.allsegs):
        if not segs:
            continue
        pts = np.concatenate(segs)
        ins = pts[(pts[:, 0] <= XL[1] * 0.93) & (pts[:, 1] >= 60.0) & (pts[:, 1] <= YL[1])]
        if len(ins):
            x_, y_ = ins[np.argmax(ins[:, 0])]
            ax.annotate("%g×" % round(10 ** lev, 2), xy=(x_, y_), xytext=(-3, 3), textcoords="offset points",
                        ha="right", va="bottom", fontsize=style.ANNOT_FS, color=F.G_C, zorder=6)
    floor = YL[0]
    y = (lambda d, on=onset: d["Re"] / on) if F.Y_TAU else (lambda d, on=None: d["Re"])
    yC = lambda d: y(d, COLLINS_ONSET) if F.Y_TAU else d["Re"]
    SIZE = {"*": 15.0, "p": 11.0, "^": 9.5, "D": 7.0, "s": 7.5}
    def series(d, mk, col, label, yy=y, line=True):
        m = yy(d) >= floor
        if line and m.sum() > 1:
            ax.plot(d["Pi"][m], yy(d)[m], "-", color="k", lw=0.8, zorder=8)
        ax.plot(d["Pi"][m], yy(d)[m], mk, mfc=col, label=label, **dict(MK, ms=SIZE.get(mk, MK["ms"])))
    for ln in ax.get_lines():
        ln.set_label("_nolegend_")
    MC = modelcmp()
    PAL = ("#f5c518", style.ACCENT_DEEP_C, "#8e44ad", "white")
    TITLES = ("Tier A", "Tier B", "Tier C", "Method")
    MK_A = [("van Erp et al. (2020)", "o", PAL[0]), ("Collins et al. (2019)", "o", PAL[1]),
            ("Ryu et al. (2003)", "s", PAL[0]), ("Boteler et al. (2012)", "s", PAL[1]),
            ("Kisitu & Ortega (2023)", "s", PAL[2]), ("Bazmi et al. (2026)", "s", PAL[3]),
            ("Cetegen (2010)", "^", PAL[0]), ("Kuiphuis et al. (2025)", "^", PAL[1])]
    SHOWN = {"Ryu et al. (2003)": "Ryu et al. (2002, 2003)"}
    REF = {"van Erp et al. (2020)": "[4]", "Collins et al. (2019)": "[20]", "Ryu et al. (2002, 2003)": "[16], [33]",
           "Boteler et al. (2012)": "[37]", "Kisitu & Ortega (2023)": "[47]", "Bazmi et al. (2026)": "[5]",
           "Cetegen (2010)": "[46]", "Kuiphuis et al. (2025)": "[11]", "Escher et al. (2010)": "[26]",
           "Tuckerman & Pease (1981)": "[14]", "Knight et al. (1992)": "[17]"}
    cite = lambda lab: "%s %s" % (lab, REF[lab])
    nrow = 9
    npad = [0]
    def title(t):
        ax.plot([], [], ls="none", label=t)
    def pad(n):
        for _ in range(n):
            npad[0] += 1
            ax.plot([], [], ls="none", label=" " * npad[0])
    title("Tier A")
    n_a = 0
    for lab, mk, col in MK_A:
        if lab == "van Erp et al. (2020)":
            for d in V.values():
                series(d, mk, col, cite(lab))
        elif lab == "Collins et al. (2019)":
            series(Cl, mk, col, cite(lab), yy=yC)
        elif lab in MC:
            series(MC[lab], mk, col, cite(SHOWN.get(lab, lab)), line=False)
        else:
            continue
        n_a += 1
    pad(nrow - 1 - n_a)
    title("Tier B")
    for chip in ESCHER:
        series(E[chip], "o", PAL[2], cite("Escher et al. (2010)"))
    title("Tier C")
    series(Dz["T&P 1981 (laminar)"], "^", PAL[2], cite("Tuckerman & Pease (1981)"))
    series(Dz["Knight 1992 (turbulent)"], "^", PAL[3], cite("Knight et al. (1992)"))
    title("Method")
    for mk, lab in (("o", "Experiment"), ("s", "CFD"), ("^", "Analytical")):
        ax.plot([], [], mk, ls="none", mfc="0.55", label=lab, **dict(MK, ms=SIZE.get(mk, MK["ms"])))
    for name, sides in (("T&P 1981 (laminar)", (-1, -1, 1)), ("Knight 1992 (turbulent)", (1, 1, 1))):
        d = Dz[name]
        for P_, R_, rv, side in zip(d["Pi"], d["Re"], d["rv"], sides):
            ax.annotate("%g×" % rv, (P_, R_), xytext=(0, 9 * side), textcoords="offset points", ha="center",
                        va="bottom" if side > 0 else "top", fontsize=style.ANNOT_FS, color="0.10", zorder=10)
    leg = style.legend_below(fig, ax, ncol=2, row_height=0.36, pad=0.05)
    for t in leg.get_texts():
        if t.get_text() in TITLES:
            t.set_fontweight("bold")
    F._cbar_rT(fig, F._cbar_axes(fig, ax, right_edge=F.MAP_RIGHT), im, Zp)
    style.trim_width(fig)
    out = out or OUT
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)

    print("wrote %s" % out)
    return out


if __name__ == "__main__":
    main()
