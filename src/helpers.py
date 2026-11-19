"""Shared helpers: CFD data loading, transition band and crossover finder."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import model
import model_cal as mc

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})


Q_HEAT = 100.0
GRID   = np.linspace(1.0, 60.0, 160)


HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data", "cfd")


def band_for_results(L_over_Dh=None):
    return model.BAND_MEASURED


def load_cfd(fname):
    Q, R, dP = [], [], []
    with open(os.path.join(DATA, fname)) as f:
        for row in f:
            row = row.strip()
            if not row or row.startswith("#") or row[0].isalpha():
                continue
            p = row.split(",")
            Q.append(float(p[0]))
            R.append(float(p[1]) if p[1] else np.nan)
            dP.append(float(p[2]) if len(p) > 2 and p[2] else np.nan)
    return np.array(Q), np.array(R), np.array(dP)


def eval_model(kind, Q_lpm, calibrated, endfed=False):
    R, dP = [], []
    g, fl = model.Geometry(), model.Fluid()
    for q in Q_lpm:
        Qs = q / 60000.0
        if kind == "SC":
            m = mc.cal_SMC(g, fl, Qs, Q_HEAT) if calibrated else model.model_SMC(g, fl, Qs, Q_HEAT)
        else:
            m = mc.cal_MMC(g, fl, Qs, Q_HEAT, endfed=endfed) if calibrated else model.model_MMC(g, fl, Qs, Q_HEAT)
        R.append(m["R_th"])
        dP.append(m["dP"])
    R, dP = np.array(R), np.array(dP)
    Wp = (Q_lpm / 60000.0) * dP
    return R, dP, Wp


def _crossover(x1, y1, x2, y2):
    from scipy.optimize import brentq

    def f(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        o = np.argsort(x[m])
        lx, ly = np.log(x[m][o]), np.log(y[m][o])
        return (lambda w: np.exp(np.interp(np.log(w), lx, ly))), (x[m][o][0], x[m][o][-1])

    fa, ra = f(x1, y1)
    fb, rb = f(x2, y2)
    lo, hi = max(ra[0], rb[0]), min(ra[1], rb[1])
    if lo >= hi:
        return None
    g = lambda w: fa(w) - fb(w)
    if g(lo) * g(hi) > 0:
        return None
    w = brentq(g, lo, hi)
    return w, float(fa(w))
