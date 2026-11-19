"""Fit the ROM calibration constants from the CFD data; --check compares them with model_cal.py."""
import os
import sys

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data", "cfd")
sys.path.insert(0, HERE)

import manifold
import model
from model import UA_prime

Q_FIT = 10.0
RE_CORR_LO, RE_CORR_HI = 150.0, 3700.0
CANON = {"smc": "smc_transition_sst_medium.csv", "mmc_real": "mmc_endfed_sst_medium.csv"}
ENDFED_FAMILY = [
    ("Baseline", "mmc_endfed_sst_medium.csv", dict()),
    ("N_arms x0.5", "sweeps/nseg_mmc_endfed_n5.csv", dict(N_seg=5)),
    ("N_arms x2", "sweeps/nseg_mmc_endfed_n20.csv", dict(N_seg=20)),
    ("D_h x0.5", "sweeps/dh_mmc_endfed_small.csv", dict(w_ch=0.35e-3, H_ch=1.0e-3, w_fin=0.20e-3, N_ch_set=91)),
    ("D_h x2", "sweeps/dh_mmc_endfed_large.csv", dict(w_ch=1.40e-3, H_ch=4.0e-3, w_fin=0.80e-3, N_ch_set=23)),
]
K_EFF_FIT = ("Baseline",)
MMC_NU_FIT = {
    "baseline": ("mmc_endfed_sst_medium.csv", 700e-6, 2.0e-3, 400e-6, 45, 10, None),
    "dh_small": ("sweeps/dh_mmc_endfed_small.csv", 350e-6, 1.0e-3, 200e-6, 91, 10, None),
    "dh_large": ("sweeps/dh_mmc_endfed_large.csv", 1400e-6, 4.0e-3, 800e-6, 23, 10, None),
}
MMC_NU_HOLDOUT = {
    "nseg_5": ("sweeps/nseg_mmc_endfed_n5.csv", 700e-6, 2.0e-3, 400e-6, 45, 5, None),
    "nseg_20": ("sweeps/nseg_mmc_endfed_n20.csv", 700e-6, 2.0e-3, 400e-6, 45, 20, None),
}


def load(fname):
    rows = []
    for line in open(os.path.join(DATA, fname)):
        line = line.strip()
        if not line or line.startswith("#") or line[0].isalpha():
            continue
        p = line.split(",")
        rows.append((float(p[0]), float(p[1]), float(p[2])))
    return sorted(rows)


def invert_nu(kind, geom, fluid, Q_lpm, R_target):
    fn = model.model_SMC if kind == "SMC" else model.model_MMC
    b = fn(geom, fluid, Q_lpm / 60000.0, 100.0, band=model.BAND_MEASURED)
    L_wet = geom.L_plate * geom.N_ch

    def resid(f):
        h = (b["Nu"] * f) * fluid.k / geom.Dh
        return b["R_caloric"] + 1.0 / (UA_prime(h, geom) * L_wet) + geom.R_base - R_target

    return brentq(resid, 0.02, 20.0), b


def fit():
    import model_cal as mc
    geom, fluid = model.Geometry(), model.Fluid()
    f_turb, k_smc = [], []
    for Q, R, dP in load(CANON["smc"]):
        f, b = invert_nu("SMC", geom, fluid, Q, R)
        if b["Re"] > model.RE_TURB_START:
            f_turb.append(f)
        if Q >= 20.0:
            k_smc.append((dP - b["dP_visc"]) / (0.5 * fluid.rho * b["V"] ** 2))
    pts = []
    for fn, w, H, wf, nch, nseg, slot in MMC_NU_FIT.values():
        g = model.Geometry(w_ch=w, H_ch=H, w_fin=wf, N_ch_set=nch, N_seg=nseg, w_slot=slot)
        for Q, R, _ in load(fn):
            if Q < Q_FIT:
                continue
            E, b = invert_nu("MMC", g, fluid, Q, R)
            if RE_CORR_LO <= b["Re"] <= RE_CORR_HI:
                pts.append((b["Re"], g.L_seg / g.Dh, E))
    M = np.column_stack([np.ones(len(pts)), np.log([p[0] for p in pts]), np.log([p[1] for p in pts])])
    coef = np.linalg.lstsq(M, np.log([p[2] for p in pts]), rcond=None)[0]
    k_eff = []
    for label, fn, kw in ENDFED_FAMILY:
        if label not in K_EFF_FIT:
            continue
        g = model.Geometry(**kw)
        n = g.N_seg
        for Q_lpm, _, dP in load(fn):
            if Q_lpm < Q_FIT:
                continue
            Q = Q_lpm / 60000.0
            b = model.model_MMC(g, fluid, Q, 100.0, band=model.BAND_MEASURED)
            dp_chan = mc.dP_mmc_coeff(g, b["Re"] * n, n, 0.0) * 0.5 * fluid.rho * (b["V"] * n) ** 2
            k_eff.append((dP - dp_chan) / (0.5 * fluid.rho * manifold.v_arm(Q, g.slot, n) ** 2))
    k = float(np.mean(k_smc))
    return {"CAL_F_SMC_TURB": float(np.mean(f_turb)), "CAL_K_SMC_INLET": k / 2.0, "CAL_K_SMC_OUTLET": k / 2.0,
            "CAL_MMC_NU_CORR_C": float(np.exp(coef[0])), "CAL_MMC_NU_CORR_RE": float(coef[1]),
            "CAL_MMC_NU_CORR_LSEG": float(coef[2]), "CAL_K_EFF": float(np.mean(k_eff))}


def main(argv):
    import model_cal as mc
    bad = 0
    for name, value in fit().items():
        stored = getattr(mc, name)
        same = round(value, len(repr(stored).split(".")[1])) == stored
        bad += not same
        print("%-22s = %10.6f   model_cal.py: %s%s" % (name, value, repr(stored), "" if same else "   MISMATCH"))
    if "--check" in argv:
        print("model_cal.py matches the CFD data" if not bad else "%d constants differ from the CFD data" % bad)
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
