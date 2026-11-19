"""Thesis Tables 2-6: error of the calibrated ROM closures against CFD."""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import manifold
import model
import model_cal as mc
import refit_cal as rc

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FLUID = model.Fluid()

SMC_CASES = [
    ("Baseline", "baseline", rc.CANON["smc"], 700e-6, 2.0000e-3, 400.0e-6, 45, 0.35, 48.2),
    ("Low alpha", "alpha0.20", "sweeps/alpha_smc_a020.csv", 622.2e-6, 3.1110e-3, 488.9e-6, 45, 0.20, 48.2),
    ("High alpha", "alpha0.70", "sweeps/alpha_smc_a070.csv", 881.4e-6, 1.2592e-3, 229.7e-6, 45, 0.70, 48.2),
    ("Low L/Dh", "dh_large", "sweeps/dh_smc_large.csv", 1400e-6, 4.0000e-3, 800.0e-6, 23, 0.35, 24.1),
    ("High L/Dh", "dh_small", "sweeps/dh_smc_small.csv", 350e-6, 1.0000e-3, 200.0e-6, 91, 0.35, 96.4),
]
MMC_NU_CASES = [("Baseline", "baseline"), ("Large D_h", "dh_large"), ("Small D_h", "dh_small"),
                ("High N_arms", "nseg_20"), ("Low N_arms", "nseg_5")]
MMC_DP_CASES = [("Baseline", "Baseline"), ("Large D_h", "D_h x2"), ("Small D_h", "D_h x0.5"),
                ("High N_arms", "N_arms x2"), ("Low N_arms", "N_arms x0.5")]


def write(name, header, rows):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote %s" % path)


def smc_errors(kind):
    rows, pool_c, pool_u = [], [], []
    for label, key, fn, w_ch, h_ch, w_fin, n_ch, alpha, l_dh in SMC_CASES:
        g = model.Geometry(w_ch=w_ch, H_ch=h_ch, w_fin=w_fin, N_ch_set=n_ch)
        band = mc.MEASURED_BANDS[key]
        cal, unc = [], []
        for q, r, dp in rc.load(fn):
            qs = q / 60000.0
            if kind == "dP":
                if not dp:
                    continue
                cal.append(abs(mc.cal_SMC(g, FLUID, qs, 100.0, band=band)["dP"] / dp * 100.0 - 100.0))
                unc.append(abs(model.model_SMC(g, FLUID, qs, 100.0, band=band)["dP"] / dp * 100.0 - 100.0))
            else:
                if not r:
                    continue
                f_req, b_req = rc.invert_nu("SMC", g, FLUID, q, r)
                nu_req = b_req["Nu"] * f_req
                b = model.model_SMC(g, FLUID, qs, 100.0, band=band)
                cal.append(abs(b["Nu"] * mc.smc_nu_factor(b["Re"], band=band) / nu_req * 100.0 - 100.0))
                unc.append(abs(b["Nu"] / nu_req * 100.0 - 100.0))
        pool_c += cal
        pool_u += unc
        rows.append([label, "%.2f" % alpha, "%.1f" % l_dh, "%.1f" % np.mean(cal), "%.1f" % np.max(cal),
                     "%.1f" % np.mean(unc), "%.1f" % np.max(unc)])
    rows.append(["Pooled", "", "", "%.1f" % np.mean(pool_c), "%.1f" % np.max(pool_c),
                 "%.1f" % np.mean(pool_u), "%.1f" % np.max(pool_u)])
    return rows, np.mean(pool_c)


def mmc_nu_errors():
    cases = dict(rc.MMC_NU_FIT)
    cases.update(rc.MMC_NU_HOLDOUT)
    rows, pool_c, pool_u = [], [], []
    for label, key in MMC_NU_CASES:
        fn, w_ch, h_ch, w_fin, n_ch, n_seg, slot = cases[key]
        g = model.Geometry(w_ch=w_ch, H_ch=h_ch, w_fin=w_fin, N_ch_set=n_ch, N_seg=n_seg, w_slot=slot)
        cal, unc = [], []
        for q, r, _ in rc.load(fn):
            e_cfd, b = rc.invert_nu("MMC", g, FLUID, q, r)
            if not (rc.RE_CORR_LO <= b["Re"] <= rc.RE_CORR_HI):
                continue
            e_corr = mc.mmc_nu_enhancement(b["Re"], g.L_seg / g.Dh)
            cal.append(abs(e_corr / e_cfd * 100.0 - 100.0))
            unc.append(abs(1.0 / e_cfd * 100.0 - 100.0))
        pool_c += cal
        pool_u += unc
        rows.append([label, "%.3f" % (g.Dh * 1e3), n_seg, "%.3f" % (g.L_seg / g.Dh),
                     "%.2f" % np.mean(cal), "%.2f" % np.max(cal), "%.2f" % np.mean(unc), "%.2f" % np.max(unc)])
    rows.append(["Pooled", "", "", "", "%.2f" % np.mean(pool_c), "%.2f" % np.max(pool_c),
                 "%.2f" % np.mean(pool_u), "%.2f" % np.max(pool_u)])
    return rows, np.mean(pool_c)


def mmc_dp_errors():
    family = {lab: (fn, kw) for lab, fn, kw in rc.ENDFED_FAMILY}
    rows, pool_c, pool_u = [], [], []
    for label, key in MMC_DP_CASES:
        fn, kw = family[key]
        g = model.Geometry(**kw)
        n = g.N_seg
        ks, cal, unc = [], [], []
        for q, _, dp in rc.load(fn):
            if not np.isfinite(dp):
                continue
            qv = q / 60000.0
            b = model.model_MMC(g, FLUID, qv, 100.0, band=model.BAND_MEASURED)
            dp_chan = mc.dP_mmc_coeff(g, b["Re"] * n, n, 0.0) * 0.5 * FLUID.rho * (b["V"] * n) ** 2
            dyn = 0.5 * FLUID.rho * manifold.v_arm(qv, g.slot, n) ** 2
            unc.append(abs(model.model_MMC(g, FLUID, qv, 100.0)["dP"] / dp - 1.0) * 100.0)
            cal.append(abs(mc.cal_MMC(g, FLUID, qv, 100.0, endfed=True)["dP"] / dp - 1.0) * 100.0)
            if q >= rc.Q_FIT:
                ks.append((dp - dp_chan) / dyn)
        pool_c += cal
        pool_u += unc
        rows.append([label, "%.3f" % (g.Dh * 1e3), n, "%g" % round(g.L_seg * 1e3, 2), "%g" % round(g.slot * 1e3, 2),
                     "%.3f" % np.mean(ks), "%.2f" % np.mean(cal), "%.2f" % np.max(cal), "%.2f" % np.mean(unc), "%.2f" % np.max(unc)])
    rows.append(["Pooled", "", "", "", "", "%.3f" % mc.CAL_K_EFF, "%.2f" % np.mean(pool_c), "%.2f" % np.max(pool_c),
                 "%.2f" % np.mean(pool_u), "%.2f" % np.max(pool_u)])
    return rows, np.mean(pool_c)


def main():
    os.makedirs(OUT, exist_ok=True)
    err = ["calibrated mean |e| (%)", "calibrated max |e| (%)", "uncalibrated mean |e| (%)", "uncalibrated max |e| (%)"]
    t2, e_smc_nu = smc_errors("Nu")
    t3, e_mmc_nu = mmc_nu_errors()
    t4, e_smc_dp = smc_errors("dP")
    t5, e_mmc_dp = mmc_dp_errors()
    write("table02_smc_nusselt_error.csv", ["case", "alpha", "L/Dh"] + err, t2)
    write("table03_mmc_nusselt_calibration.csv", ["case", "D_h (mm)", "N_arms", "L_seg/D_h"] + err, t3)
    write("table04_smc_pressure_drop_error.csv", ["case", "alpha", "L/Dh"] + err, t4)
    write("table05_mmc_pressure_drop_calibration.csv",
          ["case", "D_h (mm)", "N_arms", "L_pitch (mm)", "s (mm)", "K_eff"] + err, t5)
    held = len(SMC_CASES) - 1
    write("table06_closure_error.csv", ["closure", "cases fitted", "cases held out", "mean |e| (%)"], [
        ["SMC thermal", 1, held, "%.2f" % e_smc_nu],
        ["SMC hydraulic", 1, held, "%.2f" % e_smc_dp],
        ["MMC thermal", len(rc.MMC_NU_FIT), len(rc.MMC_NU_HOLDOUT), "%.2f" % e_mmc_nu],
        ["MMC hydraulic", len(rc.K_EFF_FIT), len(rc.ENDFED_FAMILY) - len(rc.K_EFF_FIT), "%.2f" % e_mmc_dp]])


if __name__ == "__main__":
    main()
