"""CFD-calibrated ROM closures and the measured transition bands."""
import numpy as np
import model
import manifold
from model import Nu_eff, UA_prime, RE_LAM_END

CAL_K_SMC_INLET  = 0.167
CAL_K_SMC_OUTLET = 0.167

CAL_F_SMC_LAMINAR = 1.0


CAL_K_EFF = 3.677

NU_FADE_RE_LO = model.RE_LAM_END
NU_FADE_RE_HI = model.RE_TURB_START

MEASURED_BANDS = {
    "baseline":  (4610.0,  5532.0),
    "alpha0.20": (3074.0,  4610.0),
    "alpha0.70": (6148.0,  7683.0),
    "alpha0.70_highre": (5361.0, 13401.0),
    "dh_large":  (4510.0,  6013.0),
    "dh_small":  (3039.0,  6079.0),
}

CAL_F_SMC_TURB = 0.8252


CAL_MMC_NU_CORR_C    = 1.299765
CAL_MMC_NU_CORR_RE   = 0.138469
CAL_MMC_NU_CORR_LSEG = -0.393879
CAL_MMC_NU_CORR_RE_LO = 150.0
CAL_MMC_NU_CORR_RE_HI = 3700.0


def smc_nu_factor(Re: float, band=None) -> float:
    lo, hi = band if band is not None else (model.RE_LAM_END, model.RE_TURB_START)
    if Re <= lo:
        return 1.0
    if Re >= hi:
        return CAL_F_SMC_TURB
    return 1.0 + (Re - lo) / (hi - lo) * (CAL_F_SMC_TURB - 1.0)


def mmc_nu_enhancement(Re: float, Lseg_over_Dh: float) -> float:
    Rc = min(max(Re, CAL_MMC_NU_CORR_RE_LO), CAL_MMC_NU_CORR_RE_HI)
    E = (CAL_MMC_NU_CORR_C
         * Rc ** CAL_MMC_NU_CORR_RE
         * Lseg_over_Dh ** CAL_MMC_NU_CORR_LSEG)
    if Re <= NU_FADE_RE_LO:
        return E
    if Re >= NU_FADE_RE_HI:
        return 1.0
    frac = (Re - NU_FADE_RE_LO) / (NU_FADE_RE_HI - NU_FADE_RE_LO)
    return E + frac * (1.0 - E)


def dP_mmc_coeff(g, ReS, n, K):
    LD = g.L_plate / g.Dh
    Rec = ReS / n
    return (model.f_darcy(Rec, (LD / n) / max(Rec, 1e-9), g.aspect,
                          band=model.BAND_MEASURED) * (LD / n) + K) / n ** 2


def smc_friction_factor(Re: float, band=None) -> float:
    lo, hi = band if band is not None else (NU_FADE_RE_LO, NU_FADE_RE_HI)
    if Re <= lo:
        return CAL_F_SMC_LAMINAR
    if Re >= hi:
        return 1.0
    frac = (Re - lo) / (hi - lo)
    return CAL_F_SMC_LAMINAR + frac * (1.0 - CAL_F_SMC_LAMINAR)


def cal_SMC(geom, fluid, Q: float, q_heat: float, band=None) -> dict:
    if band is None:
        band = model.BAND_MEASURED
    b = model.model_SMC(geom, fluid, Q, q_heat, band=band)
    Re, V = b["Re"], b["V"]

    Nu = b["Nu"] * smc_nu_factor(Re, band=band)
    h  = Nu * fluid.k / geom.Dh
    R_conv = 1.0 / (UA_prime(h, geom) * geom.L_plate * geom.N_ch)
    R_th   = b["R_caloric"] + R_conv + geom.R_base

    dyn = 0.5 * fluid.rho * V ** 2
    dP  = (b["dP_visc"] * smc_friction_factor(Re, band=band)
           + (CAL_K_SMC_INLET + CAL_K_SMC_OUTLET) * dyn)
    return dict(arch="SMC", Q=Q, Re=Re, Nu=Nu, h=h, R_th=R_th, dP=dP, W_pump=Q * dP)


def cal_MMC(geom, fluid, Q: float, q_heat: float, endfed: bool = False) -> dict:
    b = model.model_MMC(geom, fluid, Q, q_heat, band=model.BAND_MEASURED)
    Re, V = b["Re"], b["V"]

    Nu = b["Nu"] * mmc_nu_enhancement(Re, geom.L_seg / geom.Dh)
    if Re > RE_LAM_END:
        Nu_onset = (Nu_eff(RE_LAM_END, fluid.Pr, geom.L_seg / geom.Dh, geom.aspect,
                           band=model.BAND_MEASURED)
                    * mmc_nu_enhancement(RE_LAM_END, geom.L_seg / geom.Dh))
        Nu = max(Nu, Nu_onset)
    h = Nu * fluid.k / geom.Dh
    R_conv = 1.0 / (UA_prime(h, geom) * geom.N_ch * geom.L_plate)
    R_th   = b["R_caloric"] + R_conv + geom.R_base

    if not endfed:
        raise ValueError("only the end-fed MMC is calibrated; pass endfed=True")
    K = manifold.equivalent_K(geom, Q, fluid, CAL_K_EFF)
    n   = geom.N_seg
    V_S = V * n
    ReS = Re * n
    dP  = dP_mmc_coeff(geom, ReS, n, K) * 0.5 * fluid.rho * V_S ** 2
    return dict(arch="MMC", Q=Q, Re=Re, Nu=Nu, h=h, R_th=R_th, dP=dP, W_pump=Q * dP)

