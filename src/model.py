"""Analytical reduced-order model (ROM) of the SMC and MMC cold plates."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Fluid:
    rho: float = 997.0
    mu:  float = 8.9e-4
    cp:  float = 4180.0
    k:   float = 0.606
    T_in: float = 298.15

    @property
    def Pr(self) -> float:
        return self.cp * self.mu / self.k


@dataclass
class Geometry:
    L_plate: float = 0.050
    W_plate: float = 0.050
    w_ch:  float = 700e-6
    H_ch:  float = 2.0e-3
    w_fin: float = 400e-6
    k_fin: float = 398.0
    t_base: float = 1.0e-3
    k_base: float = 398.0
    N_seg: int = 10
    w_slot: float | None = None
    N_ch_set: int | None = None

    K_smc_inlet:  float = 0.5
    K_smc_outlet: float = 1.0
    K_mmc_turn:   float = 2.0
    K_mmc_supply: float = 0.5
    K_mmc_return: float = 1.0

    @property
    def pitch(self) -> float:
        return self.w_ch + self.w_fin

    @property
    def R_base(self) -> float:
        return self.t_base / (self.k_base * self.L_plate * self.W_plate)

    @property
    def N_ch(self) -> int:
        if self.N_ch_set is not None:
            return self.N_ch_set
        return int(self.W_plate // self.pitch)

    @property
    def A_ch(self) -> float:
        return self.w_ch * self.H_ch

    @property
    def P_ch(self) -> float:
        return 2.0 * (self.w_ch + self.H_ch)

    @property
    def Dh(self) -> float:
        return 4.0 * self.A_ch / self.P_ch

    @property
    def aspect(self) -> float:
        return min(self.w_ch, self.H_ch) / max(self.w_ch, self.H_ch)

    @property
    def L_seg(self) -> float:
        return self.L_plate / self.N_seg

    @property
    def slot(self) -> float:
        return self.w_slot if self.w_slot is not None else 0.6 * self.L_seg


_ALPHA = np.array([0.0,    0.05,   0.10,   0.125,  0.25,   0.40,   0.50,   0.70,   0.80,   0.90,   1.00])
_FRE   = np.array([96.000, 89.908, 84.676, 82.339, 72.931, 65.472, 62.192, 58.422, 57.511, 57.044, 56.908])


def fRe_fd(alpha: float) -> float:
    return float(np.interp(alpha, _ALPHA, _FRE))


_PH_AR = np.array([1.0, 2.0, 5.0, 10.0])
_PH_C = np.array([
    [141.97, -7.0603, 2603.0,  1431.7, 14364.0, -220.77],
    [142.05, -5.4166, 1481.0,  1067.8, 13177.0, -108.52],
    [142.10, -7.3374, 376.69,  800.92, 14010.0, -33.894],
    [286.65, 25.701,  337.81,  1091.5, 26415.0,  8.4098]])
PH_LP_LO = 1.0e-3
PH_LP_BLEND = 0.4
PH_LP_HI = 1.0


def _phillips_fanning(L_plus: float, row) -> float:
    C5, C6, C7, C8, C9, C10 = row
    s = np.sqrt(L_plus)
    return (C5 + C7 * s + C9 * L_plus) / (1.0 + C6 * s + C8 * L_plus + C10 * L_plus * s)


def _phillips_darcy(L_plus: float, AR: float) -> float:
    AR = max(AR, 1.0)
    if AR <= 5.0:
        j = 0 if AR <= 2.0 else 1
        a0, a1 = _PH_AR[j], _PH_AR[j + 1]
        w = (AR - a0) / (a1 - a0)
        f0 = _phillips_fanning(L_plus, _PH_C[j]); f1 = _phillips_fanning(L_plus, _PH_C[j + 1])
    else:
        w = 1.0 - 5.0 / AR if np.isfinite(AR) else 1.0
        f0 = _phillips_fanning(L_plus, _PH_C[2]); f1 = _phillips_fanning(L_plus, _PH_C[3])
    return 4.0 * ((1.0 - w) * f0 + w * f1)


def f_app_Re(L_plus: float, alpha: float) -> float:
    if alpha > 1.0:
        alpha = 1.0 / alpha
    fd = fRe_fd(alpha)
    if L_plus >= PH_LP_HI:
        return fd
    AR = 1.0 / alpha if alpha > 0.0 else np.inf
    if L_plus < PH_LP_LO:
        return max(_phillips_darcy(PH_LP_LO, AR), fd) * np.sqrt(PH_LP_LO / max(L_plus, 1e-12))
    excess = max(_phillips_darcy(L_plus, AR) - fd, 0.0)
    if L_plus > PH_LP_BLEND:
        excess *= np.log(PH_LP_HI / L_plus) / np.log(PH_LP_HI / PH_LP_BLEND)
    return fd + excess


RE_LAM_END    = 4610.0
RE_TURB_START = 5532.0

RE_LAM_END_CLASSICAL    = 2300.0
RE_TURB_START_CLASSICAL = 4000.0

BAND_MEASURED  = (RE_LAM_END, RE_TURB_START)
BAND_CLASSICAL = (RE_LAM_END_CLASSICAL, RE_TURB_START_CLASSICAL)


def transition_band(L_over_Dh: float = None, alpha: float = None, band=None):
    return band if band is not None else BAND_CLASSICAL


def f_turb(Re: float) -> float:
    return (0.790 * np.log(max(Re, 1e3)) - 1.64) ** -2


def f_darcy(Re: float, L_plus: float, alpha: float, force_laminar: bool = False,
            band=None) -> float:
    L_plus = max(L_plus, 1e-9)
    f_lam = f_app_Re(L_plus, alpha) / Re
    if force_laminar:
        return f_lam
    lam, turb = transition_band(L_plus * Re, alpha, band)
    if Re <= lam:
        return f_lam
    f_t = f_turb(Re)
    if Re >= turb:
        return f_t
    w = (Re - lam) / (turb - lam)
    return (1.0 - w) * f_lam + w * f_t


_NU_H1 = np.array([8.235, 7.451, 6.785, 6.490, 5.331, 4.472, 4.123, 3.750, 3.664, 3.620, 3.608])


def Nu_fd(alpha: float) -> float:
    return float(np.interp(alpha, _ALPHA, _NU_H1))


def Nu_dev(Re: float, Pr: float, L_over_Dh: float, alpha: float) -> float:
    Gz = (Re * Pr) / max(L_over_Dh, 1e-9)
    return Nu_fd(alpha) + (0.0668 * Gz) / (1.0 + 0.04 * Gz ** (2.0 / 3.0))


def Nu_turb(Re: float, Pr: float) -> float:
    f = f_turb(Re)
    return (f / 8.0) * (Re - 1000.0) * Pr / (1.0 + 12.7 * np.sqrt(f / 8.0) * (Pr ** (2.0 / 3.0) - 1.0))


def Nu_eff(Re: float, Pr: float, L_over_Dh: float, alpha: float,
           force_laminar: bool = False, band=None) -> float:
    Nu_lam = Nu_dev(Re, Pr, L_over_Dh, alpha)
    if force_laminar:
        return Nu_lam
    lam, turb = transition_band(L_over_Dh, alpha, band)
    if Re <= lam:
        return Nu_lam
    Nu_t = Nu_turb(Re, Pr)
    if Re >= turb:
        return Nu_t
    w = (Re - lam) / (turb - lam)
    return (1.0 - w) * Nu_lam + w * Nu_t


def fin_efficiency(h: float, geom: Geometry) -> float:
    m = np.sqrt(2.0 * h / (geom.k_fin * geom.w_fin))
    mH = m * geom.H_ch
    return 1.0 if mH < 1e-9 else np.tanh(mH) / mH


def UA_prime(h: float, geom: Geometry) -> float:
    eta = fin_efficiency(h, geom)
    return h * (geom.w_ch + eta * 2.0 * geom.H_ch)


def model_SMC(geom: Geometry, fluid: Fluid, Q: float, q_heat: float = None,
              force_laminar: bool = False, band=None) -> dict:
    N_ch = geom.N_ch
    V  = Q / (N_ch * geom.A_ch)
    Re = fluid.rho * V * geom.Dh / fluid.mu

    Nu = Nu_eff(Re, fluid.Pr, geom.L_plate / geom.Dh, geom.aspect, force_laminar, band)
    h  = Nu * fluid.k / geom.Dh
    R_conv = 1.0 / (UA_prime(h, geom) * geom.L_plate * N_ch)
    R_cal  = 1.0 / (fluid.rho * Q * fluid.cp)
    R_th   = R_cal + R_conv + geom.R_base

    L_plus = (geom.L_plate / geom.Dh) / max(Re, 1e-9)
    dyn = 0.5 * fluid.rho * V ** 2
    dP_visc  = f_darcy(Re, L_plus, geom.aspect, force_laminar, band) * (geom.L_plate / geom.Dh) * dyn
    dP_minor = (geom.K_smc_inlet + geom.K_smc_outlet) * dyn
    dP = dP_visc + dP_minor

    return dict(arch="SMC", Q=Q, V=V, Re=Re, Nu=Nu, h=h,
                R_caloric=R_cal, R_conv=R_conv, R_base=geom.R_base, R_th=R_th,
                dP_visc=dP_visc, dP_minor=dP_minor, dP=dP, W_pump=Q * dP)


def model_MMC(geom: Geometry, fluid: Fluid, Q: float, q_heat: float = None,
              force_laminar: bool = False, band=None) -> dict:
    N_ch  = geom.N_ch
    N_par = geom.N_seg * N_ch
    V  = Q / (N_par * geom.A_ch)
    Re = fluid.rho * V * geom.Dh / fluid.mu

    Nu = Nu_eff(Re, fluid.Pr, geom.L_seg / geom.Dh, geom.aspect, force_laminar, band)
    h  = Nu * fluid.k / geom.Dh
    L_wet_total = N_ch * geom.L_plate
    R_conv = 1.0 / (UA_prime(h, geom) * L_wet_total)
    R_cal  = 1.0 / (fluid.rho * Q * fluid.cp)
    R_th   = R_cal + R_conv + geom.R_base

    L_plus_seg = (geom.L_seg / geom.Dh) / max(Re, 1e-9)
    dyn = 0.5 * fluid.rho * V ** 2
    dP_visc   = f_darcy(Re, L_plus_seg, geom.aspect, force_laminar, band) * (geom.L_seg / geom.Dh) * dyn
    dP_turn   = geom.K_mmc_turn * dyn
    dP_supret = (geom.K_mmc_supply + geom.K_mmc_return) * dyn
    dP = dP_visc + dP_turn + dP_supret

    return dict(arch="MMC", Q=Q, V=V, Re=Re, Nu=Nu, h=h,
                R_caloric=R_cal, R_conv=R_conv, R_base=geom.R_base, R_th=R_th,
                dP_visc=dP_visc, dP_turn=dP_turn, dP_supret=dP_supret,
                dP=dP, W_pump=Q * dP)


if __name__ == "__main__":
    geom, fluid = Geometry(), Fluid()
    q_heat = 100.0

    for Q_lpm in [1, 2, 5, 10, 20, 40, 60]:
        Q = Q_lpm / 60_000.0
        s = model_SMC(geom, fluid, Q, q_heat)
        m = model_MMC(geom, fluid, Q, q_heat)
