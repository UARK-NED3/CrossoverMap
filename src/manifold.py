"""Manifold arm pressure drop for the end-fed tapered MMC."""
from __future__ import annotations

import numpy as np

ARM_H0 = 12.0e-3
ARM_HL = 2.0e-3


def f_Re_rect(a: float) -> float:
    return 4.0 * 24.0 * (1.0 - 1.3553 * a + 1.9467 * a ** 2 - 1.7012 * a ** 3
                         + 0.9564 * a ** 4 - 0.2537 * a ** 5)


def taper_integral(w: float, h0: float = ARM_H0, hL: float = ARM_HL, *,
                   L: float) -> float:
    return L * (1.0 / (4.0 * h0 * hL)
                + np.log(h0 / hL) / (2.0 * w * (h0 - hL))
                + 1.0 / (4.0 * w ** 2))


def v_arm(Q: float, w: float, n_seg: int, h0: float = ARM_H0) -> float:
    return Q / ((n_seg / 2.0) * w * h0)


def dP_arm_viscous(Q: float, w: float, n_seg: int, mu: float,
                   h0: float = ARM_H0, hL: float = ARM_HL, *, L: float) -> float:
    C = 0.5 * (f_Re_rect(min(w, h0) / max(w, h0)) + f_Re_rect(min(w, hL) / max(w, hL)))
    return 2.0 * (C * mu * v_arm(Q, w, n_seg, h0) * taper_integral(w, h0, hL, L=L) / 2.0)


def dP_manifold(Q: float, w: float, n_seg: int, fluid, k_eff: float,
                h0: float = ARM_H0, hL: float = ARM_HL, *, L: float) -> float:
    V = v_arm(Q, w, n_seg, h0)
    return k_eff * 0.5 * fluid.rho * V ** 2


def equivalent_K(geom, Q: float, fluid, k_eff: float) -> float:
    n = geom.N_seg
    V_ch = Q / (n * geom.N_ch * geom.A_ch)
    if V_ch <= 0.0:
        return 0.0
    return (dP_manifold(Q, geom.slot, n, fluid, k_eff, L=geom.W_plate)
            / (0.5 * fluid.rho * V_ch ** 2))
