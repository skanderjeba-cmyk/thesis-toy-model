"""
Sanity checks & invariants for the two-country rule-based model (updated spec; q-based intensity).

What we verify (rigorously, with tolerances):

0) Finiteness: arrays contain finite values where defined.
1) Positivity: K, N, L, A, Y, E, M (and w, r) are non-negative where defined.
1b) Active-labor mapping (always on, time-varying allowed): L_{i,t} == zeta_{i,t} * N_{i,t}.
2) Budget feasibility: for every country-period, s + kappa * theta^2 ≤ 1.
3) Abatement bounds: 0 ≤ theta ≤ min(theta_bar, sqrt((1-s)/kappa)).
4) Production & prices identities:
     Y = K^alpha (A_i L_i)^(1-alpha);
     w = (1-alpha) Y / L;   r = alpha Y / K.
4b) Per-capita/productivity identities:
     y = Y / N;  q = Y / L.
5) Emissions identity and nonnegativity: E = epsilon_t * (1 - theta) * Y ≥ 0.
5b) Aggregation identity: E_world == E_D + E_L (flows t = 0..T-1).
6) Migration cap (persons): 0 ≤ m_t ≤ min( m_bar N_L,t, (1 + f_L) N_L,t ).
7) Population accounting: N' = (1+f)N ± m.
8) Capital accumulation: K' = (1 - delta)K + sY.
9) Carbon stock recursion: M' = phi_M M + E_world (within tolerance).
10) Ratio condition (advisory): if tail g_E>0 but (1+g_E) ≤ phi_M, warn.

All checks are deterministic; raises AssertionError with a descriptive message on failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .params import ModelParams
from .simulate import SimResult, empirical_growth_rate


@dataclass(frozen=True)
class CheckReport:
    ok: bool
    messages: Tuple[str, ...]


def _is_finite_array(arr: np.ndarray) -> bool:
    return np.all(np.isfinite(arr))


def check_simulation(res: SimResult, mp: ModelParams, *, atol: float = 1e-10, rtol: float = 1e-9) -> CheckReport:
    msgs = []

    # ---- 0 & 1) Finiteness / positivity
    def _pos(name, arr, allow_last_nan: bool = False):
        a = np.array(arr, dtype=float)
        if allow_last_nan and a.size > 0:
            a = a[:-1]
        if a.size > 0 and not _is_finite_array(a):
            msgs.append(f"[{name}] contains non-finite values.")
        if a.size > 0 and np.any(a < -atol):
            idx = int(np.where(a < -atol)[0][0])
            msgs.append(f"[{name}] negative at index {idx}: {a[idx]}")

    # State/process paths
    _pos("A_D", res.A_D)
    _pos("A_L", res.A_L)
    _pos("M", res.M)
    _pos("E_world", res.E_world)
    _pos("m", res.m)

    # Country series (allow last flow entries to be NaN for flow-aligned arrays)
    for label, cs in (("D", res.D), ("L", res.L)):
        _pos(f"K_{label}", cs.K)
        _pos(f"N_{label}", cs.N)
        _pos(f"L_{label}", cs.L)
        _pos(f"Y_{label}", cs.Y, allow_last_nan=True)
        _pos(f"y_{label}", cs.y, allow_last_nan=True)
        _pos(f"theta_{label}", cs.theta, allow_last_nan=True)
        _pos(f"epsilon_t_{label}", cs.epsilon_t, allow_last_nan=True)
        _pos(f"E_{label}", cs.E, allow_last_nan=True)
        _pos(f"w_{label}", cs.w, allow_last_nan=True)
        _pos(f"r_{label}", cs.r, allow_last_nan=True)

    T = res.T

    # ---- 1b) Active-labor mapping: L == zeta_t * N  (all t = 0..T), allowing zeta_schedule
    for label, cs, cp in (("D", res.D, mp.D), ("L", res.L, mp.L)):
        zeta_t = np.array([cp.zeta_at(t) for t in range(len(cs.N))], dtype=float)
        L_id = zeta_t * cs.N
        if not np.allclose(cs.L, L_id, rtol=rtol, atol=atol):
            dif = np.abs(cs.L - L_id)
            thresh = atol + rtol * np.maximum(1.0, np.abs(L_id))
            tbad = int(np.where(dif > thresh)[0][0])
            msgs.append(f"[active labor {label}] L != zeta_t*N at t={tbad} (L={cs.L[tbad]}, zeta_t*N={L_id[tbad]}).")

    # ---- 2) Budget feasibility & 3) Abatement bounds
    for label, cs, cp in (("D", res.D, mp.D), ("L", res.L, mp.L)):
        cap_budget = np.sqrt(max(0.0, (1.0 - cp.s) / cp.kappa))
        theta = np.array(cs.theta[:-1], dtype=float)  # flows only
        if theta.size > 0:
            if np.any(theta < -atol) or np.any(theta > 1.0 + atol):
                msgs.append(f"[{label} theta] outside [0,1] at some t.")
            if np.any(theta > cap_budget + rtol):
                msgs.append(f"[{label} theta] exceeds budget cap sqrt((1-s)/kappa) at some t.")
            if np.any(theta > cp.theta_bar + rtol):
                msgs.append(f"[{label} theta] exceeds theta_bar at some t.")
            feas_lhs = cp.s + cp.kappa * (np.nan_to_num(theta) ** 2)
            if np.any(feas_lhs > 1.0 + 1e-12):
                tbad = int(np.where(feas_lhs > 1.0 + 1e-12)[0][0])
                msgs.append(f"[budget feasibility {label}] violated at t={tbad}: "
                            f"s + kappa*theta^2 = {feas_lhs[tbad]:.6f}")

    # ---- 4) Production & prices identities (flows t=0..T-1) with country-specific A_t
    alpha = mp.globals.alpha
    A_D_t = np.array(res.A_D[:-1], dtype=float)
    A_L_t = np.array(res.A_L[:-1], dtype=float)
    for label, cs, A_t in (("D", res.D, A_D_t), ("L", res.L, A_L_t)):
        Y_id = (cs.K[:-1] ** alpha) * ((A_t * cs.L[:-1]) ** (1.0 - alpha))
        if not np.allclose(cs.Y[:-1], Y_id, rtol=rtol, atol=atol):
            dif = np.abs(cs.Y[:-1] - Y_id)
            thresh = atol + rtol * np.maximum(1.0, np.abs(Y_id))
            tbad = int(np.where(dif > thresh)[0][0])
            msgs.append(f"[production {label}] Y != K^a (A L)^(1-a) at t={tbad}.")
        w_id = (1.0 - alpha) * (cs.Y[:-1] / cs.L[:-1])
        r_id = alpha * (cs.Y[:-1] / cs.K[:-1])
        if not np.allclose(cs.w[:-1], w_id, rtol=rtol, atol=atol):
            dif = np.abs(cs.w[:-1] - w_id)
            thresh = atol + rtol * np.maximum(1.0, np.abs(w_id))
            tbad = int(np.where(dif > thresh)[0][0])
            msgs.append(f"[wage {label}] w != (1-a)Y/L at t={tbad}.")
        if not np.allclose(cs.r[:-1], r_id, rtol=rtol, atol=atol):
            dif = np.abs(cs.r[:-1] - r_id)
            thresh = atol + rtol * np.maximum(1.0, np.abs(r_id))
            tbad = int(np.where(dif > thresh)[0][0])
            msgs.append(f"[rental {label}] r != a Y/K at t={tbad}.")

    # ---- 4b) y and q identities (flows)
    for label, cs in (("D", res.D), ("L", res.L)):
        y_id = cs.Y[:-1] / cs.N[:-1]
        q_id = cs.Y[:-1] / cs.L[:-1]
        if not np.allclose(cs.y[:-1], y_id, rtol=rtol, atol=atol):
            dif = np.abs(cs.y[:-1] - y_id)
            thresh = atol + rtol * np.maximum(1.0, np.abs(y_id))
            tbad = int(np.where(dif > thresh)[0][0])
            msgs.append(f"[y identity {label}] y != Y/N at t={tbad}.")
        if not np.allclose(cs.q[:-1], q_id, rtol=rtol, atol=atol):
            dif = np.abs(cs.q[:-1] - q_id)
            thresh = atol + rtol * np.maximum(1.0, np.abs(q_id))
            tbad = int(np.where(dif > thresh)[0][0])
            msgs.append(f"[q identity {label}] q != Y/L at t={tbad}.")

    # ---- 5) Emissions identity and nonnegativity
    for label, cs in (("D", res.D), ("L", res.L)):
        lhs = cs.E[:-1]
        rhs = cs.epsilon_t[:-1] * (1.0 - cs.theta[:-1]) * cs.Y[:-1]
        if not np.allclose(lhs, rhs, rtol=rtol, atol=atol):
            dif = np.abs(lhs - rhs)
            thresh = atol + rtol * np.maximum(1.0, np.abs(rhs))
            tbad = int(np.where(dif > thresh)[0][0])
            msgs.append(f"[emissions identity {label}] mismatch at t={tbad} (E vs eps*(1-theta)*Y).")
        if np.any(lhs < -atol):
            msgs.append(f"[E_{label}] negative at some t.")

    # ---- 5b) Aggregation identity: E_world == E_D + E_L for flows (t=0..T-1)
    E_world_flow = np.array(res.E_world, dtype=float)
    E_sum_flow = res.D.E[:-1] + res.L.E[:-1]
    if E_world_flow.shape != E_sum_flow.shape:
        msgs.append("[E aggregation] shape mismatch between E_world and E_D+E_L.")
    else:
        if not np.allclose(E_world_flow, E_sum_flow, rtol=rtol, atol=atol):
            dif = np.abs(E_world_flow - E_sum_flow)
            thresh = atol + rtol * np.maximum(1.0, np.abs(E_sum_flow))
            tbad = int(np.where(dif > thresh)[0][0])
            msgs.append(f"[E aggregation] E_world != E_D+E_L at t={tbad}.")

    # ---- 6) Migration cap (persons) for t=0..T-1
    NL = res.L.N[:-1]
    cap1 = mp.migration.m_bar * NL
    cap2 = (1.0 + mp.L.f) * NL
    cap = np.minimum(cap1, cap2)
    if np.any(res.m < -atol):
        msgs.append("[migration] m_t < 0 at some t.")
    if np.any(res.m - cap > atol):
        tbad = int(np.where(res.m - cap > atol)[0][0])
        msgs.append(f"[migration] cap violated at t={tbad}: m={res.m[tbad]}, cap={cap[tbad]}.")

    # ---- 7) Population accounting: N' = (1+f)N ± m
    NDp = res.D.N[:-1]
    NLp = res.L.N[:-1]
    NDn = res.D.N[1:]
    NLn = res.L.N[1:]
    ND_id = (1.0 + mp.D.f) * NDp + res.m
    NL_id = (1.0 + mp.L.f) * NLp - res.m
    if not np.allclose(NDn, ND_id, rtol=rtol, atol=atol):
        dif = np.abs(NDn - ND_id)
        thresh = atol + rtol * np.maximum(1.0, np.abs(ND_id))
        tbad = int(np.where(dif > thresh)[0][0])
        msgs.append(f"[population D] accounting mismatch at t={tbad}.")
    if not np.allclose(NLn, NL_id, rtol=rtol, atol=atol):
        dif = np.abs(NLn - NL_id)
        thresh = atol + rtol * np.maximum(1.0, np.abs(NL_id))
        tbad = int(np.where(dif > thresh)[0][0])
        msgs.append(f"[population L] accounting mismatch at t={tbad}.")

    # ---- 8) Capital accumulation: K' = (1 - delta)K + sY
    KDp = res.D.K[:-1]
    KLp = res.L.K[:-1]
    KDn = res.D.K[1:]
    KLn = res.L.K[1:]
    KD_id = (1.0 - mp.globals.delta) * KDp + mp.D.s * res.D.Y[:-1]
    KL_id = (1.0 - mp.globals.delta) * KLp + mp.L.s * res.L.Y[:-1]
    if not np.allclose(KDn, KD_id, rtol=rtol, atol=atol):
        dif = np.abs(KDn - KD_id)
        thresh = atol + rtol * np.maximum(1.0, np.abs(KD_id))
        tbad = int(np.where(dif > thresh)[0][0])
        msgs.append(f"[capital D] accumulation mismatch at t={tbad}.")
    if not np.allclose(KLn, KL_id, rtol=rtol, atol=atol):
        dif = np.abs(KLn - KL_id)
        thresh = atol + rtol * np.maximum(1.0, np.abs(KL_id))
        tbad = int(np.where(dif > thresh)[0][0])
        msgs.append(f"[capital L] accumulation mismatch at t={tbad}.")

    # ---- 9) Carbon recursion: M' = phi_M * M + E_world
    Mp = res.M[:-1]
    Mn = res.M[1:]
    Mrhs = mp.globals.phi_M * Mp + E_world_flow
    if not np.allclose(Mn, Mrhs, rtol=rtol, atol=atol):
        dif = np.abs(Mn - Mrhs)
        thresh = atol + rtol * np.maximum(1.0, np.abs(Mrhs))
        tbad = int(np.where(dif > thresh)[0][0])
        msgs.append(f"[carbon] recursion mismatch at t={tbad}.")

    # ---- 10) Ratio advisory (BGP condition)
    window = int(min(20, max(2, len(E_world_flow) - 1)))
    gE = empirical_growth_rate(E_world_flow, window=window)
    if gE == gE and (1.0 + gE) <= mp.globals.phi_M + 1e-12:
        msgs.append(f"[ratio advisory] g_E≈{gE:.6f} violates (1+g_E) > phi_M={mp.globals.phi_M:.6f}; "
                    "M/E ratio limit would not exist. (Advisory only)")

    return CheckReport(ok=(len(msgs) == 0), messages=tuple(msgs))


def assert_simulation_ok(res: SimResult, mp: ModelParams, *, atol: float = 1e-10, rtol: float = 1e-9) -> None:
    """
    Run all checks and raise AssertionError on the first failure.
    """
    report = check_simulation(res, mp, atol=atol, rtol=rtol)
    if not report.ok:
        raise AssertionError("Simulation invariants failed:\n  - " + "\n  - ".join(report.messages))
