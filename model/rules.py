"""
Core rule functions (pure & testable).

Implements:
- Production and factor prices
- Abatement rule with feasibility cap
- q-based emissions intensity: epsilon(q) = epsilon * (q/q_star)^eta
- Wages per active worker
- Persons-scaled, one-way migration rule with headcount wedge and feasibility cap
- Capital and population updates (with active labor L = zeta * N)
- Emissions and carbon stock update
- Basic feasibility checks and a positive-part helper
- TFP update and recorded ratio helper (M_{t+1}/E_t)

All functions are deterministic, side-effect free, and designed to be unit-tested.
They mirror the updated model document and respect the invariants described there.
"""

from __future__ import annotations

from typing import Tuple, Dict, Optional
import math

from .params import (
    CountryParams,
    GlobalParams,
    MigrationParams,
)

# ----------------------------
# Small helpers
# ----------------------------

def pos(x: float) -> float:
    """Positive-part operator: max{x, 0}."""
    return x if x > 0.0 else 0.0


def clip(x: float, lo: float, hi: float) -> float:
    """Clip x to [lo, hi]."""
    return min(max(x, lo), hi)


def wage_ratio(wD: float, wL: float) -> float:
    """Wage ratio varrho = w_D / w_L with a guard on w_L>0."""
    if wL <= 0:
        raise ValueError(f"wage_ratio requires wL>0; got wL={wL}")
    return wD / wL


# ----------------------------
# Production & prices
# ----------------------------

def produce(K: float, A: float, L: float, alpha: float) -> float:
    """
    Cobb–Douglas output: Y = K^alpha (A L)^(1-alpha).
    Requires K>0, A>0, L>0, alpha∈(0,1).
    """
    if K <= 0 or A <= 0 or L <= 0:
        raise ValueError(f"produce requires K>0, A>0, L>0; got K={K}, A={A}, L={L}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1); got {alpha}")
    return (K ** alpha) * ((A * L) ** (1.0 - alpha))


def wages_per_active_worker(Y: float, L: float, alpha: float) -> float:
    """
    Wage per active worker (marginal product of labor):
    w = (1 - alpha) * Y / L.
    Requires Y>0, L>0.
    """
    if Y <= 0 or L <= 0:
        raise ValueError(f"wage requires Y>0 and L>0; got Y={Y}, L={L}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1); got {alpha}")
    return (1.0 - alpha) * (Y / L)


def rental_rate(Y: float, K: float, alpha: float) -> float:
    """
    Rental rate of capital (marginal product of capital):
    r = alpha * Y / K. Requires Y>0, K>0.
    """
    if Y <= 0 or K <= 0:
        raise ValueError(f"rental requires Y>0 and K>0; got Y={Y}, K={K}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1); got {alpha}")
    return alpha * (Y / K)


# ----------------------------
# Abatement helpers and intensity rules
# ----------------------------

def theta_from_y(y: float, s: float, kappa: float, xi: float, theta_bar: float) -> float:
    """
    Per-capita abatement rule theta(y) = min{theta_bar, xi*y, sqrt((1-s)/kappa)} clipped to [0,1].
    Pure helper (idempotent; no side effects).
    """
    if y < 0:
        raise ValueError(f"theta_from_y requires y>=0; got y={y}")
    if not (0.0 < theta_bar <= 1.0):
        raise ValueError(f"theta_bar must be in (0,1]; got {theta_bar}")
    if not (0.0 < kappa):
        raise ValueError(f"kappa must be > 0; got {kappa}")
    if not (0.0 < s < 1.0):
        raise ValueError(f"s must be in (0,1); got {s}")
    if xi < 0.0:
        raise ValueError(f"xi must be ≥ 0; got {xi}")

    cap = math.sqrt(max(0.0, (1.0 - s) / kappa))
    return clip(min(theta_bar, xi * y, cap), 0.0, 1.0)


def theta_rule(y: float, cp: CountryParams) -> float:
    """
    Abatement rule (per-capita income responsive with feasibility cap), using country params:
        theta = min{ theta_bar, xi*y, sqrt((1 - s)/kappa) }, clipped to [0,1].
    Requires y>=0 (we allow y=0 ⇒ theta=0).
    """
    return theta_from_y(y, s=cp.s, kappa=cp.kappa, xi=cp.xi, theta_bar=cp.theta_bar)


def epsilon_rule(q: float, cp: CountryParams, gp: GlobalParams) -> float:
    """
    Emissions intensity as a function of labor productivity q = Y/L:
        epsilon_t = epsilon * (q / q_star)^eta
    Requires q>0, q_star>0 (validated in params).
    """
    if q <= 0:
        raise ValueError(f"epsilon_rule requires q>0; got q={q}")
    return cp.epsilon * ((q / gp.q_star) ** gp.eta)


# ----------------------------
# Migration (persons-scaled, one-way L→D with headcount wedge)
# ----------------------------

def migration_flow(
    wD: float,
    wL: float,
    NL: float,
    mig: MigrationParams,
    f_L: float,
) -> float:
    """
    One-way migration from L to D, scaled by persons N_L,t, with feasibility cap.

      varrho  = wD / wL   (well-defined since wL>0)
      tilde_m = [ mu * varrho * N_L  -  tau_H ]_+
      cap     = min( m_bar * N_L, (1 + f_L) * N_L )
      m       = min( tilde_m, cap )

    Preconditions: wL>0, NL>0.
    """
    if wL <= 0:
        raise ValueError(f"migration_flow requires wL>0; got wL={wL}")
    if NL <= 0:
        raise ValueError(f"migration_flow requires NL>0; got NL={NL}")

    varrho = wD / wL
    tilde_m = pos(mig.mu * varrho * NL - mig.tau_H)
    cap = min(mig.m_bar * NL, (1.0 + f_L) * NL)
    return clip(tilde_m, 0.0, cap)


# ----------------------------
# Capital accumulation & TFP
# ----------------------------

def capital_next(K: float, Y: float, s: float, delta: float) -> float:
    """
    Capital accumulation with constant savings:
      I = s * Y
      K' = (1 - delta) * K + I
    Requires K>=0, Y>=0, s∈(0,1), delta∈(0,1).
    """
    if K < 0 or Y < 0:
        raise ValueError(f"capital_next requires K>=0 and Y>=0; got K={K}, Y={Y}")
    if not (0.0 < s < 1.0):
        raise ValueError(f"s must be in (0,1); got {s}")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0,1); got {delta}")
    return (1.0 - delta) * K + s * Y


def tfp_next(A: float, g_A: float) -> float:
    """TFP update: A' = (1 + g_A) * A; requires 1+g_A > 0 handled in params validation."""
    return (1.0 + g_A) * A


# ----------------------------
# Demography and active labor
# ----------------------------

def population_next(
    ND: float,
    NL: float,
    f_D: float,
    f_L: float,
    m: float,
) -> Tuple[float, float]:
    """
    Population accounting at end of period t:
      N_D' = (1 + f_D) N_D + m
      N_L' = (1 + f_L) N_L - m
    Requires ND>0, NL>0, and 0 ≤ m ≤ min(m_bar N_L, (1+f_L) N_L) elsewhere.
    """
    if ND <= 0 or NL <= 0:
        raise ValueError(f"population_next requires ND>0 and NL>0; got ND={ND}, NL={NL}")
    ND_next = (1.0 + f_D) * ND + m
    NL_next = (1.0 + f_L) * NL - m
    if ND_next <= 0 or NL_next <= 0:
        raise ValueError(f"population_next produced nonpositive values ND'={ND_next}, NL'={NL_next}")
    return ND_next, NL_next


def active_labor_from_population(zeta: float, N: float) -> float:
    """
    Active labor: L = zeta * N, with zeta in (0,1]. Baseline uses constant zeta.
    """
    if not (0.0 < zeta <= 1.0):
        raise ValueError(f"zeta must be in (0,1]; got {zeta}")
    if N <= 0:
        raise ValueError(f"N must be > 0; got {N}")
    return zeta * N


# ----------------------------
# Emissions and carbon stock
# ----------------------------

def emissions(epsilon_t: float, theta: float, Y: float) -> float:
    """
    Emissions proportional to output:
      E = epsilon_t * (1 - theta) * Y
    Requires epsilon_t>=0, theta in [0,1], Y>=0.
    """
    if epsilon_t < 0:
        raise ValueError(f"epsilon_t must be ≥ 0; got {epsilon_t}")
    if not (0.0 <= theta <= 1.0):
        raise ValueError(f"theta must be in [0,1]; got {theta}")
    if Y < 0:
        raise ValueError(f"Y must be ≥ 0; got {Y}")
    return epsilon_t * (1.0 - theta) * Y


def carbon_next(M: float, E_sum: float, phi_M: float) -> float:
    """
    Carbon stock dynamics:
      M' = phi_M * M + E_sum
    Requires phi_M in (0,1), M≥0, E_sum≥0.
    """
    if M < 0 or E_sum < 0:
        raise ValueError(f"M and E_sum must be ≥ 0; got M={M}, E_sum={E_sum}")
    if not (0.0 < phi_M < 1.0):
        raise ValueError(f"phi_M must be in (0,1); got {phi_M}")
    return phi_M * M + E_sum


def recorded_ratio(M_next: float, E_t: float) -> float:
    """
    Recorded ratio helper for figures/tables:
      ratio_rec_t = M_{t+1} / E_t
    Returns float('inf') if E_t == 0 to avoid ZeroDivisionError.
    """
    if E_t < 0 or M_next < 0:
        raise ValueError(f"recorded_ratio requires nonnegative inputs; got M_next={M_next}, E_t={E_t}")
    return float('inf') if E_t == 0.0 else (M_next / E_t)


# ----------------------------
# Feasibility checks
# ----------------------------

def check_budget_feasibility(theta: float, cp: CountryParams) -> None:
    """
    Per-period consumption feasibility requires:
      s + kappa * theta^2 ≤ 1
    (This is ensured if theta ≤ sqrt((1 - s)/kappa). The theta_rule implements this cap.)
    """
    lhs = cp.s + cp.kappa * (theta ** 2)
    if lhs > 1.0 + 1e-12:
        raise ValueError(
            f"Budget feasibility violated: s + kappa*theta^2 = {lhs:.6f} > 1. "
            f"Consider reducing theta or adjusting (s, kappa)."
        )


# ----------------------------
# Convenience wrappers for country i
# ----------------------------

def country_output_and_prices(
    K: float,
    A: float,
    N: float,
    cp: CountryParams,
    gp: GlobalParams,
    zeta_override: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute Y, L, w, r, y, q for one country given (K, A, N).
    Active labor is always used: L = zeta * N.
    If zeta_override is provided, use it (to support time-varying zeta_t); else use cp.zeta.
    Returns dict with keys: Y, L, w, r, y, q.
    """
    zeta = zeta_override if zeta_override is not None else cp.zeta
    L = active_labor_from_population(zeta, N)
    Y = produce(K, A, L, gp.alpha)
    w = wages_per_active_worker(Y, L, gp.alpha)
    r = rental_rate(Y, K, gp.alpha)
    y = Y / N
    q = Y / L
    return {"Y": Y, "L": L, "w": w, "r": r, "y": y, "q": q}


def country_emissions_and_theta(
    y: float,
    q: float,
    Y: float,
    cp: CountryParams,
    gp: GlobalParams,
) -> Dict[str, float]:
    """
    Compute theta (abatement), epsilon_t(q) (intensity), and E (emissions) for one country.
    Inputs: y = Y/N, q = Y/L, and Y.
    """
    theta = theta_rule(y, cp)
    check_budget_feasibility(theta, cp)
    eps_t = epsilon_rule(q, cp, gp)
    E = emissions(eps_t, theta, Y)
    return {"theta": theta, "epsilon_t": eps_t, "E": E}
