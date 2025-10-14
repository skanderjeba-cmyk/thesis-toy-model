"""
Closed-form Solow steady states (updated model).

Implements:
- effective_labor_growth_rate: growth of efficiency units [(1+g_A)(1+n) - 1]
- k_hat_star: steady-state capital per efficiency unit
- y_per_worker_star: normalized per-active-worker output (per A), i.e. (k_hat*)^alpha
- q_star_from_A: per-active-worker output q* = A * (k_hat*)^alpha
- y_per_capita_star: per-capita income y* = zeta * q*
- steady_state_country: wrapper returning k_hat*, q*, y*, theta*, epsilon_t*, e_per_capita*
- carbon_stationary_level: M* = E_sum / (1 - phi_M) when g_E = 0
- carbon_ratio_limit:  M_t / E_t -> 1 / ((1+g_E) - phi_M) when (1+g_E) > phi_M
- carbon_ratio_recorded_limit:  M_{t+1} / E_t -> (1+g_E) / ((1+g_E) - phi_M)

Notes
-----
• Active labor is always L = zeta * N (no toggle).
• Emissions intensity depends on productivity q = Y/L via epsilon(q) = ε * (q / q_star)^η.
• The closed-form fixed point is in efficiency units; levels scale with A.
"""

from __future__ import annotations

from typing import Dict, Optional

from .params import CountryParams, GlobalParams
from .rules import (
    theta_rule,     # theta(y) = min{theta_bar, xi*y, sqrt((1-s)/kappa)}
    epsilon_rule,   # epsilon(q) = ε * (q / q_star)^η (uses gp.eta, gp.q_star)
)

__all__ = [
    "effective_labor_growth_rate",
    "k_hat_star",
    "y_per_worker_star",
    "q_star_from_A",
    "y_per_capita_star",
    "steady_state_country",
    "carbon_stationary_level",
    "carbon_ratio_limit",
    "carbon_ratio_recorded_limit",
]


# ----------------------------
# Effective labor growth
# ----------------------------

def effective_labor_growth_rate(n: float, g_A: float) -> float:
    """
    Growth rate of effective labor (per period), where n is the ACTIVE-labor growth rate:
        (1+g_A)(1+n) - 1 = g_A + n + g_A*n
    Requires n > -1 and 1+g_A > 0.
    """
    if n <= -1.0:
        raise ValueError(f"n must be > -1; got {n}")
    if 1.0 + g_A <= 0.0:
        raise ValueError(f"1+g_A must be > 0; got g_A={g_A}")
    return g_A + n + g_A * n


# ----------------------------
# Steady-state formulas
# ----------------------------

def k_hat_star(s: float, delta: float, g_A: float, n: float, alpha: float) -> float:
    """
    Steady-state capital per efficiency unit:
        k_hat* = [ s / (delta + g_A + n + g_A*n) ]^(1/(1-alpha))

    Canonical argument order: (s, delta, g_A, n, alpha).

    Requires s∈(0,1), delta∈(0,1), alpha∈(0,1), n>-1, 1+g_A>0, and
    (delta + g_A + n + g_A*n) > 0.
    """
    if not (0.0 < s < 1.0):
        raise ValueError(f"s must be in (0,1); got {s}")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0,1); got {delta}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1); got {alpha}")
    if not (n > -1.0):
        raise ValueError(f"n must be > -1; got {n}")
    if not (1.0 + g_A > 0.0):
        raise ValueError(f"1+g_A must be > 0; got g_A={g_A}")

    denom = delta + g_A + n + g_A * n
    if denom <= 0.0:
        raise ValueError(f"delta+g_A+n+g_A*n must be > 0; got {denom:.6f}")

    return (s / denom) ** (1.0 / (1.0 - alpha))


def y_per_worker_star(k_hat: float, alpha: float) -> float:
    """
    Normalized per-active-worker output (per A):
        (Y/L)/A = (k_hat)^alpha.
    Returns (k_hat)^alpha. To obtain q* = (Y/L)*, multiply by A.
    """
    if k_hat <= 0.0:
        raise ValueError(f"k_hat must be > 0; got {k_hat}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1); got {alpha}")
    return k_hat ** alpha


def q_star_from_A(A: float, k_hat: float, alpha: float) -> float:
    """
    Per-active-worker output level:
        q* = (Y/L)* = A * (k_hat)^alpha.
    """
    if A <= 0.0:
        raise ValueError(f"A must be > 0; got {A}")
    return A * y_per_worker_star(k_hat, alpha)


def y_per_capita_star(zeta: float, A: float, k_hat: float, alpha: float) -> float:
    """
    Per-capita income in steady state:
        y* = zeta * q* = zeta * A * (k_hat)^alpha.
    """
    if not (0.0 < zeta <= 1.0):
        raise ValueError(f"zeta must be in (0,1]; got {zeta}")
    return zeta * q_star_from_A(A, k_hat, alpha)


# ----------------------------
# Country-level steady state wrapper
# ----------------------------

def steady_state_country(
    cp: CountryParams,
    gp: GlobalParams,
    n: float,
    A_level: float = 1.0,
) -> Dict[str, float]:
    """
    Closed-form steady state for one country given:
      - demographics (active-labor growth) n,
      - global params gp,
      - country params cp,
      - an A_level (country-specific productivity level at which to report).

    Returns dict:
      {
        "k_hat": k_hat*,
        "q":     q* = A * (k_hat)^alpha,
        "y":     y* = zeta * q*,
        "theta": theta* = min{theta_bar, xi*y*, sqrt((1-s)/kappa)},
        "epsilon_t": epsilon(q*),
        "e_per_capita": e* = epsilon_t * (1-theta) * y*
      }

    Notes
    -----
    • This wrapper reports per-capita emissions e* rather than total E*
      (the latter requires N* which is not pinned by Solow in levels).
    """
    k_hat = k_hat_star(cp.s, gp.delta, gp.g_A, n, gp.alpha)
    q_star = q_star_from_A(A_level, k_hat, gp.alpha)
    y_star = cp.zeta * q_star

    theta = theta_rule(y_star, cp)  # budget feasibility ensured by cap inside theta_rule
    eps_t = epsilon_rule(q_star, cp, gp)
    e_star = eps_t * (1.0 - theta) * y_star  # per-capita emissions

    return {
        "k_hat": k_hat,
        "q": q_star,
        "y": y_star,
        "theta": theta,
        "epsilon_t": eps_t,
        "e_per_capita": e_star,
    }


# ----------------------------
# Carbon stock limits
# ----------------------------

def carbon_stationary_level(E_sum: float, phi_M: float) -> Optional[float]:
    """
    Stationary carbon stock if aggregate emissions are constant (g_E = 0):
        M* = E_sum / (1 - phi_M).
    Requires phi_M ∈ (0,1).
    """
    if not (0.0 < phi_M < 1.0):
        raise ValueError(f"phi_M must be in (0,1); got {phi_M}")
    return E_sum / (1.0 - phi_M)


def carbon_ratio_limit(g_E: float, phi_M: float) -> float:
    """
    Asymptotic ratio when aggregate emissions grow at rate g_E:
        M_t / E_t → 1 / ( (1+g_E) - phi_M ).
    Valid whenever (1+g_E) > phi_M (includes g_E = 0).
    """
    if not (0.0 < phi_M < 1.0):
        raise ValueError(f"phi_M must be in (0,1); got {phi_M}")
    if (1.0 + g_E) <= phi_M:
        raise ValueError(f"Condition (1+g_E) > phi_M required; got g_E={g_E}, phi_M={phi_M}")
    return 1.0 / ((1.0 + g_E) - phi_M)


def carbon_ratio_recorded_limit(g_E: float, phi_M: float) -> float:
    """
    Asymptotic 'recorded' ratio used in figures/tables:
        M_{t+1} / E_t → (1+g_E) / ( (1+g_E) - phi_M ).
    Valid whenever (1+g_E) > phi_M (includes g_E = 0).
    """
    if not (0.0 < phi_M < 1.0):
        raise ValueError(f"phi_M must be in (0,1); got {phi_M}")
    if (1.0 + g_E) <= phi_M:
        raise ValueError(f"Condition (1+g_E) > phi_M required; got g_E={g_E}, phi_M={phi_M}")
    return (1.0 + g_E) / ((1.0 + g_E) - phi_M)
