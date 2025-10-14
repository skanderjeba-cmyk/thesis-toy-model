"""
Deterministic time-path simulator for the two-region toy model (updated model).

Timing at each period t (states observed at start of t):
  1) Active labor: L_{i,t} = zeta_{i,t} * N_{i,t}  (active labor always used).
  2) Production:   Y_{i,t} = K_{i,t}^α (A_{i,t} L_{i,t})^(1-α).
  3) Prices:       w_{i,t} = (1-α) Y_{i,t} / L_{i,t} ; r_{i,t} = α Y_{i,t} / K_{i,t}.
  4) Rules:
       y_{i,t} = Y_{i,t} / N_{i,t} , q_{i,t} = Y_{i,t} / L_{i,t}
       θ_{i,t} = min{ θ̄_i, ξ_i y_{i,t}, sqrt((1-s_i)/κ_i) }
       ε_{i,t} = ε_i * (q_{i,t} / q★)^η   # intensity uses q = Y/L
  5) Emissions:    E_{i,t} = ε_{i,t} (1 - θ_{i,t}) Y_{i,t}.
  6) Migration (persons-scaled, one-way L→D, headcount wedge):
       ρ_t = w_{D,t}/w_{L,t}
       \tilde m_t = [ μ * ρ_t * N_{L,t} - τ_H ]_+
       m_t = min{ \tilde m_t, m̄ N_{L,t}, (1 + f_L) N_{L,t} }.
  7) Capital:      K_{i,t+1} = (1-δ)K_{i,t} + s_i Y_{i,t}.
  8) Population:   N_{D,t+1} = (1+f_D)N_{D,t} + m_t ;
                   N_{L,t+1} = (1+f_L)N_{L,t} - m_t.
  9) Productivity: A_{i,t+1} = (1+g_A) A_{i,t}  (common g_A, country-specific levels).
 10) Carbon:       M_{t+1} = φ_M M_t + ∑_i E_{i,t}.

Returns full paths for D and L, plus world aggregates and the M/E ratio (recorded as M_{t+1}/E_t).
All feasibility checks are enforced by rule functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .params import ModelParams
from .rules import (
    country_output_and_prices,
    country_emissions_and_theta,
    migration_flow,
    capital_next,
    population_next,
    carbon_next,
    tfp_next,
    recorded_ratio,
)


@dataclass(frozen=True)
class CountrySeries:
    K: np.ndarray           # capital stock (t=0..T)
    N: np.ndarray           # population (t=0..T)
    L: np.ndarray           # active labor (t=0..T)
    Y: np.ndarray           # output (t=0..T-1 valid; last NaN)
    w: np.ndarray           # wage per active worker (t=0..T-1; last NaN)
    r: np.ndarray           # rental rate of capital (t=0..T-1; last NaN)
    y: np.ndarray           # output per capita (t=0..T-1; last NaN)
    q: np.ndarray           # output per active worker (t=0..T-1; last NaN)
    theta: np.ndarray       # abatement share (t=0..T-1; last NaN)
    epsilon_t: np.ndarray   # emissions intensity (t=0..T-1; last NaN)
    E: np.ndarray           # emissions (t=0..T-1; last NaN)


@dataclass(frozen=True)
class SimResult:
    T: int
    tgrid: np.ndarray
    # Country-specific productivity paths; A mirrors A_D for backward compatibility
    A: np.ndarray
    A_D: np.ndarray
    A_L: np.ndarray
    M: np.ndarray
    E_world: np.ndarray       # world emissions in period t (length T)
    M_over_E: np.ndarray      # ratio recorded as M_{t+1} / E_t (length T+1; index 0 NaN)
    m: np.ndarray             # migration flow in period t (length T)
    D: CountrySeries
    L: CountrySeries


def _nan_pad(length: int) -> np.ndarray:
    a = np.empty(length, dtype=float)
    a[:] = np.nan
    return a


def _read_init_pair(state, dict_name: str, fieldD: str, fieldL: str):
    """
    Read a D/L pair from either a dict attribute (e.g., state.K['D'])
    or from separate fields (e.g., state.K_D, state.K_L).
    """
    if hasattr(state, dict_name):
        v = getattr(state, dict_name)
        if isinstance(v, dict) and ("D" in v) and ("L" in v):
            return float(v["D"]), float(v["L"])
    return float(getattr(state, fieldD)), float(getattr(state, fieldL))


def simulate(mp: ModelParams, T: Optional[int] = None) -> SimResult:
    """
    Run the simulator for T periods (default: mp.simulation.T).
    Stock arrays (K, N, L, A_D, A_L, M) have length T+1 (include t=0).
    Flow arrays (Y, w, r, y, q, theta, epsilon_t, E, m, E_world) have length T.
    """
    if T is None:
        T = mp.simulation.T
    if not (isinstance(T, int) and T >= 1):
        raise ValueError(f"T must be an integer ≥ 1; got {T}")

    gp = mp.globals
    Dp = mp.D
    Lp = mp.L
    mig = mp.migration

    # Allocate arrays
    tgrid = np.arange(T + 1, dtype=int)

    # Stocks
    K_D = np.empty(T + 1, dtype=float); K_L = np.empty(T + 1, dtype=float)
    N_D = np.empty(T + 1, dtype=float); N_L = np.empty(T + 1, dtype=float)
    L_D = np.empty(T + 1, dtype=float); L_L = np.empty(T + 1, dtype=float)
    A_D = np.empty(T + 1, dtype=float); A_L = np.empty(T + 1, dtype=float)
    M   = np.empty(T + 1, dtype=float)

    # Flows (length T)
    Y_D = _nan_pad(T); Y_L = _nan_pad(T)
    w_D = _nan_pad(T); w_L = _nan_pad(T)
    r_D = _nan_pad(T); r_L = _nan_pad(T)
    y_D = _nan_pad(T); y_L = _nan_pad(T)
    q_D = _nan_pad(T); q_L = _nan_pad(T)
    theta_D = _nan_pad(T); theta_L = _nan_pad(T)
    eps_D = _nan_pad(T); eps_L = _nan_pad(T)
    E_D = _nan_pad(T); E_L = _nan_pad(T)
    m = _nan_pad(T)
    E_world = _nan_pad(T)
    M_over_E = _nan_pad(T + 1)  # record M_{t+1}/E_t at index t+1; M_over_E[0]=NaN

    # ---------- Initial conditions (accept dict-style or field-style) ----------
    K_D0, K_L0 = _read_init_pair(mp.initial_state, "K",  "K_D",  "K_L")
    N_D0, N_L0 = _read_init_pair(mp.initial_state, "N",  "N_D",  "N_L")
    A_D0, A_L0 = _read_init_pair(mp.initial_state, "A0", "A0_D", "A0_L")
    M0 = float(getattr(mp.initial_state, "M0", 0.0))

    K_D[0] = K_D0
    K_L[0] = K_L0
    N_D[0] = N_D0
    N_L[0] = N_L0
    A_D[0] = A_D0
    A_L[0] = A_L0
    M[0]   = M0
    # -------------------------------------------------------------------------

    # Precompute active-labor shares zeta_t (supports zeta_schedule cleanly)
    zetaD_path = np.array([Dp.zeta_at(t) for t in range(T + 1)], dtype=float)
    zetaL_path = np.array([Lp.zeta_at(t) for t in range(T + 1)], dtype=float)

    # Iterate
    for t in range(T):
        # (1) Active labor at start of t
        L_D[t] = zetaD_path[t] * N_D[t]
        L_L[t] = zetaL_path[t] * N_L[t]

        # Positivity of states (guards)
        if not (K_D[t] > 0 and K_L[t] > 0 and N_D[t] > 0 and N_L[t] > 0 and
                A_D[t] > 0 and A_L[t] > 0 and L_D[t] > 0 and L_L[t] > 0):
            raise ValueError(f"Nonpositive state encountered at t={t}")

        # (2-3) Output & prices (country-specific A)
        outD = country_output_and_prices(K_D[t], A_D[t], N_D[t], Dp, gp, zeta_override=zetaD_path[t])
        outL = country_output_and_prices(K_L[t], A_L[t], N_L[t], Lp, gp, zeta_override=zetaL_path[t])

        Y_D[t], L_D[t], w_D[t], r_D[t], y_D[t], q_D[t] = (
            outD["Y"], outD["L"], outD["w"], outD["r"], outD["y"], outD["q"]
        )
        Y_L[t], L_L[t], w_L[t], r_L[t], y_L[t], q_L[t] = (
            outL["Y"], outL["L"], outL["w"], outL["r"], outL["y"], outL["q"]
        )

        # (4-5) Abatement, intensity(q), emissions
        emiD = country_emissions_and_theta(y_D[t], q_D[t], Y_D[t], Dp, gp)
        emiL = country_emissions_and_theta(y_L[t], q_L[t], Y_L[t], Lp, gp)

        theta_D[t], eps_D[t], E_D[t] = emiD["theta"], emiD["epsilon_t"], emiD["E"]
        theta_L[t], eps_L[t], E_L[t] = emiL["theta"], emiL["epsilon_t"], emiL["E"]

        # (6) Migration (persons-scaled; wage ratio with headcount wedge)
        m[t] = migration_flow(wD=w_D[t], wL=w_L[t], NL=N_L[t], mig=mig, f_L=Lp.f)

        # (7) Capital updates
        K_D[t + 1] = capital_next(K_D[t], Y_D[t], Dp.s, gp.delta)
        K_L[t + 1] = capital_next(K_L[t], Y_L[t], Lp.s, gp.delta)

        # (8) Population updates
        N_D[t + 1], N_L[t + 1] = population_next(
            ND=N_D[t], NL=N_L[t], f_D=Dp.f, f_L=Lp.f, m=m[t]
        )

        # (9) Productivity updates (common g_A, country-specific levels)
        A_D[t + 1] = tfp_next(A_D[t], gp.g_A)
        A_L[t + 1] = tfp_next(A_L[t], gp.g_A)
        if A_D[t + 1] <= 0 or A_L[t + 1] <= 0:
            raise ValueError(f"A became nonpositive at t={t+1}; check g_A")

        # (10) Carbon update
        E_world[t] = E_D[t] + E_L[t]
        M[t + 1] = carbon_next(M[t], E_world[t], gp.phi_M)

        # Record ratio as M_{t+1}/E_t (use helper for robust handling of E_t=0)
        M_over_E[t + 1] = recorded_ratio(M[t + 1], E_world[t])

    # Final active labor (t = T) after N updates
    L_D[T] = zetaD_path[T] * N_D[T]
    L_L[T] = zetaL_path[T] * N_L[T]

    # Pack results
    D_series = CountrySeries(
        K=K_D, N=N_D, L=L_D,
        Y=np.append(Y_D, np.nan),
        w=np.append(w_D, np.nan),
        r=np.append(r_D, np.nan),
        y=np.append(y_D, np.nan),
        q=np.append(q_D, np.nan),
        theta=np.append(theta_D, np.nan),
        epsilon_t=np.append(eps_D, np.nan),
        E=np.append(E_D, np.nan),
    )
    L_series = CountrySeries(
        K=K_L, N=N_L, L=L_L,
        Y=np.append(Y_L, np.nan),
        w=np.append(w_L, np.nan),
        r=np.append(r_L, np.nan),
        y=np.append(y_L, np.nan),
        q=np.append(q_L, np.nan),
        theta=np.append(theta_L, np.nan),
        epsilon_t=np.append(eps_L, np.nan),
        E=np.append(E_L, np.nan),
    )

    return SimResult(
        T=T,
        tgrid=tgrid,
        A=A_D.copy(),   # backward-compatibility; primary paths are A_D and A_L
        A_D=A_D,
        A_L=A_L,
        M=M,
        E_world=E_world,
        M_over_E=M_over_E,
        m=m,
        D=D_series,
        L=L_series,
    )


# -----------------------------------
# Optional helpers for analysis
# -----------------------------------

def empirical_growth_rate(series: np.ndarray, window: int = 20) -> float:
    """
    Approximate long-run growth rate g from a positive series (e.g., E_world).
    Uses average log-difference over the last `window` periods.
    Returns np.nan if the window is invalid or any value ≤ 0 occurs.
    """
    if window < 2 or len(series) < window:
        return np.nan
    tail = series[-window:]
    if np.any(tail <= 0):
        return np.nan
    logs = np.log(tail)
    diffs = np.diff(logs)
    return float(np.mean(diffs))
