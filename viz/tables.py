"""
viz/tables.py  — Comparative-statistics summary table (updated model)

Single public function:

    comparative_statics_table(res_all) -> pandas.DataFrame

It expects the structure produced by experiments.runner.run_all():
res_all["raw"][shock_name] -> {
    "res_base": SimResult,
    "res_shock": SimResult
}

We report compact %-changes on end-of-horizon values:
- (Y/L)_D and (Y/L)_L : taken from q = Y/L at t = T-1
- World E              : use E_world at t = T-1
- Carbon               : if world emissions are (approximately) stationary in BOTH
                         baseline and shock (|g_E| small), compare M_T (levels).
                         Otherwise, compare tail average of the recorded ratio M_{t+1}/E_t.

Arrow encoding: "↑" (> +0.1%), "↓" (< -0.1%), "∅" (otherwise).

Module-level constants (kept in sync with the Companion):
    TAIL_WINDOW_K = 20
    STATIONARY_ABS_GROWTH_THRESHOLD = 1e-3
"""

from __future__ import annotations
from typing import Dict, Any, List
import os
import numpy as np
import pandas as pd

# --------------------------------------------------------------------
# Module constants (documented in the Companion; please keep in sync)
# --------------------------------------------------------------------
TAIL_WINDOW_K = 20
STATIONARY_ABS_GROWTH_THRESHOLD = 1e-3

# -------------------------
# Helpers
# -------------------------

def _pct_change(a: float, b: float) -> float:
    """
    Percent change with graceful zero-baseline handling and |a| denominator:
      %Δ = 100 * (b - a) / |a|
      If a == 0: return 0.0 if b == 0 else NaN.
    """
    a = float(a)
    b = float(b)
    if a == 0.0:
        return 0.0 if b == 0.0 else np.nan
    return 100.0 * (b - a) / abs(a)


def _arrow(x: float, tol: float = 0.1) -> str:
    if not np.isfinite(x):
        return "—"
    if x > tol:
        return "↑"
    if x < -tol:
        return "↓"
    return "∅"


def _end_q(res_country) -> float:
    """
    End-of-horizon output per active worker q = Y/L at t = T-1.
    CountrySeries.q typically ends with a NaN sentinel; fall back if not.
    """
    arr = np.asarray(res_country.q, dtype=float)
    return float(arr[-2] if np.isnan(arr[-1]) else arr[-1])


def _world_E_end(res) -> float:
    """World emissions at t = T-1 (flows index T-1)."""
    return float(res.E_world[-1])


def _approx_gE_tail(E_world: np.ndarray, tail: int = TAIL_WINDOW_K) -> float:
    """
    Approximate tail growth of world emissions using mean log-diff over the last `tail` points.
    Returns NaN if insufficient points or nonpositive values are present.
    """
    e_tail = np.asarray(E_world[-tail:], dtype=float)
    if e_tail.size < 2 or np.any(e_tail <= 0.0):
        return np.nan
    log_g = np.log(e_tail[1:] / e_tail[:-1])
    return float(np.mean(log_g))


def _ratio_tail_avg(M_over_E: np.ndarray, tail: int = TAIL_WINDOW_K) -> float:
    """
    Tail average of recorded ratio series M_{t+1}/E_t (NaNs ignored; average last `tail` finite values).
    """
    finite = np.asarray(M_over_E, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan
    tail_vals = finite[-min(tail, finite.size):]
    return float(np.mean(tail_vals))


# -------------------------
# Main public function
# -------------------------

def comparative_statics_table(res_all: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a compact comparative-statics table (arrows and small % numbers)
    for the baseline shocks. Returns a pandas DataFrame with rows = shocks and
    columns = indicators.
    """
    # Preferred order if available:
    preferred_order: List[str] = ["nD_up", "nL_up", "abatement_up", "ease_migration"]
    nice_names = {
        "nD_up": "n_D ↑",
        "nL_up": "n_L ↑",
        "abatement_up": "Abatement ↑",
        "ease_migration": "Ease of migration ↑",
    }

    store: Dict[str, Any] = res_all["raw"]
    # Only include shocks that are actually present (keeps function robust)
    shocks_order = [s for s in preferred_order if s in store] + [
        s for s in store.keys() if s not in preferred_order
    ]

    rows = []

    for shock in shocks_order:
        rec = store[shock]
        base = rec["res_base"]
        shck = rec["res_shock"]

        # (Y/L) = q paths (end-of-horizon)
        qD_base, qD_shock = _end_q(base.D), _end_q(shck.D)
        qL_base, qL_shock = _end_q(base.L), _end_q(shck.L)

        d_qD = _pct_change(qD_base, qD_shock)
        d_qL = _pct_change(qL_base, qL_shock)

        # World emissions: end-of-horizon comparison
        E_base = _world_E_end(base)
        E_shock = _world_E_end(shck)
        d_E = _pct_change(E_base, E_shock)

        # Carbon: choose metric based on (approx) stationarity in BOTH runs
        gE_tail_base  = _approx_gE_tail(base.E_world,  tail=TAIL_WINDOW_K)
        gE_tail_shock = _approx_gE_tail(shck.E_world, tail=TAIL_WINDOW_K)
        stationary_base  = (np.isfinite(gE_tail_base)  and abs(gE_tail_base)  < STATIONARY_ABS_GROWTH_THRESHOLD)
        stationary_shock = (np.isfinite(gE_tail_shock) and abs(gE_tail_shock) < STATIONARY_ABS_GROWTH_THRESHOLD)

        if stationary_base and stationary_shock:
            # Compare levels of M at T
            M_base = float(base.M[-1])
            M_shock = float(shck.M[-1])
            d_M = _pct_change(M_base, M_shock)
            M_arrow, M_pct = _arrow(d_M), d_M
            ME_arrow, ME_pct = "—", np.nan
        else:
            # Compare tail average of recorded ratio M_{t+1}/E_t
            r_base  = _ratio_tail_avg(base.M_over_E,  tail=TAIL_WINDOW_K)
            r_shock = _ratio_tail_avg(shck.M_over_E, tail=TAIL_WINDOW_K)
            d_ME = _pct_change(r_base, r_shock)
            ME_arrow, ME_pct = _arrow(d_ME), d_ME
            M_arrow, M_pct = "—", np.nan

        row = {
            "Shock": nice_names.get(shock, shock),
            "(Y/L)_D": _arrow(d_qD),
            "(Y/L)_D %Δ": d_qD,
            "(Y/L)_L": _arrow(d_qL),
            "(Y/L)_L %Δ": d_qL,
            "World E": _arrow(d_E),
            "World E %Δ": d_E,
            "M level (if stationary)": M_arrow,
            "M %Δ": M_pct,
            "M_{t+1}/E_t (else)": ME_arrow,
            "M_{t+1}/E_t %Δ": ME_pct,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # Column order
    df = df[
        [
            "Shock",
            "(Y/L)_D", "(Y/L)_D %Δ",
            "(Y/L)_L", "(Y/L)_L %Δ",
            "World E", "World E %Δ",
            "M level (if stationary)", "M %Δ",
            "M_{t+1}/E_t (else)", "M_{t+1}/E_t %Δ",
        ]
    ]
    return df


# --- Convenience helper to build and save CS table ---
def save_table(res_all, outdir: str = "results", filename: str = "cs_table.csv") -> pd.DataFrame:
    """
    Build the comparative-statics table and save it to results/cs_table.csv (by default).
    Returns the DataFrame.
    """
    os.makedirs(outdir, exist_ok=True)
    df = comparative_statics_table(res_all)
    df.to_csv(os.path.join(outdir, filename), index=False)
    return df
