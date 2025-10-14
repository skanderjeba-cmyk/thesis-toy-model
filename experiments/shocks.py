# experiments/shocks.py
"""
Experiment engine: define shocks and standardized runs (updated model).

Implements baseline experiments:
  (a) n_D ↑  via a small fertility bump f_D
  (b) n_L ↑  via a small fertility bump f_L
  (c) abatement ↑ via (xi_i ↑ and/or theta_bar_i ↑)
  (d) ease of migration ↑ via (mu ↑ and/or tau_H ↓)

Also provides:
  - apply_shock(...): safe overlay of a params dict (supports *add*/*mult*)
  - run_experiment(...): simulate baseline vs shock, extract metrics, compare

Notes
-----
• The updated model has no toggles. Active labor is always L=ζN, and
  intensity depends on productivity q=Y/L.

• Migration uses a headcount wedge τ_H (persons/period) and a persons cap:
     m_t = min{ [ μ·(wD/wL)·N_L − τ_H ]_+ , m_bar·N_L , (1+f_L)·N_L }.

• Comparative metrics emphasize (Y/L) for D and L, world E, the ratio M/E,
  and the tail growth rate g_E. We report both:
    - theoretical limit for M_t/E_t  = 1 / ((1+g_E) - φ_M),
    - implied limit for the recorded ratio M_{t+1}/E_t = (1+g_E) / ((1+g_E) - φ_M).

Implementation note
-------------------
To keep runs admissible without editing your base YAML, `apply_shock(...)`
now clamps `migration.tau_H = max(0, tau_H)` after applying any overlay.
(If `tau_share` exists in your config, it is also clamped to ≥ 0.)
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import copy
import numpy as np
import pandas as pd  # kept for downstream callers; not required here

from model.params import load_params, ModelParams
from model.simulate import simulate, empirical_growth_rate
from model.solow import carbon_ratio_limit


# ----------------------------
# Small utilities
# ----------------------------

def _deep_update(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (overlay or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _sign_arrow(x: float, tol: float = 1e-9) -> str:
    if x > tol:
        return "↑"
    elif x < -tol:
        return "↓"
    else:
        return "∅"


def _pct_change(a: float, b: float) -> float:
    """
    %Δ = (b - a)/|a| * 100; uses |a| to avoid sign flips for negatives.
    If a == 0:
      - return 0.0 if b == 0 (clean 'no change' signal),
      - else return np.nan.
    """
    if a == 0:
        return 0.0 if b == 0 else np.nan
    return (b - a) / abs(a) * 100.0


# ----------------------------
# Shock overlays
# ----------------------------

def shock_nD_df(df: float = 0.005) -> dict:
    """
    Increase f_D (net natural increase) by df (e.g., 0.005 = +0.5 pp).
    (Additive overlay.)
    """
    return {"countries": {"D": {"f": df}}}


def shock_nL_df(df: float = 0.005) -> dict:
    """
    Increase f_L by df (e.g., 0.005 = +0.5 pp).
    (Additive overlay.)
    """
    return {"countries": {"L": {"f": df}}}


def shock_abatement(
    dxi_D: float = 0.10,
    dxi_L: float = 0.10,
    dtheta_D: float = 0.10,
    dtheta_L: float = 0.10,
) -> dict:
    """
    Raise abatement responsiveness xi and/or cap theta_bar (additive deltas).
    Feasibility s + kappa * theta_bar^2 ≤ 1 is validated by the loader.
    """
    return {
        "countries": {
            "D": {"xi": dxi_D, "theta_bar": dtheta_D},
            "L": {"xi": dxi_L, "theta_bar": dtheta_L},
        }
    }


def shock_ease_migration(mu_mult: float = 1.5, tau_H_delta: float = -0.05) -> dict:
    """
    Make migration easier:
      - multiply μ by mu_mult (>0),
      - reduce headcount wedge τ_H by tau_H_delta (<0 typical).
    """
    if not (mu_mult > 0):
        raise ValueError(f"mu_mult must be > 0; got {mu_mult}")
    return {"migration": {"mu": ("*mult*", mu_mult), "tau_H": ("*add*", tau_H_delta)}}


# ----------------------------
# Apply overlays safely
# ----------------------------

def apply_shock(base_cfg: dict, overlay: dict) -> dict:
    """
    Apply an overlay to a params dict. Supported special forms inside the overlay:
      - ("*mult*", x): multiply the baseline value by x
      - ("*add*",  x): add x to the baseline value

    Additive behavior for country keys: f, xi, theta_bar, zeta.

    Additionally (Option B): after applying migration overlays, we clamp
    tau_H >= 0 (and tau_share >= 0 if present) to keep the config admissible.
    """
    cfg = copy.deepcopy(base_cfg)

    def _apply_country(country_key: str, dct: dict):
        tgt = cfg["countries"][country_key]
        for k, v in dct.items():
            if k in ("f", "xi", "theta_bar", "zeta"):
                tgt[k] = float(tgt[k]) + float(v)
            else:
                tgt[k] = v

    for k, v in (overlay or {}).items():
        if k == "countries":
            for ckey, cdiff in v.items():
                _apply_country(ckey, cdiff)
        elif k == "migration":
            mig = cfg["migration"]
            for mk, mv in v.items():
                if isinstance(mv, tuple) and len(mv) == 2 and mv[0] in ("*mult*", "*add*"):
                    tag, val = mv
                    if tag == "*mult*":
                        mig[mk] = float(mig[mk]) * float(val)
                    else:
                        mig[mk] = float(mig[mk]) + float(val)
                else:
                    mig[mk] = mv
            # --- Clamp to admissible set (prevents tau_H < 0 after overlays) ---
            if "tau_H" in mig:
                mig["tau_H"] = max(0.0, float(mig["tau_H"]))
            if "tau_share" in mig:
                mig["tau_share"] = max(0.0, float(mig["tau_share"]))
        elif k in ("globals", "initial_state", "simulation"):
            cfg[k] = _deep_update(cfg[k], v) if k in cfg else copy.deepcopy(v)
        else:
            cfg[k] = v
    return cfg


# ----------------------------
# Metric extraction
# ----------------------------

def _last_flow(arr: np.ndarray) -> float:
    """Return the last valid (non-NaN) entry from a flow series with a terminal NaN."""
    if len(arr) == 0:
        return np.nan
    return float(arr[-2]) if np.isnan(arr[-1]) else float(arr[-1])


def extract_metrics(res) -> Dict[str, float]:
    """
    Extract standardized metrics from a SimResult:
      - end-of-horizon (Y/L) for D and L (productivity q)
      - world emissions last period
      - approx g_E (log-avg over last 20 or until series length)
      - last finite M/E (recorded as M_{t+1}/E_t)
    """
    qD_last = _last_flow(res.D.q)
    qL_last = _last_flow(res.L.q)

    E_last = float(res.E_world[-1])

    # Growth rate: average over last min(20, T-1) periods
    window = int(min(20, max(2, len(res.E_world) - 1)))
    gE = empirical_growth_rate(res.E_world, window=window)

    # Ratio: last finite entry in M_over_E (recorded as M_{t+1}/E_t)
    me = res.M_over_E[~np.isnan(res.M_over_E)]
    MoverE_last = float(me[-1]) if len(me) > 0 else np.nan

    return {
        "Y_over_L_D_last": qD_last,
        "Y_over_L_L_last": qL_last,
        "E_last": E_last,
        "gE": float(gE) if gE == gE else np.nan,
        "M_over_E_last": MoverE_last,
    }


def compare_metrics(baseline: Dict[str, float], shocked: Dict[str, float]) -> Dict[str, Any]:
    """
    Compare metric dicts: return %Δ and arrows for (Y/L)_D, (Y/L)_L, E_last, M/E.
    For gE we return the shocked value and an arrow relative to baseline.
    """
    out: Dict[str, Any] = {}
    for key in ("Y_over_L_D_last", "Y_over_L_L_last", "E_last", "M_over_E_last"):
        d = _pct_change(baseline[key], shocked[key])
        out[f"{key}_pct"] = d
        out[f"{key}_arrow"] = _sign_arrow(d if d == d else 0.0)

    dg = shocked["gE"] - baseline["gE"]
    out["gE_shocked"] = shocked["gE"]
    out["gE_arrow"] = _sign_arrow(dg if dg == dg else 0.0)
    return out


# ----------------------------
# Standardized experiment runner (in-memory)
# ----------------------------

def run_experiment(
    base_yaml_path: str,
    shock_name: str,  # one of {"nD_up","nL_up","abatement_up","ease_migration","ease_migration_strong"}
    shock_kwargs: Optional[dict] = None,
    T: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run one experiment: baseline vs shock (single model, no toggles).
    Returns a dict with baseline metrics, shocked metrics, comparisons, and references.
    """
    if shock_kwargs is None:
        shock_kwargs = {}

    # Load base config as dict
    import yaml
    with open(base_yaml_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}

    # Choose shock overlay
    if shock_name == "nD_up":
        overlay = shock_nD_df(**shock_kwargs)  # expects {"df": ...}
    elif shock_name == "nL_up":
        overlay = shock_nL_df(**shock_kwargs)
    elif shock_name == "abatement_up":
        overlay = shock_abatement(**shock_kwargs)
    elif shock_name == "ease_migration":
        overlay = shock_ease_migration(**shock_kwargs)
    elif shock_name == "ease_migration_strong":
        # Same mechanism as ease_migration, but caller passes stronger kwargs
        overlay = shock_ease_migration(**shock_kwargs)
    else:
        raise ValueError(
            "shock_name must be one of {'nD_up','nL_up','abatement_up','ease_migration','ease_migration_strong'}"
        )

    # Build shocked config (with admissibility clamp)
    cfg_shocked = apply_shock(base_cfg, overlay)

    # Dataclasses
    mp_base = load_params_from_dict(base_cfg)
    mp_shocked = load_params_from_dict(cfg_shocked)

    # Simulate
    res_base = simulate(mp_base, T=T)
    res_shock = simulate(mp_shocked, T=T)

    # Metrics
    mb = extract_metrics(res_base)
    ms = extract_metrics(res_shock)
    cmpd = compare_metrics(mb, ms)

    # Theoretical ratio limits (optional diagnostics):
    # - For M_t/E_t:          1/((1+g_E) - phi_M)
    # - For M_{t+1}/E_t:     (1.0 + g_E)/((1+g_E) - phi_M)
    theo_M_over_E = np.nan
    theo_recorded_Mp1_over_E = np.nan
    if ms["gE"] == ms["gE"] and (1.0 + ms["gE"]) > mp_shocked.globals.phi_M:
        theo_M_over_E = carbon_ratio_limit(ms["gE"], mp_shocked.globals.phi_M)
        theo_recorded_Mp1_over_E = (1.0 + ms["gE"]) * theo_M_over_E

    out: Dict[str, Any] = {
        "shock": shock_name,
        "params_used": {
            "T": int(T) if T is not None else int(mp_base.simulation.T),
            "shock_kwargs": shock_kwargs,
        },
        "baseline_metrics": mb,
        "shocked_metrics": ms,
        "comparison": cmpd,
        "theoretical_M_over_E_limit": float(theo_M_over_E) if theo_M_over_E == theo_M_over_E else np.nan,
        "theoretical_recorded_Mp1_over_E_limit": float(theo_recorded_Mp1_over_E) if theo_recorded_Mp1_over_E == theo_recorded_Mp1_over_E else np.nan,
        # Keep references for downstream plots/tables if needed:
        "res_base": res_base,
        "res_shock": res_shock,
        "mp_base": mp_base,
        "mp_shock": mp_shocked,
    }
    return out


# ----------------------------
# Helper: load params from dict (bypass file write)
# ----------------------------

def load_params_from_dict(cfg: dict) -> ModelParams:
    """
    Reconstruct ModelParams from an in-memory dict identical to the YAML structure.
    """
    import tempfile, yaml, os
    txt = yaml.safe_dump(cfg)
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        tmp.write(txt)
        tmp.flush()
        path = tmp.name
    try:
        mp = load_params(path)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
    return mp
