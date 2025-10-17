"""
Runner for the baseline experiments (single model; no toggles).

What this script does:
- Loads params.yaml
- Runs the shocks with standard magnitudes
- Collects standardized metrics (from experiments/shocks.py)
- Writes a compact CSV summary to results/experiments_summary.csv
- Returns full SimResult objects in-memory (via run_all()) for figures/tables

Note: We do not plot here. We only compute and save numeric summaries.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import os
import pandas as pd

from .shocks import run_experiment

# Default shock set (magnitudes can be overridden by caller)
DEFAULT_SHOCKS: List[Tuple[str, Dict[str, Any]]] = [
    # (shock_name, shock_kwargs)
    ("nD_up",                 {"df": 0.005}),  # +0.5 pp to f_D
    ("nL_up",                 {"df": 0.005}),  # +0.5 pp to f_L
    ("abatement_up",          {"dxi_D": 0.1, "dxi_L": 0.1, "dtheta_D": 0.1, "dtheta_L": 0.1}),
    ("ease_migration",        {"mu_mult": 1.5, "tau_H_delta": -0.05}),
    # Stronger easing while keeping tau_H ≥ 0 (admissible set)
    ("ease_migration_strong", {"mu_mult": 2.0, "tau_H_delta": -0.05}),
]


def _kwargs_repr(d: Dict[str, Any]) -> str:
    """
    Stable, compact, human-readable serialization for CSV.
    Prefers one-line YAML; falls back to str(d) if PyYAML is unavailable.
    """
    try:
        import yaml  # local import to keep runner lightweight
        return yaml.safe_dump(d, default_flow_style=True, sort_keys=True).strip()
    except Exception:
        return str(d)


def _flatten_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten the output of run_experiment into a single-row dict for CSV.
    Aligns with extract_metrics()/compare_metrics in experiments/shocks.py.
    Also tolerates old/new names for theoretical limits.
    """
    # Backward/forward-compatible grabs for theoretical limits
    theo_ME = rec.get("theoretical_ratio_limit")  # legacy name
    if theo_ME is None:
        theo_ME = rec.get("theoretical_M_over_E_limit")

    theo_recorded = rec.get("theoretical_recorded_Mp1_over_E_limit")

    row = {
        "shock": rec["shock"],
        "T": rec["params_used"]["T"],
        "shock_kwargs": _kwargs_repr(rec["params_used"]["shock_kwargs"]),
        # baseline metrics
        "base_Y_over_L_D": rec["baseline_metrics"]["Y_over_L_D_last"],
        "base_Y_over_L_L": rec["baseline_metrics"]["Y_over_L_L_last"],
        "base_E": rec["baseline_metrics"]["E_last"],
        "base_gE": rec["baseline_metrics"]["gE"],
        "base_M_over_E": rec["baseline_metrics"]["M_over_E_last"],
        # shocked metrics
        "shock_Y_over_L_D": rec["shocked_metrics"]["Y_over_L_D_last"],
        "shock_Y_over_L_L": rec["shocked_metrics"]["Y_over_L_L_last"],
        "shock_E": rec["shocked_metrics"]["E_last"],
        "shock_gE": rec["shocked_metrics"]["gE"],
        "shock_M_over_E": rec["shocked_metrics"]["M_over_E_last"],
        # comparisons (percent changes and arrows)
        "d_Y_over_L_D_pct": rec["comparison"]["Y_over_L_D_last_pct"],
        "d_Y_over_L_D_dir": rec["comparison"]["Y_over_L_D_last_arrow"],
        "d_Y_over_L_L_pct": rec["comparison"]["Y_over_L_L_last_pct"],
        "d_Y_over_L_L_dir": rec["comparison"]["Y_over_L_L_last_arrow"],
        "d_E_pct": rec["comparison"]["E_last_pct"],
        "d_E_dir": rec["comparison"]["E_last_arrow"],
        "gE_shocked": rec["comparison"]["gE_shocked"],
        "d_gE_dir": rec["comparison"]["gE_arrow"],
        "d_M_over_E_pct": rec["comparison"]["M_over_E_last_pct"],
        "d_M_over_E_dir": rec["comparison"]["M_over_E_last_arrow"],
        # theoretical ratios (robust to naming)
        "theory_M_over_E_limit": theo_ME,
        "theory_recorded_Mp1_over_E_limit": theo_recorded,
    }
    return row


def run_all(
    base_yaml_path: str = "params.yaml",
    T: int = 200,
    shocks: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """
    Run all experiments and return a dict with:
      - 'records': list of flattened rows (for CSV)
      - 'raw':     dict of raw outputs keyed by shock name (for plots/tables)
    """
    if shocks is None:
        shocks = DEFAULT_SHOCKS

    records: List[Dict[str, Any]] = []
    raw: Dict[str, Any] = {}

    for name, kwargs in shocks:
        rec = run_experiment(base_yaml_path, shock_name=name, shock_kwargs=kwargs, T=T)
        records.append(_flatten_record(rec))
        raw[name] = rec  # keep full object for downstream use

    return {"records": records, "raw": raw}


def write_summary_csv(records: list, out_path: str = "results/experiments_summary.csv") -> None:
    """
    Write the flattened records to a CSV file (creates folders as needed).
    """
    dirn = os.path.dirname(out_path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    df = pd.DataFrame.from_records(records)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    # Run with defaults and write CSV
    res = run_all(base_yaml_path="params.yaml", T=200)
    write_summary_csv(res["records"], out_path="results/experiments_summary.csv")
    print("Wrote results/experiments_summary.csv")

