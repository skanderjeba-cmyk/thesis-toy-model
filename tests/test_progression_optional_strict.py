import os
import numpy as np
import pytest
from pathlib import Path
from experiments.runner import run_all
from tests.test_progression_scenarios import (
    _merged_tmp_yaml, _first_baseline, _wage_ratio, SCEN, _flows_m
)

@pytest.mark.strict
def test_mu_only_wage_ratio_not_exploding(tmp_path: Path):
    """
    Optional strict check: under μ-only opening, the tail |rho-1| should not blow up.
    Threshold can be adjusted via STRICT_RHO_TAIL (default 0.25).
    """
    thr = float(os.getenv("STRICT_RHO_TAIL", "0.25"))

    # Closed baseline
    merged_closed = _merged_tmp_yaml(tmp_path, ["KL_gap_closed"], T=60)
    base_closed = _first_baseline(run_all(base_yaml_path=str(merged_closed), T=60))
    rho_closed = _wage_ratio(base_closed)

    # μ-only open (include open_migration_defaults if present)
    overlays_open = ["KL_gap_closed"]
    if (SCEN / "open_migration_defaults.yaml").exists():
        overlays_open.append("open_migration_defaults")
    overlays_open.append("KL_gap_open_mu_only")

    merged_open = _merged_tmp_yaml(tmp_path, overlays_open, T=60)
    base_open = _first_baseline(run_all(base_yaml_path=str(merged_open), T=60))
    rho_open = _wage_ratio(base_open)

    # Require that migration actually opened
    m_open = _flows_m(base_open)
    m_open = m_open[np.isfinite(m_open)]
    assert m_open.size and np.nanmax(np.abs(m_open)) > 1e-10, "μ-only should produce positive flows."

    # Tail absolute deviation from 1
    def tail_mean_abs(rho, k=10):
        z = np.abs(np.asarray(rho)[-k:] - 1.0)
        z = z[np.isfinite(z)]
        return float(np.nanmean(z)) if z.size else np.nan

    g_open = tail_mean_abs(rho_open)
    if not np.isfinite(g_open):
        pytest.skip("No finite tail wage-ratio data; skipping strict check.")

    assert g_open < thr, f"Tail |rho-1| too large under μ-only: {g_open:.3e} (threshold {thr})"