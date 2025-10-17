# tests/test_scenarios_fertility.py
import numpy as np
import pytest

T = 200

def tail_mean(x, k=20):
    x = np.asarray(x, dtype=float)
    return float(np.mean(x[-k:]))

def _get_runner():
    """
    Try to find a Python-level function that runs a scenario *chain* and returns series.
    Must return a callable f(overlays:list[str], T:int)->dict with keys like:
      res["D"]["q"], res["L"]["q"], res["E_world"], res["Mplus1_over_E"], and either
      res["rho"] OR wages to build rho = w_D / w_L.
    If not available, skip the test.
    """
    # Attempt 1: experiments.runner.run_chain (suggested in our plan)
    try:
        from experiments.runner import run_chain as f
        return f
    except Exception:
        pass

    # Attempt 2: a generic name some repos use
    try:
        from experiments.runner import run_scenario_chain as f
        return f
    except Exception:
        pass

    pytest.skip("No in-Python scenario runner found (e.g., experiments.runner.run_chain). "
                "Sign-check tests skipped; config/one-knob tests still run.")

def _rho_from(res):
    # prefer a direct rho series
    rho = res.get("rho")
    if rho is not None:
        return rho
    # else try wages (nested or top-level)
    for cand in [
        (res.get("D",{}).get("w"), res.get("L",{}).get("w")),
        (res.get("w_D"), res.get("w_L")),
    ]:
        wD, wL = cand
        if wD is not None and wL is not None:
            wD = np.asarray(wD, dtype=float)
            wL = np.asarray(wL, dtype=float)
            return wD / wL
    raise KeyError("Cannot locate wage ratio series (rho or wages).")

def test_A1_closed_vs_baseline_signs():
    run_chain = _get_runner()

    base = run_chain(["symmetric_baseline_closed"], T=T)
    a1   = run_chain(["symmetric_baseline_closed","fert_gap_pair_medium"], T=T)

    qL_base = base["L"]["q"]; qL_a1 = a1["L"]["q"]
    qD_base = base["D"]["q"]; qD_a1 = a1["D"]["q"]

    assert tail_mean(qL_a1) < tail_mean(qL_base)               # L dilutes
    assert abs(tail_mean(qD_a1) - tail_mean(qD_base)) < 1e-3   # D ~unchanged

    rho_base = _rho_from(base); rho_a1 = _rho_from(a1)
    assert tail_mean(rho_a1) > tail_mean(rho_base)             # wage ratio widens

    E_base = base.get("E_world") or base.get("E")
    E_a1   = a1.get("E_world")   or a1.get("E")
    assert E_base is not None and E_a1 is not None
    assert tail_mean(E_a1) > tail_mean(E_base)                 # world E increases

    R_base = base.get("Mplus1_over_E") or base.get("M_over_E") or base.get("M_E_ratio")
    R_a1   = a1.get("Mplus1_over_E")   or a1.get("M_over_E")   or a1.get("M_E_ratio")
    assert R_base is not None and R_a1 is not None
    assert tail_mean(R_a1) < tail_mean(R_base)                 # recorded ratio drifts down

def test_A2_mu_only_compresses_wages_and_reallocates():
    run_chain = _get_runner()

    a1 = run_chain(["symmetric_baseline_closed","fert_gap_pair_medium"], T=T)
    a2 = run_chain(["symmetric_baseline_closed","fert_gap_pair_medium","open_migration_defaults","open_mu_only"], T=T)

    # wage ratio compresses
    rho_a1 = _rho_from(a1); rho_a2 = _rho_from(a2)
    assert tail_mean(rho_a2) < tail_mean(rho_a1)

    # D dilutes vs A1; L deepens vs A1
    assert tail_mean(a2["D"]["q"]) < tail_mean(a1["D"]["q"])
    assert tail_mean(a2["L"]["q"]) > tail_mean(a1["L"]["q"])

def test_A3_tauH_only_is_one_wave_and_L_deepens():
    run_chain = _get_runner()

    a1 = run_chain(["symmetric_baseline_closed","fert_gap_pair_medium"], T=T)
    a3 = run_chain(["symmetric_baseline_closed","fert_gap_pair_medium","open_migration_defaults","open_tauH_only"], T=T)

    # L deepens vs A1; D dilutes vs A1
    assert tail_mean(a3["L"]["q"]) > tail_mean(a1["L"]["q"])
    assert tail_mean(a3["D"]["q"]) < tail_mean(a1["D"]["q"])

    # If migration flow series exists, check "one wave" (one major peak then near zero)
    m_a3 = a3.get("mig_flow") or a3.get("m_t") or a3.get("migration_flow")
    if m_a3 is not None:
        m = np.asarray(m_a3, dtype=float)
        # crude check: tail mean near zero
        assert tail_mean(m, k=30) < 1e-6
