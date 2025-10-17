# tests/test_overlays_oneknob.py
import re
from pathlib import Path

import yaml
import pytest

# We import the helpers from your runner module
import run_scenarios as rs

REPO = Path(__file__).resolve().parents[1]
SCEN = REPO / "scenarios"

def load_yaml(p):
    return yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}

def test_fert_gap_pair_medium_changes_only_fertility():
    base = yaml.safe_load((REPO / "params.yaml").read_text(encoding="utf-8")) or {}
    overlays = ["symmetric_baseline_closed","fert_gap_pair_medium"]

    # Build the sequence of states exactly as your runner does
    states = [base]
    for name in overlays:
        states.append(
            rs.deep_merge(
                states[-1],
                yaml.safe_load((SCEN / f"{name}.yaml").read_text(encoding="utf-8")) or {}
            )
        )

    # We only care about the fert overlay's effect (state[1] -> state[2])
    changed2 = rs._changed_keys(states[1], states[2])

    # Allow that symmetric_baseline_closed may already set D.f=0.010.
    # So fert overlay may change L.f only, or both D.f and L.f — but nothing else.
    allowed = {"countries.D.f", "countries.L.f"}
    assert set(changed2).issubset(allowed), f"Unexpected keys changed: {set(changed2) - allowed}"
    assert len(changed2) >= 1, "Fertility overlay should change at least one country's f"

def test_open_mu_only_changes_only_migration_mu_and_gets_ok_tag():
    base = yaml.safe_load((REPO / "params.yaml").read_text(encoding="utf-8")) or {}
    overlays = ["symmetric_baseline_closed","fert_gap_pair_medium","open_migration_defaults","open_mu_only"]
    report_lines, violations = rs.build_overlay_diff_report(base, overlays)
    text = "\n".join(report_lines)
    # Look for the OK tag on the open_mu_only line
    assert "[OK one-knob: mu]" in text
    # and no violations were reported
    assert not violations

def test_open_tauH_only_changes_only_migration_tauH_and_gets_ok_tag():
    base = yaml.safe_load((REPO / "params.yaml").read_text(encoding="utf-8")) or {}
    overlays = ["symmetric_baseline_closed","fert_gap_pair_medium","open_migration_defaults","open_tauH_only"]
    report_lines, violations = rs.build_overlay_diff_report(base, overlays)
    text = "\n".join(report_lines)
    # Look for the OK tag on the open_tauH_only line
    assert "[OK one-knob: tau_H]" in text
    # and no violations were reported
    assert not violations
