import os, yaml
from pathlib import Path
from run_scenarios import deep_merge, load_yaml, build_overlay_diff_report, REPO_ROOT

def _merge_chain(names):
    base = load_yaml(REPO_ROOT / "params.yaml")
    cfg = dict(base)
    for n in names:
        cfg = deep_merge(cfg, load_yaml(REPO_ROOT / "scenarios" / f"{n}.yaml"))
    return cfg, base

def test_mu_only_changes_one_knob():
    names = ["KL_gap_closed", "open_migration_defaults", "KL_gap_open_mu_only"]
    base = load_yaml(REPO_ROOT / "params.yaml")
    lines, violations = build_overlay_diff_report(base, names)
    assert not violations, f"Expected no violations; got {violations}"

def test_tauH_only_changes_one_knob():
    names = ["KL_gap_closed", "open_migration_defaults", "KL_gap_open_tauH_only"]
    base = load_yaml(REPO_ROOT / "params.yaml")
    lines, violations = build_overlay_diff_report(base, names)
    assert not violations, f"Expected no violations; got {violations}"
