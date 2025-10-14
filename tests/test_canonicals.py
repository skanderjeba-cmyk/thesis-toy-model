import os, re, subprocess, sys, glob

def test_canonicals_created(tmp_path):
    # Small run to speed up
    cmd = [sys.executable, "run_scenarios.py", "--scenarios", "KL_gap_closed,open_migration_defaults,KL_gap_open_mu_only", "--T", "60"]
    subprocess.check_call(cmd)

    runs = sorted(glob.glob("runs/*_KL_gap_closed__open_migration_defaults__KL_gap_open_mu_only"))
    assert runs, "No run directory created"
    run = runs[-1]

    # Check summary has 'already named' for both
    summary = open(os.path.join(run, "results", "run_summary.txt"), "r", encoding="utf-8").read()
    assert "Canonical professor figures:" in summary
    assert re.search(r"fig_wage_ratio_.*: already named", summary)
    assert re.search(r"fig_migration_diagnostics_.*: already named", summary)
