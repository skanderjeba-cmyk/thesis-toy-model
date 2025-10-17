# Companion — Fertility-gap package (A/B/C waves)

**Purpose.** Introduce a fertility gap between a symmetric pair, then study how
open-migration treatments (μ-only vs τᴴ-only) and an initial K/L gap shape dilution,
wage compression, and E vs M/E.

## Waves & scenario chains (slugs become filenames)
- **A1 (closed):** `symmetric_baseline_closed, fert_gap_pair_medium`
- **A2 (open μ-only):** `symmetric_baseline_closed, fert_gap_pair_medium, open_migration_defaults, open_mu_only`
- **A3 (open τᴴ-only):** `symmetric_baseline_closed, fert_gap_pair_medium, open_migration_defaults, open_tauH_only`
- **B1 (K/L gap, closed):** `KL_gap_closed, fert_gap_pair_medium`
- **C1 (K/L + μ-only):** `KL_gap_closed, open_migration_defaults, open_mu_only, fert_gap_pair_medium`
- **C2 (K/L + τᴴ-only):** `KL_gap_closed, open_migration_defaults, open_tauH_only, fert_gap_pair_medium`

## Reproduce (T=200) — PowerShell
python run_scenarios.py --scenarios symmetric_baseline_closed,fert_gap_pair_medium --T 200
python run_scenarios.py --scenarios symmetric_baseline_closed,fert_gap_pair_medium,open_migration_defaults,open_mu_only --T 200 --strict-one-knob
python run_scenarios.py --scenarios symmetric_baseline_closed,fert_gap_pair_medium,open_migration_defaults,open_tauH_only --T 200 --strict-one-knob
python run_scenarios.py --scenarios KL_gap_closed,fert_gap_pair_medium --T 200
python run_scenarios.py --scenarios KL_gap_closed,open_migration_defaults,open_mu_only,fert_gap_pair_medium --T 200 --strict-one-knob
python run_scenarios.py --scenarios KL_gap_closed,open_migration_defaults,open_tauH_only,fert_gap_pair_medium --T 200 --strict-one-knob

## What reviewers will see per run
- `figures/all/` (12 PNGs), `figures/canonical/` (4 slug-named PNGs)
- `results/cs_table.csv`, `run_summary.txt` (diffs + one-knob tags)
