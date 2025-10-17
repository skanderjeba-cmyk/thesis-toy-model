# Fertility-gap overlays — what each file changes

This folder adds four focused overlays used in the A/B/C waves:
- `fert_gap_pair_medium.yaml`  → sets `countries.D.f: 0.010`, `countries.L.f: 0.020`
- `fert_gap_pair_small.yaml`   → sets `countries.D.f: 0.0125`, `countries.L.f: 0.0175`
- `open_mu_only.yaml`          → sets `migration.mu` **only** (one-knob)
- `open_tauH_only.yaml`        → sets `migration.tau_H` **only** (one-knob)

Notes:
- Each overlay changes exactly **one idea**; chains combine ideas.
- One-knob enforcement appears in each run’s `run_summary.txt`.
- Canonical figures include the **scenario slug** in the filename
  (e.g., `fig_wage_ratio_<slug>.png`) so they drop straight into slides.
- See `docs/COMPANION_fertility_gap.md` for the A/B/C chains and how to reproduce.
