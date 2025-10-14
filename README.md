# Migration, Abatement & Carbon Stock — Toy Model (Phase 1)

Two-country (D, L) **rule-based** growth–emissions model. Active labor is always used (`L = ζ·N`). Emissions intensity is **q-based** (`q = Y/L`):
[
\varepsilon_{i,t}=\varepsilon_i\left(\frac{q_{i,t}}{q_\star}\right)^{\eta},\qquad q_{i,t}=\frac{Y_{i,t}}{L_{i,t}}.
]
Deterministic, reproducible, and unit-tested.

---

## Model at a glance

* **Production:** (Y_{i,t}=K_{i,t}^{\alpha},(A_{i,t}L_{i,t})^{1-\alpha}) with country-specific productivity levels evolving via (A_{i,t+1}=(1+g_A)A_{i,t}) (common growth rate (g_A)).
* **Prices:** (w_{i,t}=(1-\alpha),Y_{i,t}/L_{i,t}), (r_{i,t}=\alpha,Y_{i,t}/K_{i,t}).
* **Active labor (always on):** (L_{i,t}=\zeta_i N_{i,t}), (0<\zeta_i\le1).
* **Capital:** (K_{i,t+1}=(1-\delta)K_{i,t}+s_iY_{i,t}).
* **Abatement rule:** (\theta_{i,t}=\min{\bar\theta_i,;\xi_i y_{i,t},;\sqrt{(1-s_i)/\kappa_i}}\in[0,1]) with (y_{i,t}=Y_{i,t}/N_{i,t}).
* **Intensity (q-based):** (\varepsilon_{i,t}=\varepsilon_i,(q_{i,t}/q_\star)^{\eta}).
* **Emissions:** (E_{i,t}=\varepsilon_{i,t}(1-\theta_{i,t})Y_{i,t}).
* **Migration (one-way (L\to D), persons-scaled):**
  [
  m_t=\min!\left{,[\mu,(w_{D,t}/w_{L,t}),N_{L,t}-\tau_H]*+,\ \bar m,N*{L,t},\ (1+f_L)N_{L,t}\right}.
  ]
* **Population:** (N_{D,t+1}=(1+f_D)N_{D,t}+m_t,\quad N_{L,t+1}=(1+f_L)N_{L,t}-m_t).
* **Carbon stock:** (M_{t+1}=\phi_M M_t+\sum_i E_{i,t}).

**Key analytic results**

* Solow steady state in efficiency units:
  [
  \hat k^*=\Big(\tfrac{s}{\delta+g_A+n+g_A n}\Big)^{!\frac{1}{1-\alpha}},\qquad
  \Big(\tfrac{Y}{L}\Big)*{i}^{!*}=A*{i,t},(\hat k^*)^{\alpha},\qquad
  y_i^*=\zeta_i,\Big(\tfrac{Y}{L}\Big)_{i}^{!*}.
  ]
* If aggregate emissions grow at (g_E>0):
  [
  \frac{M_t}{E_t}\to \frac{1}{(1+g_E)-\phi_M}.
  ]
  *(Figures plot the **recorded** ratio (M_{t+1}/E_t), whose limit is (\frac{1+g_E}{(1+g_E)-\phi_M}) when defined.)*
  If (g_E=0): (M^* = E /(1-\phi_M)).

---

## Quick start (one-command repro)

```bash
# optional: create & activate a virtual env
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate

# install minimal deps
python -m pip install -r requirements.txt         # or: pip install numpy pandas matplotlib pyyaml pytest

# regenerate ALL Phase-1 artifacts from params.yaml
python make_all_figures.py

# optional: unit tests + audit
pytest -q tests
python audit_phase1.py
```

**Outputs written by `make_all_figures.py`**

* Figures: `figures/*.png`

  * `fig_capital_dilution_{shock}.png`
  * `fig_emissions_ratio_{shock}.png`  *(ratio panel shows (M_{t+1}/E_t))*
  * `fig_ambiguity_eta-{η}.png`
* Table: `results/cs_table.csv`
* Run log: `results/run_summary.txt`

*(Optional numeric summary: running `python -m experiments.runner` writes `results/experiments_summary.csv`.)*

---

## Repository map (short)

```
model/
  params.py     # dataclasses, YAML loader (validations; q_star, eta)
  rules.py      # production, prices, theta(y), epsilon(q), migration, updates
  simulate.py   # deterministic time-path simulator (returns SimResult)
  solow.py      # steady states & M/E ratio limit; q-based intensity
  checks.py     # invariant & identity checks for a SimResult
experiments/
  shocks.py     # shock overlays + helpers (load-from-dict)
  runner.py     # run_all(...) → {'records', 'raw'}; optional CSV writer
viz/
  plots.py      # capital dilution; emissions & M/E; ambiguity map
  tables.py     # comparative-statics CSV (single spec → cs_table.csv)
tests/
  test_theory.py     # Solow/elasticity/M/E limit
  test_simulation.py # smoke + invariants + Solow consistency
figures/       # created; all .png figures
results/       # created; CSV(s) + run_summary.txt
params.yaml    # configuration (globals, countries D/L, migration, scenarios)
make_all_figures.py
audit_phase1.py
README.md
```

---

## Parameters (`params.yaml`)

* **Globals:** `alpha, delta, phi_M, g_A, q_star, eta`
* **Countries (D, L):** `s, f, epsilon, kappa, xi, theta_bar, zeta`
* **Migration:** `mu, tau_H, m_bar, epsilon_slack`
* **Initial state:** `K, N, A0, M0`
* **Simulation:** `T`
* **Scenarios (overlays):** `high_abatement`, `easy_migration`, `aging_in_D`

No feature toggles. Active labor and q-based intensity are always on.

---

## Experiments (what gets computed & plotted)

We run four shocks (single canonical specification):

1. `nD_up` — raise (f_D) → capital dilution lowers ((Y/L)_D).
2. `nL_up` — raise (f_L) → capital dilution lowers ((Y/L)_L).
3. `abatement_up` — raise (\xi_i) and/or (\bar\theta_i) → world (E) falls.
4. `ease_migration` — higher (\mu) and/or lower (\tau_H) → reallocation across D/L; world (E) ambiguous.

**Figures**

* Capital dilution panels for `nD_up` and `nL_up`: baseline (solid) vs shock (dashed) paths of ((Y/L)_D) and ((Y/L)_L).
* Emissions & ratio panels (all four shocks): world (E_t) and **recorded** ratio (M_{t+1}/E_t) with optional asymptote lines when identifiable.
* Ambiguity map: sign boundary for the interior slope of per-capita emissions at (\xi y = \tfrac12) (η only annotated).

**Table**

* `results/cs_table.csv` — arrows and %-changes for ((Y/L)_D), ((Y/L)_L), world (E), and (M) **or** (M/E) (uses (M) if both baseline and shocked (g_E\approx 0); else compares tail averages of (M/E)).

---

## Invariants & checks (enforced)

* **Positivity:** (K,Y,L,N,w) well-defined; no negative stocks/flows.
* **Budget feasibility:** (s_i+\kappa_i\theta_{i,t}^2\le 1) (guaranteed by the rule’s cap).
* **Abatement bounds:** (0\le \theta_{i,t}\le \min{\bar\theta_i,\sqrt{(1-s_i)/\kappa_i}}).
* **Migration cap:** (0\le m_t\le \min{\bar m,N_{L,t},,(1+f_L),N_{L,t}}).
* **Accounting identities:** production, prices, emissions, population, capital, and carbon recursions match to tolerance.
* **Ratio advisory:** if empirical (g_E>0) but ((1+g_E)\le \phi_M), warn (no ratio limit).

Run `python audit_phase1.py` after a reproducibility run for a compact PASS/INFO report.

---

## Environment

* Python ≥ 3.10
* `numpy`, `pandas`, `matplotlib`, `pyyaml`, `pytest`
* Headless plotting via `MPLBACKEND=Agg` (set automatically in scripts)

---

## Naming discipline & provenance

* Figures: `fig_<category>_<shock>.png`
  (e.g., `fig_capital_dilution_nD_up.png`, `fig_emissions_ratio_ease_migration.png`, `fig_ambiguity_eta-0.3.png`).
* Table: `results/cs_table.csv`.
* Reproducibility harness writes `results/run_summary.txt` with timestamps.
* Recommended: commit before/after a run to pin code state.
