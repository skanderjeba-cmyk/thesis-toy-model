#!/usr/bin/env python
"""
make_all_figures.py

One-command reproducibility harness for the Migration–Abatement–Carbon Stock toy model (Phase 1).

This script:
  1) Loads parameters from params.yaml
  2) Runs the baseline experiments (single specification; no toggles)
  3) Generates all figures requested in Phase 1
  4) Builds the comparative-statics table and saves it to CSV
  5) Runs integrity checks on a baseline simulation
  6) Writes a short run log into results/run_summary.txt

Run from repo root:
    python make_all_figures.py

Outputs:
  - figures/*.png
  - results/cs_table.csv
  - results/run_summary.txt
"""

import os
import sys
import traceback
from datetime import datetime

# Use a non-interactive backend so this works headless, too
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from model.params import load_params
from model.checks import assert_simulation_ok
from experiments.runner import run_all
from viz.plots import capital_dilution_plot, emissions_and_ratio_plot, ambiguity_map
from viz.tables import save_table

# Try to import the dedicated diagnostics helper; fall back to a local implementation if missing
try:
    from viz.diagnostics import migration_diagnostics_plot  # optional convenience module
except Exception:
    migration_diagnostics_plot = None


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _migration_diagnostics_fallback(rec, shock_name: str, outdir: str) -> None:
    """
    Fallback diagnostics plot if viz.diagnostics.migration_diagnostics_plot
    is not available. Produces a 2x2 panel with:
      - m_t (flows), wage ratio rho_t, N_L (stock), L_D (stock).
    """
    import matplotlib.pyplot as plt

    base = rec["res_base"]
    shock = rec["res_shock"]

    t_flow = base.tgrid[:-1]
    # Guard against division by zero (shouldn't happen with invariants, but be safe)
    rho_base = base.D.w[:-1] / np.maximum(base.L.w[:-1], 1e-12)
    rho_shock = shock.D.w[:-1] / np.maximum(shock.L.w[:-1], 1e-12)

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))

    # (1) Migration flows
    ax[0, 0].plot(t_flow, base.m, label="baseline")
    ax[0, 0].plot(t_flow, shock.m, "--", label="shock")
    ax[0, 0].set_title("Migration flows $m_t$")
    ax[0, 0].set_xlabel("t")
    ax[0, 0].legend()

    # (2) Wage ratio
    ax[0, 1].plot(t_flow, rho_base, label=r"baseline $\rho_t$")
    ax[0, 1].plot(t_flow, rho_shock, "--", label=r"shock $\rho_t$")
    ax[0, 1].set_title(r"Wage ratio $\rho_t = w_D / w_L$")
    ax[0, 1].set_xlabel("t")
    ax[0, 1].legend()

    # (3) Origin population N_L (stocks use tgrid length T+1)
    ax[1, 0].plot(base.tgrid, base.L.N, label=r"baseline $N_L$")
    ax[1, 0].plot(shock.tgrid, shock.L.N, "--", label=r"shock $N_L$")
    ax[1, 0].set_title("Population in L")
    ax[1, 0].set_xlabel("t")
    ax[1, 0].legend()

    # (4) Destination active labor L_D (stocks)
    ax[1, 1].plot(base.tgrid, base.D.L, label=r"baseline $L_D$")
    ax[1, 1].plot(shock.tgrid, shock.D.L, "--", label=r"shock $L_D$")
    ax[1, 1].set_title("Active labor in D")
    ax[1, 1].set_xlabel("t")
    ax[1, 1].legend()

    fig.suptitle(f"Migration diagnostics — {shock_name}")
    fig.tight_layout()
    _ensure_dir(outdir)
    fname = os.path.join(outdir, f"fig_migration_diagnostics_{shock_name}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def main() -> int:
    repo_root = os.path.abspath(os.path.dirname(__file__))
    os.chdir(repo_root)

    figures_dir = os.path.join(repo_root, "figures")
    results_dir = os.path.join(repo_root, "results")
    _ensure_dir(figures_dir)
    _ensure_dir(results_dir)

    log_lines = []

    def log(msg: str):
        print(msg, flush=True)
        log_lines.append(msg)

    log("=== Reproducibility run started ===")
    log(f"Timestamp (local): {datetime.now().isoformat(timespec='seconds')}")
    log(f"Working directory: {os.getcwd()}")

    # 1) Load params
    try:
        mp = load_params("params.yaml")
        log("Loaded params.yaml successfully.")
    except Exception as e:
        log("ERROR: Failed to load params.yaml")
        log(str(e))
        return 1

    # 2) Run all experiments (single model; no toggles)
    try:
        T = int(mp.simulation.T)
        res_all = run_all(base_yaml_path="params.yaml", T=T)
        log(f"Experiments completed (T={T}).")
    except Exception:
        log("ERROR: Experiments failed in run_all().")
        log(traceback.format_exc())
        return 1

    # 3) Generate figures
    try:
        # Include the new strong-easing case
        shock_order = ("nD_up", "nL_up", "abatement_up", "ease_migration", "ease_migration_strong")
        generated = 0

        for shock_name in shock_order:
            if shock_name not in res_all["raw"]:
                continue
            rec = res_all["raw"][shock_name]

            # Capital dilution panels for demographic shocks and migration easing (both variants)
            if shock_name in ("nD_up", "nL_up", "ease_migration", "ease_migration_strong"):
                capital_dilution_plot(
                    rec["res_base"], rec["res_shock"], shock_name, outdir=figures_dir
                )
                generated += 1

            # Emissions + ratio for all shocks (use true phi_M)
            emissions_and_ratio_plot(
                rec["res_base"], rec["res_shock"], shock_name,
                phi_M=float(mp.globals.phi_M),
                outdir=figures_dir,
            )
            generated += 1

            # Migration diagnostics only for migration shocks
            if shock_name in ("ease_migration", "ease_migration_strong"):
                if migration_diagnostics_plot is not None:
                    migration_diagnostics_plot(rec, shock_name=shock_name, outdir=figures_dir)
                else:
                    _migration_diagnostics_fallback(rec, shock_name=shock_name, outdir=figures_dir)
                generated += 1

        # Ambiguity map (η annotated; interior boundary at ξ·y = 1/2)
        ambiguity_map(
            xi_vals=np.linspace(0.05, 1.0, 60),
            y_vals=np.linspace(0.1, 3.0, 60),
            eta=float(mp.globals.eta),
            outdir=figures_dir,
        )
        generated += 1

        log(f"Figures generated into figures/*.png (count ≈ {generated}).")
    except Exception:
        log("ERROR: Failed while generating figures.")
        log(traceback.format_exc())
        return 1

    # 4) Comparative statics table to CSV (single spec)
    try:
        df = save_table(res_all, outdir=results_dir, filename="cs_table.csv")
        log(f"Comparative-statics table saved to {os.path.join(results_dir, 'cs_table.csv')}")
    except Exception:
        log("ERROR: Failed while writing CS table.")
        log(traceback.format_exc())
        return 1

    # 5) Integrity check on a baseline simulation (raises on failure)
    try:
        base_any = res_all["raw"]["nD_up"]["res_base"]  # any baseline works
        assert_simulation_ok(base_any, mp)
        log("Sanity checks passed on baseline simulation.")
    except Exception:
        log("ERROR: Invariant checks failed. See traceback below.")
        log(traceback.format_exc())
        return 1

    # 6) Write a short run summary
    try:
        summary_path = os.path.join(results_dir, "run_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("Reproducibility run summary\n")
            f.write("===========================\n")
            f.write(f"Time: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"Repo: {repo_root}\n\n")
            f.write("Experiments: completed (single specification; no toggles)\n")
            f.write("Figures: figures/*.png\n")
            f.write("Tables: results/cs_table.csv\n")
            f.write("Checks: baseline invariants passed\n")
        log(f"Wrote summary: {summary_path}")
    except Exception as e:
        log("WARNING: Could not write run summary.")
        log(str(e))

    log("=== Reproducibility run finished successfully ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
