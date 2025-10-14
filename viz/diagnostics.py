# viz/diagnostics.py
import os
import numpy as np
import matplotlib.pyplot as plt

from viz.plots import _ensure_dir  # reuse existing helper

def migration_diagnostics_plot(rec, shock_name: str = "ease_migration", outdir: str = "figures") -> str:
    """
    Diagnostic figure for migration: plots m_t, wage ratio rho_t, N_L, L_D
    for baseline vs shock. Expects a 'rec' dict from experiments.runner.run_all()['raw'][shock].
    """
    _ensure_dir(outdir)

    base = rec["res_base"]
    shock = rec["res_shock"]

    t = base.tgrid[:-1]  # flows index
    # Wage ratio (safe: wages are positive by model invariants)
    rho_base  = np.asarray(base.D.w[:-1], float)  / np.asarray(base.L.w[:-1], float)
    rho_shock = np.asarray(shock.D.w[:-1], float) / np.asarray(shock.L.w[:-1], float)

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))

    # m_t
    ax[0, 0].plot(t, base.m, label="Baseline")
    ax[0, 0].plot(t, shock.m, linestyle="--", label="Shock")
    ax[0, 0].set_title("Migration flows $m_t$")
    ax[0, 0].set_xlabel("t"); ax[0, 0].legend()

    # rho_t
    ax[0, 1].plot(t, rho_base, label="Baseline")
    ax[0, 1].plot(t, rho_shock, linestyle="--", label="Shock")
    ax[0, 1].set_title("Wage ratio $\\rho_t = w_D/w_L$")
    ax[0, 1].set_xlabel("t"); ax[0, 1].legend()

    # N_L (stocks index)
    ax[1, 0].plot(base.tgrid, base.L.N, label="Baseline")
    ax[1, 0].plot(shock.tgrid, shock.L.N, linestyle="--", label="Shock")
    ax[1, 0].set_title("Population in L ($N_L$)")
    ax[1, 0].set_xlabel("t"); ax[1, 0].legend()

    # L_D (stocks index)
    ax[1, 1].plot(base.tgrid, base.D.L, label="Baseline")
    ax[1, 1].plot(shock.tgrid, shock.D.L, linestyle="--", label="Shock")
    ax[1, 1].set_title("Active labor in D ($L_D$)")
    ax[1, 1].set_xlabel("t"); ax[1, 1].legend()

    fig.suptitle(f"Migration diagnostics — {shock_name}")
    fig.tight_layout()
    fname = os.path.join(outdir, f"fig_migration_diagnostics_{shock_name}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname
