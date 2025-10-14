# scripts/sweep_ease_migration.py
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import os
import itertools
import numpy as np
import pandas as pd

from experiments.shocks import run_experiment

def tail_mean(x, k=40):
    x = np.asarray(x, float)
    k = min(k, x.size)
    if k <= 0: return np.nan
    tail = x[-k:]
    return float(np.nanmean(tail))

def wedge_bind_rate(res, mu, tau_H):
    # raw = mu * rho * N_L - tau_H <= 0  ⇒ wedge binds
    wD = np.asarray(res.D.w[:-1], float)
    wL = np.asarray(res.L.w[:-1], float)
    rho = wD / wL
    NL  = np.asarray(res.L.N[:-1], float)
    raw = mu * rho * NL - tau_H
    return float(np.mean(raw <= 0.0))

def cap_bind_rate(res, m_bar, f_L):
    # cap = min(m_bar*N_L, (1+f_L)N_L). Bind if m_t >= cap - tiny_tol
    NL = np.asarray(res.L.N[:-1], float)
    cap = np.minimum(m_bar * NL, (1.0 + f_L) * NL)
    m  = np.asarray(res.m, float)
    tol = 1e-12
    return float(np.mean(m >= (cap - tol)))

def main():
    os.makedirs("results", exist_ok=True)

    grid_mu  = [1.2, 1.5, 2.0, 2.5]
    grid_tau = [-0.05, -0.10, -0.20, -0.30]
    rows=[]

    for mu_mult, tau_H_delta in itertools.product(grid_mu, grid_tau):
        rec = run_experiment(
            base_yaml_path="params.yaml",
            shock_name="ease_migration",
            shock_kwargs={"mu_mult": mu_mult, "tau_H_delta": tau_H_delta},
            T=200
        )

        base  = rec["res_base"];  shock = rec["res_shock"]
        mp_b  = rec["mp_base"];   mp_s  = rec["mp_shock"]

        # tail means (flows index)
        d_tail_m  = tail_mean(shock.m, 40) - tail_mean(base.m, 40)
        d_YL_D    = float(shock.D.q[-2])   - float(base.D.q[-2])   # end-of-horizon Δ(Y/L)_D

        # binding stats
        wb_base = wedge_bind_rate(base,  mp_b.migration.mu,  mp_b.migration.tau_H)
        wb_shck = wedge_bind_rate(shock, mp_s.migration.mu,  mp_s.migration.tau_H)
        cb_base = cap_bind_rate(base,    mp_b.migration.m_bar, mp_b.L.f)
        cb_shck = cap_bind_rate(shock,   mp_s.migration.m_bar, mp_s.L.f)

        rows.append(dict(
            mu_mult=mu_mult, tau_H_delta=tau_H_delta,
            d_tail_m=d_tail_m,
            d_YL_D_last=d_YL_D,
            wedge_bind_base=wb_base, wedge_bind_shock=wb_shck,
            cap_bind_base=cb_base,   cap_bind_shock=cb_shck,
        ))

    df = pd.DataFrame(rows).sort_values(["mu_mult", "tau_H_delta"])
    out_csv = "results/ease_migration_sweep.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    # quick console view
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
