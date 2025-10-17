#!/usr/bin/env python
# Phase 1 audit — 10 checks (A1:2 + A2:1 + B:1 + C:3 + D:3)

import os
import sys
import numpy as np

from model.params import load_params
from model.solow import (
    k_hat_star,
    carbon_ratio_limit,
    carbon_ratio_recorded_limit,
)
from model.simulate import simulate
from model.checks import assert_simulation_ok
from experiments.runner import run_all

FIG_DIR = "figures"
RES_DIR = "results"

# Tail/window conventions (aligned with Companion and viz/tables.py)
TAIL_WINDOW_K = 20  # used for growth-rate and ratio tail estimates


# ---------------------------
# Small helpers
# ---------------------------

def rel_err(a, b, eps: float = 1e-10) -> float:
    """Relative error |a-b|/max(|b|,eps) with NaN-guard."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("inf")
    return abs(a - b) / max(abs(b), eps)


def tail_rate(series, tail: int = TAIL_WINDOW_K) -> float:
    """
    Approximate geometric growth over the tail:
      g ≈ exp(mean(Δ log x_tail)) - 1
    Clips series to avoid log(0).
    """
    x = np.asarray(series, float)
    x = np.clip(x, 1e-16, None)
    k = min(tail, x.size)  # robust for very short series
    if k < 2:
        return float("nan")
    dx = np.diff(np.log(x[-k:]))
    return float(np.exp(np.nanmean(dx)) - 1.0)


def tail_avg_recorded_ratio(M_over_E, tail: int = TAIL_WINDOW_K) -> float:
    """
    Tail average of the recorded ratio series M_{t+1}/E_t.
    Ignores NaNs and averages the last `tail` finite values.
    """
    r = np.asarray(M_over_E, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    return float(np.mean(r[-min(tail, r.size):]))


# ---------------------------
# A1: Solow fixed points
# ---------------------------

def a1_solow_fixed_points():
    """
    Compare simulated (Y/L)/A_i to the closed-form (k̂*)^α (country-specific deflators).
    We estimate n from the tail growth of active labor L (always L = ζ N).
    """
    mp = load_params("params.yaml")
    res = simulate(mp, T=140)

    alpha = mp.globals.alpha
    delta = mp.globals.delta
    gA = mp.globals.g_A

    # Tail growth of active labor (stocks, length T+1)
    nD_hat = tail_rate(res.D.L)
    nL_hat = tail_rate(res.L.L)

    # Theoretical k̂* with those n
    kD = k_hat_star(s=mp.D.s, delta=delta, g_A=gA, n=nD_hat, alpha=alpha)
    kL = k_hat_star(s=mp.L.s, delta=delta, g_A=gA, n=nL_hat, alpha=alpha)

    # Simulated (Y/L)/A_i series (flows t=0..T-1), deflate with country-specific A paths
    yL_D_flow = np.asarray(res.D.Y[:-1], float) / np.asarray(res.D.L[:-1], float)
    yL_L_flow = np.asarray(res.L.Y[:-1], float) / np.asarray(res.L.L[:-1], float)
    A_D_flow  = np.asarray(res.A_D[:-1], float)
    A_L_flow  = np.asarray(res.A_L[:-1], float)

    sim_D_deflated = float(np.nanmean((yL_D_flow / A_D_flow)[-10:]))
    sim_L_deflated = float(np.nanmean((yL_L_flow / A_L_flow)[-10:]))

    target_D = kD ** alpha
    target_L = kL ** alpha

    errD = rel_err(sim_D_deflated, target_D)
    errL = rel_err(sim_L_deflated, target_L)

    # 7% tolerance is generous but robust given finite-horizon & migration feedbacks
    okD = np.isfinite(errD) and errD <= 0.07
    okL = np.isfinite(errL) and errL <= 0.07
    return okD, errD, nD_hat, okL, errL, nL_hat


# ---------------------------
# A2: Ratio condition advisory (+ theory vs empirical ratio INFO)
# ---------------------------

def a2_ratio_condition():
    """
    Compute tail growth of world emissions and check (1+g_E) > φ_M,
    which guarantees a finite M_t/E_t limit.

    Also (informational only):
      - compute tail average of recorded ratio M_{t+1}/E_t
      - compute theoretical recorded ratio limit using carbon_ratio_recorded_limit(gE, phi_M)
      - return both for reporting; does not affect pass/fail counting.
    """
    mp = load_params("params.yaml")
    res = simulate(mp, T=220)

    E = np.asarray(res.E_world, float)
    E = np.clip(E, 1e-16, None)
    gE = tail_rate(E, tail=TAIL_WINDOW_K)

    ok = ((1.0 + gE) > mp.globals.phi_M)

    # Informational comparison of recorded ratio vs theory
    R_emp = tail_avg_recorded_ratio(res.M_over_E, tail=TAIL_WINDOW_K)
    R_theory = None
    theory_err = None
    theory_msg = ""
    if ok and np.isfinite(R_emp):
        try:
            R_theory = carbon_ratio_recorded_limit(gE, mp.globals.phi_M)
            # absolute percent error (relative to theory magnitude)
            if R_theory != 0.0:
                theory_err = 100.0 * abs(R_emp - R_theory) / abs(R_theory)
            else:
                theory_err = np.nan
        except ValueError as ve:
            theory_msg = f"(theory helper raised: {ve})"
    else:
        if not ok:
            theory_msg = "(no theoretical limit: (1+g_E) <= phi_M)"

    return {
        "ok": ok,
        "gE": gE,
        "phi": mp.globals.phi_M,
        "R_emp": R_emp,
        "R_theory": R_theory,
        "R_theory_abs_pct_err": theory_err,
        "note": theory_msg,
    }


# ---------------------------
# B: Invariants
# ---------------------------

def invariants_ok():
    mp = load_params("params.yaml")
    res = simulate(mp, T=120)
    assert_simulation_ok(res, mp)
    return True


# ---------------------------
# C: Signs (dilution via Solow, abatement via full run)
# ---------------------------

def c_signs_and_abatement():
    """
    C tests = 3 checks:
      - Capital-dilution signs (n_D↑ lowers (Y/L)_D; n_L↑ lowers (Y/L)_L) using
        CLOSED-FORM SOLOW steady states (pure theory).
      - Abatement-up sign (World E falls) using the FULL dynamic system via runner.
    """
    mp = load_params("params.yaml")

    # (i) Closed-form Solow dilution checks
    alpha, delta, gA = mp.globals.alpha, mp.globals.delta, mp.globals.g_A
    nD0, nL0 = mp.D.f, mp.L.f
    dn = 0.005  # +0.5 pp

    def yL_star(s, n):
        k = k_hat_star(s=s, delta=delta, g_A=gA, n=n, alpha=alpha)
        return k ** alpha  # per effective worker; A deflation handled implicitly

    yL_D_0 = yL_star(mp.D.s, nD0)
    yL_D_1 = yL_star(mp.D.s, nD0 + dn)
    yL_L_0 = yL_star(mp.L.s, nL0)
    yL_L_1 = yL_star(mp.L.s, nL0 + dn)

    solow_ok_nD = (yL_D_1 < yL_D_0)
    solow_ok_nL = (yL_L_1 < yL_L_0)

    # (ii) Runner-based abatement sign (full system)
    out = run_all(base_yaml_path="params.yaml", T=120)

    def Ew_end(simres) -> float:
        # E_world is already aggregated; use last period flow
        return float(np.asarray(simres.E_world, float)[-1])

    rec = out["raw"]["abatement_up"]
    abate_ok = (Ew_end(rec["res_shock"]) < Ew_end(rec["res_base"]))

    return {
        "nD": solow_ok_nD,
        "nL": solow_ok_nL,
        "abate": abate_ok,
    }


# ---------------------------
# D: Artifacts present
# ---------------------------

def artifacts_written():
    figs_ok = False
    nfigs = 0
    if os.path.isdir(FIG_DIR):
        pngs = [f for f in os.listdir(FIG_DIR) if f.lower().endswith(".png")]
        nfigs = len(pngs)
        # Expect ~7–12 in a full run; set a conservative floor
        figs_ok = nfigs >= 6
    cs_csv = os.path.join(RES_DIR, "cs_table.csv")
    run_sum = os.path.join(RES_DIR, "run_summary.txt")
    return figs_ok, os.path.isfile(cs_csv), os.path.isfile(run_sum), nfigs


# ---------------------------
# Main
# ---------------------------

def main():
    print("=== Phase 1 Audit (single-spec) ===")
    print(f"[INFO] Repo: {os.getcwd()}")
    passed = 0
    failed = 0

    # A1 (two checks)
    try:
        okD, eD, nDh, okL, eL, nLh = a1_solow_fixed_points()
        print(f"[{'PASS' if okD else 'FAIL'}] A1.D Solow fixed point (rel err {eD*100:.3f}%, n_hat={nDh:.4f})")
        print(f"[{'PASS' if okL else 'FAIL'}] A1.L Solow fixed point (rel err {eL*100:.3f}%, n_hat={nLh:.4f})")
        passed += int(okD) + int(okL)
        failed += int(not okD) + int(not okL)
    except Exception as e:
        print(f"[FAIL] A1 exception: {e}")
        failed += 2

    # A2 (one check) + informational theory vs empirical ratio
    try:
        A2 = a2_ratio_condition()
        ok = A2["ok"]
        gE = A2["gE"]
        phi = A2["phi"]
        print(f"[{'PASS' if ok else 'FAIL'}] A2 M/E condition: 1+g_E={1.0+gE:.3f} vs phi_M={phi:.3f}")
        passed += int(ok)
        failed += int(not ok)

        # Informational lines (do not affect pass/fail)
        R_emp = A2["R_emp"]
        R_theory = A2["R_theory"]
        note = A2["note"]
        if np.isfinite(R_emp):
            print(f"[INFO] A2 recorded ratio tail  ~ {R_emp:.4f}")
        else:
            print(f"[INFO] A2 recorded ratio tail  ~ n/a")

        if R_theory is not None and np.isfinite(R_theory):
            err = A2["R_theory_abs_pct_err"]
            err_str = f"{err:.2f}%" if (err is not None and np.isfinite(err)) else "n/a"
            print(f"[INFO] A2 theory recorded limit ~ {R_theory:.4f} (abs % err vs tail: {err_str})")
        elif note:
            print(f"[INFO] A2 theory recorded limit ~ n/a {note}")

    except Exception as e:
        print(f"[FAIL] A2 exception: {e}")
        failed += 1

    # B (one check)
    try:
        okB = invariants_ok()
        print(f"[{'PASS' if okB else 'FAIL'}] B Invariants (positivity & budget)")
        passed += int(okB)
        failed += int(not okB)
    except Exception as e:
        print(f"[FAIL] B exception: {e}")
        failed += 1

    # C (three checks: {nD_up, nL_up, abatement_up})
    try:
        R = c_signs_and_abatement()
        okD = R["nD"]
        okL = R["nL"]
        okAb = R["abate"]

        print(f"[{'PASS' if okD else 'FAIL'}] C nD_up: (Y/L)_D falls")
        print(f"[{'PASS' if okL else 'FAIL'}] C nL_up: (Y/L)_L falls")
        print(f"[{'PASS' if okAb else 'FAIL'}] C abatement_up: World E falls")

        flags = [okD, okL, okAb]
        passed += sum(int(bool(x)) for x in flags)
        failed += sum(1 - int(bool(x)) for x in flags)
    except Exception as e:
        print(f"[FAIL] C exception: {e}")
        failed += 3

    # D (three checks: artifacts)
    try:
        okF, okCS, okRun, nfig = artifacts_written()
        print(f"[{'PASS' if okF else 'FAIL'}] D1 Figures generated ({nfig} png files found)")
        print(f"[{'PASS' if okCS else 'FAIL'}] D2 CS table (results/cs_table.csv)")
        print(f"[{'PASS' if okRun else 'FAIL'}] D3 Run summary (results/run_summary.txt)")
        passed += int(okF) + int(okCS) + int(okRun)
        failed += int(not okF) + int(not okCS) + int(not okRun)
    except Exception as e:
        print(f"[FAIL] D exception: {e}")
        failed += 3

    print("\n=== Audit Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

