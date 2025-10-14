import numpy as np

from model.solow import (
    k_hat_star,
    carbon_ratio_limit,
    carbon_ratio_recorded_limit,  # new: explicit helper for recorded ratio
)

# ---------- Helpers for theory checks ----------

def elasticity_dln_e_dln_y(xi: float, y: float, eta: float) -> float:
    """
    Appendix C.2 (q-based intensity, holding q fixed):
    For interior abatement θ = ξ y, emissions per capita scale as
        e ∝ y * (1 - ξ y)   (since q is held fixed),
    so
        ∂ln e / ∂ln y |_q = 1 - (ξ y) / (1 - ξ y).

    Note: `eta` is unused here because q is held fixed; kept for signature symmetry.
    """
    assert 0.0 <= xi * y < 1.0, "Interior rule requires 0 ≤ ξ y < 1"
    return 1.0 - (xi * y) / (1.0 - xi * y)


def build_carbon_path_from_growth(gE: float, phi_M: float, T: int = 400, E0: float = 1.0):
    """
    Synthetic series with constant emissions growth gE to test M/E ratio limit.
    Timing: M_{t+1} = φ_M M_t + E_t.
    """
    E = np.empty(T); E[0] = E0
    for t in range(1, T):
        E[t] = (1.0 + gE) * E[t - 1]
    M = np.empty(T); M[0] = 0.0
    for t in range(1, T):
        M[t] = phi_M * M[t - 1] + E[t - 1]
    return M, E


# ---------- Tests ----------

def test_k_hat_star_monotone_in_n():
    # Parameters
    s = 0.25
    delta = 0.06
    gA = 0.0
    alpha = 0.33
    n_low = 0.00
    n_high = 0.02

    k_low = k_hat_star(s, delta, gA, n_low, alpha)
    k_high = k_hat_star(s, delta, gA, n_high, alpha)

    assert k_low > 0 and k_high > 0
    # Monotonicity: ∂k*/∂n < 0 → higher n gives lower k*
    assert k_high < k_low, "k* must decrease when n increases (capital dilution)."


def test_elasticity_threshold_interior_rule():
    # Hold q fixed; slope changes sign at ξ y = 1/2
    eta = 0.2  # annotated only; doesn't affect the threshold when q is fixed
    xi  = 0.4
    thr = 0.5  # threshold for (ξ y)

    # Choose y_low with ξ y below threshold, and y_high with ξ y above threshold (but < 1)
    xi_y_low  = thr * 0.8
    xi_y_high = min(thr * 1.2, 0.99)  # ensure interior: ξ y < 1

    y_low  = xi_y_low  / xi
    y_high = xi_y_high / xi

    # Sanity on interior condition
    assert 0.0 <= xi * y_low  < thr
    assert thr <  xi * y_high < 1.0

    e_slope_low  = elasticity_dln_e_dln_y(xi, y_low,  eta)
    e_slope_high = elasticity_dln_e_dln_y(xi, y_high, eta)

    assert e_slope_low  > 0.0, "Below threshold, e increases with y (slope > 0)."
    assert e_slope_high < 0.0, "Above threshold, e decreases with y (slope < 0)."


def test_carbon_ratio_limit_convergence():
    phi_M = 0.1
    gE = 0.015  # 1.5% growth
    assert (1.0 + gE) > phi_M

    M, E = build_carbon_path_from_growth(gE, phi_M, T=600, E0=2.0)
    ratio = M / E
    limit = carbon_ratio_limit(gE, phi_M)

    # Check last 50 periods are close to the theoretical limit
    tail = ratio[-50:]
    assert np.all(np.isfinite(tail))
    assert np.allclose(tail, limit, rtol=0.0, atol=1e-3), "M/E tail must be near 1/((1+gE)-phi_M)."


def test_recorded_ratio_limit_convergence():
    """
    Simulator records M_{t+1}/E_t; its limit is (1+gE)/((1+gE)-phi_M).
    """
    phi_M = 0.1
    gE = 0.015
    assert (1.0 + gE) > phi_M

    M, E = build_carbon_path_from_growth(gE, phi_M, T=600, E0=2.0)
    recorded_ratio = M[1:] / E[:-1]  # M_{t+1}/E_t

    expected = carbon_ratio_recorded_limit(gE, phi_M)  # use helper directly

    tail = recorded_ratio[-50:]
    assert np.all(np.isfinite(tail))
    assert np.allclose(tail, expected, rtol=0.0, atol=1e-3), "M_{t+1}/E_t tail must be near (1+gE)/((1+gE)-phi_M)."
