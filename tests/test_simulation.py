import numpy as np

from model.params import load_params
from model.simulate import simulate
from model.checks import assert_simulation_ok
from model.solow import k_hat_star


def test_simulator_smoke_and_invariants():
    mp = load_params("params.yaml")
    res = simulate(mp, T=60)

    # Quick sanity — stocks positive (check both countries' A paths explicitly)
    assert res.A_D[-1] > 0.0 and res.A_L[-1] > 0.0
    assert res.D.K[-1] > 0.0 and res.L.K[-1] > 0.0
    assert res.D.N[-1] > 0.0 and res.L.N[-1] > 0.0

    # Full invariant check (raises if any identity/bound is violated)
    assert_simulation_ok(res, mp)


def test_solow_steady_state_level_path_consistency():
    """
    Balanced-growth/short-horizon ballpark check:
    With active labor always on (L = ζ N), per-capita income satisfies
        y ≈ ζ_t * A_{i,t} * (k_hat*)^α
    when migration effects on n are modest. We compare the last-flow simulated y
    to this target using country-specific A paths and flow-aligned ζ_t.
    """
    mp = load_params("params.yaml")
    T = 40
    res = simulate(mp, T=T)

    alpha = mp.globals.alpha
    gA = mp.globals.g_A
    delta = mp.globals.delta

    # Approximate long-run n using fertility only; migration makes exact n endogenous.
    nD = mp.D.f
    nL = mp.L.f

    kD_star = k_hat_star(mp.D.s, delta, gA, nD, alpha)
    kL_star = k_hat_star(mp.L.s, delta, gA, nL, alpha)

    # Flow-aligned (t = T-1) ζ_t; fall back to constant ζ if zeta_at is not provided
    t_flow = T - 1
    zetaD_t = mp.D.zeta_at(t_flow) if hasattr(mp.D, "zeta_at") else mp.D.zeta
    zetaL_t = mp.L.zeta_at(t_flow) if hasattr(mp.L, "zeta_at") else mp.L.zeta

    # Use country-specific A paths at the last valid flow index (t = T-1 → -2)
    A_D_last = float(res.A_D[-2])
    A_L_last = float(res.A_L[-2])

    yD_theory = zetaD_t * (kD_star ** alpha) * A_D_last
    yL_theory = zetaL_t * (kL_star ** alpha) * A_L_last

    # Last available simulated per-capita outputs (CountrySeries.y ends with NaN sentinel)
    yD_last = float(res.D.y[-2])
    yL_last = float(res.L.y[-2])

    # Not tight equality (migration and possibly time-varying ζ_t); require ballpark proximity
    for sim, theo in [(yD_last, yD_theory), (yL_last, yL_theory)]:
        assert np.isfinite(sim) and np.isfinite(theo)
        assert sim > 0 and theo > 0
        ratio = sim / theo
        assert 0.2 < ratio < 5.0, "Simulated y and theoretical target should be in the same ballpark."
