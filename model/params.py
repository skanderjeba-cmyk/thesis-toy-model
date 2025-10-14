"""
Parameter objects, YAML loader, scenario overlay, and validations.

This module defines:
- CountryParams, GlobalParams, MigrationParams, InitialState, SimulationConfig
- ModelParams: top-level container with two CountryParams (D, L) + globals
- load_params(yaml_path, scenario=None): load from params.yaml, apply optional scenario overlay,
  build dataclasses, and run validations aligned with the paper.

Academic invariants enforced here (raise ValueError on violation):
- Domains: alpha∈(0,1), delta∈(0,1), phi_M∈(0,1), eta≥0, q_star>0, g_A>-1 (so 1+g_A>0),
  s∈(0,1), kappa>0, epsilon>0, xi≥0, theta_bar∈(0,1], f>-1,
  mu>0, tau_H≥0, m_bar∈(0,1].
- Budget cap at the *cap*: s_i + kappa_i * theta_bar_i^2 ≤ 1 for each country.
- Positivity for initial states: K_D,K_L,N_D,N_L,A0_D,A0_L > 0; M0 ≥ 0.
- Migration cap stated in persons (N_L,t); wages per active worker (used in rules.py).

NOTE: (i) Emissions intensity is always q-based: eps(q) = epsilon * (q / q_star)^eta with q=Y/L.
      (ii) Active labor is always on: L = zeta * N (optionally time-varying via zeta_schedule).
      (iii) The per-period abatement cap sqrt((1-s)/kappa) is applied in rules.py; here we ensure
            feasibility even when theta = theta_bar.
      (iv) YAML backward-compat: we accept either the spec flat initial_state keys
           {K_D,K_L,N_D,N_L,A0_D,A0_L,M0}, or the legacy flat {*0 keys}, or the legacy nested
           form {K:{D,L}, N:{D,L}, A0:{D,L} or scalar, M0}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List
import copy
import yaml


# ----------------------------
# Dataclasses
# ----------------------------

@dataclass(frozen=True)
class CountryParams:
    """
    Country-specific parameters. Active labor is L = zeta * N.
    If zeta_schedule is provided (list of length ≥ 1), it can be used by the simulator
    to vary zeta over time; otherwise the scalar zeta is used each period.
    """
    s: float               # savings rate in (0,1)
    f: float               # net natural increase > -1
    epsilon: float         # baseline emissions intensity > 0 (emissions/good)
    kappa: float           # abatement cost weight > 0
    xi: float              # abatement responsiveness ≥ 0 (theta = xi * y)
    theta_bar: float       # abatement cap in (0,1]
    zeta: float = 1.0      # active-labor share varsigma ∈ (0,1]; L = zeta * N
    zeta_schedule: Optional[List[float]] = None  # optional time path for zeta_t in (0,1]

    # Convenience: pick zeta for a given time index t (falls back to scalar)
    def zeta_at(self, t: int) -> float:
        if self.zeta_schedule is None:
            return self.zeta
        if t < 0:
            raise ValueError(f"t must be ≥ 0; got {t}")
        idx = min(t, len(self.zeta_schedule) - 1)
        return float(self.zeta_schedule[idx])


@dataclass(frozen=True)
class GlobalParams:
    alpha: float           # capital share ∈ (0,1)
    delta: float           # depreciation ∈ (0,1)
    phi_M: float           # carbon persistence ∈ (0,1)
    g_A: float             # common TFP growth rate with 1+g_A>0
    q_star: float          # scaling constant > 0 for q=Y/L in eps(q)
    eta: float             # intensity elasticity ≥ 0 in eps(q) = epsilon * (q/q_star)^eta


@dataclass(frozen=True)
class MigrationParams:
    mu: float              # responsiveness > 0 (1/period)
    tau_H: float           # headcount wedge ≥ 0 (persons/period)
    m_bar: float           # feasibility cap share ∈ (0,1] of N_L
    mig_slack: float = 0.0  # strict-slack calibration buffer (≥0); see cross-validation


@dataclass(frozen=True)
class InitialState:
    # Spec-conformant (flat) shape
    K_D: float
    K_L: float
    N_D: float
    N_L: float
    A0_D: float
    A0_L: float
    M0: float

    # --- Backward-compat alias properties (legacy *0 names) ---

    @property
    def K_D0(self) -> float:
        return self.K_D

    @property
    def K_L0(self) -> float:
        return self.K_L

    @property
    def N_D0(self) -> float:
        return self.N_D

    @property
    def N_L0(self) -> float:
        return self.N_L

    @property
    def A_D0(self) -> float:
        return self.A0_D

    @property
    def A_L0(self) -> float:
        return self.A0_L

    # --- Dictionary views expected by simulators/tests ---

    @property
    def K(self) -> Dict[str, float]:
        """Dictionary view: {'D': K_D, 'L': K_L}."""
        return {"D": float(self.K_D), "L": float(self.K_L)}

    @property
    def N(self) -> Dict[str, float]:
        """Dictionary view: {'D': N_D, 'L': N_L}."""
        return {"D": float(self.N_D), "L": float(self.N_L)}

    @property
    def A0_map(self) -> Dict[str, float]:
        """Dictionary view of initial A: {'D': A0_D, 'L': A0_L}."""
        return {"D": float(self.A0_D), "L": float(self.A0_L)}

    @property
    def A(self) -> Dict[str, float]:
        """Alias sometimes used for initial productivity by country."""
        return self.A0_map


@dataclass(frozen=True)
class SimulationConfig:
    T: int


@dataclass(frozen=True)
class ModelParams:
    globals: GlobalParams
    countries: Dict[str, CountryParams]  # keys: "D", "L"
    migration: MigrationParams
    initial_state: InitialState
    simulation: SimulationConfig

    # Convenience accessors
    @property
    def D(self) -> CountryParams:
        return self.countries["D"]

    @property
    def L(self) -> CountryParams:
        return self.countries["L"]


# ----------------------------
# YAML loading and overlay
# ----------------------------

def _deep_update(base: dict, overlay: dict) -> dict:
    """
    Recursively update 'base' with 'overlay' (both dicts).
    Returns a *new* dict (does not mutate inputs).
    """
    result = copy.deepcopy(base)
    for k, v in (overlay or {}).items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_update(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


# ----------------------------
# Validations
# ----------------------------

def _validate_country(name: str, p: CountryParams) -> None:
    if not (0.0 < p.s < 1.0):
        raise ValueError(f"{name}: savings s must be in (0,1); got {p.s}")
    if not (p.f > -1.0):
        raise ValueError(f"{name}: net natural increase f must be > -1; got {p.f}")
    if not (p.epsilon > 0.0):
        raise ValueError(f"{name}: epsilon must be > 0; got {p.epsilon}")
    if not (p.kappa > 0.0):
        raise ValueError(f"{name}: kappa must be > 0; got {p.kappa}")
    if not (p.xi >= 0.0):
        raise ValueError(f"{name}: xi must be ≥ 0; got {p.xi}")
    if not (0.0 < p.theta_bar <= 1.0):
        raise ValueError(f"{name}: theta_bar must be in (0,1]; got {p.theta_bar}")
    if not (0.0 < p.zeta <= 1.0):
        raise ValueError(f"{name}: zeta (active-labor share) must be in (0,1]; got {p.zeta}")
    # Optional schedule validations
    if p.zeta_schedule is not None:
        if not isinstance(p.zeta_schedule, list) or len(p.zeta_schedule) == 0:
            raise ValueError(f"{name}: zeta_schedule must be a non-empty list if provided.")
        for i, z in enumerate(p.zeta_schedule):
            zf = float(z)
            if not (0.0 < zf <= 1.0):
                raise ValueError(f"{name}: zeta_schedule[{i}] must be in (0,1]; got {z}")
    # Global budget feasibility at the cap: s + kappa * theta_bar^2 ≤ 1
    if p.s + p.kappa * (p.theta_bar ** 2) > 1.0 + 1e-12:
        raise ValueError(
            f"{name}: budget cap violated at theta=theta_bar: "
            f"s + kappa*theta_bar^2 = {p.s + p.kappa * (p.theta_bar ** 2):.4f} > 1"
        )


def _validate_globals(g: GlobalParams) -> None:
    if not (0.0 < g.alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1); got {g.alpha}")
    if not (0.0 < g.delta < 1.0):
        raise ValueError(f"delta must be in (0,1); got {g.delta}")
    if not (0.0 < g.phi_M < 1.0):
        raise ValueError(f"phi_M must be in (0,1); got {g.phi_M}")
    if not (g.q_star > 0.0):
        raise ValueError(f"q_star must be > 0; got {g.q_star}")
    if not (g.eta >= 0.0):
        raise ValueError(f"eta must be ≥ 0; got {g.eta}")
    if not (1.0 + g.g_A > 0.0):
        raise ValueError(f"g_A must satisfy 1+g_A > 0; got g_A={g.g_A}")


def _validate_migration(m: MigrationParams) -> None:
    if not (m.mu > 0.0):
        raise ValueError(f"mu must be > 0; got {m.mu}")
    if not (m.tau_H >= 0.0):
        raise ValueError(f"tau_H must be ≥ 0; got {m.tau_H}")
    if not (0.0 < m.m_bar <= 1.0):
        raise ValueError(f"m_bar must be in (0,1]; got {m.m_bar}")
    if not (m.mig_slack >= 0.0):
        raise ValueError(f"mig_slack must be ≥ 0; got {m.mig_slack}")


def _validate_initial_state(init: InitialState) -> None:
    checks = [
        ("K_D", init.K_D, True),
        ("K_L", init.K_L, True),
        ("N_D", init.N_D, True),
        ("N_L", init.N_L, True),
        ("A0_D", init.A0_D, True),
        ("A0_L", init.A0_L, True),
        ("M0",   init.M0,   False),
    ]
    for name, val, strictly_pos in checks:
        if strictly_pos:
            if not (val > 0.0):
                raise ValueError(f"Initial {name} must be > 0; got {val}")
        else:
            if not (val >= 0.0):
                raise ValueError(f"Initial {name} must be ≥ 0; got {val}")


def _validate_sim(sim: SimulationConfig) -> None:
    if not (isinstance(sim.T, int) and sim.T >= 1):
        raise ValueError(f"T must be an integer ≥ 1; got {sim.T}")


def _validate_structure(mp: ModelParams) -> None:
    if set(mp.countries.keys()) != {"D", "L"}:
        raise ValueError("countries must contain exactly two keys: 'D' and 'L'.")


def _validate_cross(mp: ModelParams) -> None:
    """
    Cross-field checks that need multiple blocks (e.g., migration slack using f_L).
    """
    # Strict-slack calibration for migration (Appendix): ensure the fertility branch is slack.
    f_L = mp.L.f
    ms = mp.migration.mig_slack if mp.migration.mig_slack is not None else 0.0
    # mig_slack ∈ [0, 1+f_L) is sensible
    if ms >= 1.0 + f_L:
        raise ValueError(
            f"mig_slack must be < 1+f_L (= {1.0+f_L:.4f}); got {ms}"
        )
    # Enforce m_bar ≤ min{1, 1+f_L - mig_slack}
    rhs = min(1.0, 1.0 + f_L - ms)
    if mp.migration.m_bar > rhs + 1e-12:
        raise ValueError(
            "m_bar violates strict-slack calibration: require m_bar ≤ "
            f"min(1, 1+f_L - mig_slack) = {rhs:.4f}; got m_bar={mp.migration.m_bar:.4f}"
        )


# ----------------------------
# Loader
# ----------------------------

def _parse_initial_state(init_cfg: dict) -> InitialState:
    """
    Accepts either the spec-conformant flat form:
        {K_D, K_L, N_D, N_L, A0_D, A0_L, M0}
    or the legacy flat form:
        {K_D0, K_L0, N_D0, N_L0, A_D0, A_L0, M0}
    or the legacy nested form:
        {K:{D,L}, N:{D,L}, A0:{D,L} *or scalar*, M0}
    """
    # Spec flat keys?
    spec_keys = {"K_D", "K_L", "N_D", "N_L", "A0_D", "A0_L", "M0"}
    if spec_keys.issubset(init_cfg.keys()):
        return InitialState(
            K_D=float(init_cfg["K_D"]),
            K_L=float(init_cfg["K_L"]),
            N_D=float(init_cfg["N_D"]),
            N_L=float(init_cfg["N_L"]),
            A0_D=float(init_cfg["A0_D"]),
            A0_L=float(init_cfg["A0_L"]),
            M0=float(init_cfg["M0"]),
        )

    # Legacy flat keys?
    legacy_flat = {"K_D0", "K_L0", "N_D0", "N_L0", "A_D0", "A_L0", "M0"}
    if legacy_flat.issubset(init_cfg.keys()):
        return InitialState(
            K_D=float(init_cfg["K_D0"]),
            K_L=float(init_cfg["K_L0"]),
            N_D=float(init_cfg["N_D0"]),
            N_L=float(init_cfg["N_L0"]),
            A0_D=float(init_cfg["A_D0"]),
            A0_L=float(init_cfg["A_L0"]),
            M0=float(init_cfg["M0"]),
        )

    # Legacy nested form
    try:
        K = init_cfg["K"]
        N = init_cfg["N"]
        A0 = init_cfg["A0"]
        M0 = init_cfg["M0"]
    except KeyError as e:
        missing = str(e).strip("'")
        raise KeyError(
            f"initial_state missing required key '{missing}'. Expected either "
            "flat keys {K_D,K_L,N_D,N_L,A0_D,A0_L,M0} or legacy flat {*0} keys or nested K/N/A0 blocks."
        )

    # A0 may be a scalar or a dict with D/L
    if isinstance(A0, dict):
        A0_D = float(A0.get("D"))
        A0_L = float(A0.get("L"))
    else:
        A0_D = A0_L = float(A0)

    return InitialState(
        K_D=float(K["D"]),
        K_L=float(K["L"]),
        N_D=float(N["D"]),
        N_L=float(N["L"]),
        A0_D=A0_D,
        A0_L=A0_L,
        M0=float(M0),
    )


def load_params(yaml_path: str, scenario: Optional[str] = None) -> ModelParams:
    """
    Load parameters from YAML, apply optional scenario overlay by key, and validate.

    Parameters
    ----------
    yaml_path : str
        Path to params.yaml.
    scenario : Optional[str]
        If provided and found under 'scenarios' in the YAML, apply that overlay.

    Returns
    -------
    ModelParams
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}

    # Apply scenario overlay if requested
    if scenario:
        scenarios = base_cfg.get("scenarios", {})
        if scenario not in scenarios:
            raise KeyError(f"Scenario '{scenario}' not found in params YAML.")
        cfg = _deep_update(base_cfg, scenarios[scenario])
    else:
        cfg = base_cfg

    # Build dataclasses
    g = cfg.get("globals", {})
    globals_dc = GlobalParams(
        alpha=float(g["alpha"]),
        delta=float(g["delta"]),
        phi_M=float(g["phi_M"]),
        g_A=float(g["g_A"]),
        q_star=float(g["q_star"]),
        eta=float(g["eta"]),
    )

    c = cfg.get("countries", {})
    if "D" not in c or "L" not in c:
        raise KeyError("YAML 'countries' must contain both 'D' and 'L' entries.")
    # Country D
    D_dc = CountryParams(
        s=float(c["D"]["s"]),
        f=float(c["D"]["f"]),
        epsilon=float(c["D"]["epsilon"]),
        kappa=float(c["D"]["kappa"]),
        xi=float(c["D"]["xi"]),
        theta_bar=float(c["D"]["theta_bar"]),
        zeta=float(c["D"].get("zeta", 1.0)),
        zeta_schedule=[float(x) for x in c["D"].get("zeta_schedule", [])] if "zeta_schedule" in c["D"] else None,
    )
    # Country L
    L_dc = CountryParams(
        s=float(c["L"]["s"]),
        f=float(c["L"]["f"]),
        epsilon=float(c["L"]["epsilon"]),
        kappa=float(c["L"]["kappa"]),
        xi=float(c["L"]["xi"]),
        theta_bar=float(c["L"]["theta_bar"]),
        zeta=float(c["L"].get("zeta", 1.0)),
        zeta_schedule=[float(x) for x in c["L"].get("zeta_schedule", [])] if "zeta_schedule" in c["L"] else None,
    )

    m = cfg.get("migration", {})
    migration_dc = MigrationParams(
        mu=float(m["mu"]),
        tau_H=float(m["tau_H"]),
        m_bar=float(m["m_bar"]),
        mig_slack=float(m.get("mig_slack", 0.0)),
    )

    init_cfg = cfg.get("initial_state", {})
    initial_state_dc = _parse_initial_state(init_cfg)

    sim = cfg.get("simulation", {})
    simulation_dc = SimulationConfig(T=int(sim["T"]))

    mp = ModelParams(
        globals=globals_dc,
        countries={"D": D_dc, "L": L_dc},
        migration=migration_dc,
        initial_state=initial_state_dc,
        simulation=simulation_dc,
    )

    # Validations
    _validate_structure(mp)
    _validate_globals(mp.globals)
    _validate_country("D", mp.D)
    _validate_country("L", mp.L)
    _validate_migration(mp.migration)
    _validate_initial_state(mp.initial_state)
    _validate_sim(mp.simulation)
    _validate_cross(mp)

    return mp


# ----------------------------
# Convenience summary (for quick sanity in notebooks)
# ----------------------------

def summarize(mp: ModelParams) -> str:
    """Return a compact, human-readable summary (no side effects)."""
    lines = []
    lines.append("=== Globals ===")
    lines.append(
        f"alpha={mp.globals.alpha}, delta={mp.globals.delta}, phi_M={mp.globals.phi_M}, "
        f"g_A={mp.globals.g_A}, q_star={mp.globals.q_star}, eta={mp.globals.eta}"
    )
    for name in ("D", "L"):
        cp = mp.countries[name]
        zs = f", zeta_schedule_len={len(cp.zeta_schedule)}" if cp.zeta_schedule is not None else ""
        lines.append(
            f"== {name} == s={cp.s}, f={cp.f}, eps={cp.epsilon}, kappa={cp.kappa}, "
            f"xi={cp.xi}, theta_bar={cp.theta_bar}, zeta={cp.zeta}{zs}"
        )
    lines.append(
        f"== Migration == mu={mp.migration.mu}, tau_H={mp.migration.tau_H}, "
        f"m_bar={mp.migration.m_bar}, mig_slack={mp.migration.mig_slack}"
    )
    lines.append(
        "== Initial == "
        f"K_D={mp.initial_state.K_D}, K_L={mp.initial_state.K_L}, "
        f"N_D={mp.initial_state.N_D}, N_L={mp.initial_state.N_L}, "
        f"A0_D={mp.initial_state.A0_D}, A0_L={mp.initial_state.A0_L}, "
        f"M0={mp.initial_state.M0}"
    )
    lines.append(f"== Simulation == T={mp.simulation.T}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Manual smoke test (optional): `python -m model.params` from project root
    mp = load_params("params.yaml")
    print(summarize(mp))
