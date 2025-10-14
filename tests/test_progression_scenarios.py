# tests/test_progression_scenarios.py
from __future__ import annotations
import numpy as np
import yaml
import pytest
from pathlib import Path
from typing import Any, Dict, List

from experiments.runner import run_all

REPO_ROOT = Path(__file__).resolve().parents[1]
SCEN = REPO_ROOT / "scenarios"
BASE_YAML = REPO_ROOT / "params.yaml"


# ---------------- helpers ----------------

def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def _load_yaml(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _write_yaml(p: Path, d: Dict[str, Any]) -> None:
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, sort_keys=False)

def _merged_tmp_yaml(tmp_path: Path, overlay_names: List[str], T: int = 60) -> Path:
    base = _load_yaml(BASE_YAML)
    merged = dict(base)
    for name in overlay_names:
        yml = SCEN / f"{name}.yaml"
        assert yml.exists(), f"Missing overlay file: {yml}"
        merged = _deep_merge(merged, _load_yaml(yml))
    merged.setdefault("simulation", {})
    merged["simulation"]["T"] = int(T)
    out = tmp_path / "params.merged.yaml"
    _write_yaml(out, merged)
    return out

def _first_baseline(res_all: Dict[str, Any]) -> Any:
    raw = res_all.get("raw", {})
    for k in ("nD_up", "nL_up", "ease_migration", "ease_migration_strong", "abatement_up"):
        if k in raw:
            return raw[k]["res_base"]
    k0 = next(iter(raw.keys()))
    return raw[k0]["res_base"]

def _wage_ratio(res) -> np.ndarray:
    wD = np.asarray(res.D.w, dtype=float)
    wL = np.asarray(res.L.w, dtype=float)
    wL = np.maximum(wL, 1e-12)
    rho = wD / wL
    return rho

def _flows_m(res) -> np.ndarray:
    return np.asarray(res.m, dtype=float)

def _y_over_L_series(country_obj) -> np.ndarray | None:
    # common attribute names
    for name in ("y_over_L", "Y_over_L", "y_per_L", "Y_per_L", "yl", "YL"):
        if hasattr(country_obj, name):
            arr = np.asarray(getattr(country_obj, name), dtype=float)
            if arr.size:
                return arr
    # fallback from Y and L
    if hasattr(country_obj, "Y") and hasattr(country_obj, "L"):
        Y = np.asarray(country_obj.Y, dtype=float)
        L = np.maximum(np.asarray(country_obj.L, dtype=float), 1e-12)
        if Y.size and L.size and Y.shape == L.shape:
            return Y / L
    return None

def _finite_mask_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & np.isfinite(b)

def _tail_mean_abs(x: np.ndarray, tail: int = 10) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    start = max(0, x.shape[0] - tail)
    y = x[start:]
    y = y[np.isfinite(y)]
    if y.size == 0:
        # fallback to entire series finite entries
        y = x[np.isfinite(x)]
    return float(np.nanmean(np.abs(y))) if y.size > 0 else np.nan


# ---------------- tests ----------------

@pytest.mark.fast
def test_symmetry_closed_migration(tmp_path: Path):
    merged_yaml = _merged_tmp_yaml(tmp_path, ["symmetric_baseline_closed"], T=60)
    res_all = run_all(base_yaml_path=str(merged_yaml), T=60)
    base = _first_baseline(res_all)

    # flows off
    m = _flows_m(base)
    assert np.allclose(m, 0.0, atol=1e-12), "Migration should be off under closed-migration symmetry."

    # prefer Y/L equality; ignore NaNs; otherwise fall back to wage ratio ~ 1
    yL_D = _y_over_L_series(base.D)
    yL_L = _y_over_L_series(base.L)
    if yL_D is not None and yL_L is not None and yL_D.size and yL_L.size:
        yL_D = np.asarray(yL_D, dtype=float)
        yL_L = np.asarray(yL_L, dtype=float)
        mask = _finite_mask_pair(yL_D, yL_L)
        if mask.any():
            diff = np.nanmax(np.abs(yL_D[mask] - yL_L[mask]))
            assert diff < 1e-8, f"Symmetric Y/L should coincide; max finite diff = {diff}"
        else:
            # fallback to wage ratio
            rho = _wage_ratio(base)
            rho = rho[np.isfinite(rho)]
            assert rho.size and np.nanmax(np.abs(rho - 1.0)) < 1e-6, "Wage ratio should be ~1 under symmetry."
    else:
        rho = _wage_ratio(base)
        rho = rho[np.isfinite(rho)]
        assert rho.size and np.nanmax(np.abs(rho - 1.0)) < 1e-6, "Wage ratio should be ~1 under symmetry."


@pytest.mark.fast
def test_KL_gap_closed_has_no_migration(tmp_path: Path):
    merged_yaml = _merged_tmp_yaml(tmp_path, ["KL_gap_closed"], T=60)
    res_all = run_all(base_yaml_path=str(merged_yaml), T=60)
    base = _first_baseline(res_all)

    # still closed migration
    m = _flows_m(base)
    assert np.allclose(m, 0.0, atol=1e-12), "Migration should be off for KL_gap_closed."

    # wage ratio reflects initial K/L gap (use any finite t=0 or earliest finite point)
    rho = _wage_ratio(base)
    # find first finite element
    idx = np.argmax(np.isfinite(rho))
    assert np.isfinite(rho[idx]), "No finite wage-ratio observation found."
    assert abs(float(rho[idx]) - 1.0) > 1e-6, "Initial (or earliest finite) wage ratio should differ from 1 under a K/L gap."


@pytest.mark.fast
def test_open_migration_one_knob_changes_flows(tmp_path: Path):
    """
    KL_gap_closed vs (KL_gap_closed + mu-only open migration):
      - flows are positive under μ-only opening
      - D's active labor (L_D) is not lower (tail)
      - L's population (N_L) is not higher (tail)
    This avoids over-asserting wage-ratio convergence, which can be calibration-sensitive.
    """
    # closed baseline
    merged_closed = _merged_tmp_yaml(tmp_path, ["KL_gap_closed"], T=60)
    res_closed = run_all(base_yaml_path=str(merged_closed), T=60)
    base_closed = _first_baseline(res_closed)

    # μ-only open case (optionally include defaults if present)
    overlays_open = ["KL_gap_closed"]
    if (SCEN / "open_migration_defaults.yaml").exists():
        overlays_open.append("open_migration_defaults")
    overlays_open.append("KL_gap_open_mu_only")

    merged_open = _merged_tmp_yaml(tmp_path, overlays_open, T=60)
    res_open = run_all(base_yaml_path=str(merged_open), T=60)
    base_open = _first_baseline(res_open)

    # 1) positive flows somewhere
    m_open = _flows_m(base_open)
    finite_m = m_open[np.isfinite(m_open)]
    assert finite_m.size and np.nanmax(np.abs(finite_m)) > 1e-10, \
        "Open migration (μ only) should produce positive flows."

    # 2) stocks move in the expected directions (compare tail medians for robustness)
    def tail_median(x, k=10):
        tail = np.asarray(x)[-k:]
        tail = tail[np.isfinite(tail)]
        return float(np.nanmedian(tail)) if tail.size else np.nan

    L_D_closed_tail = tail_median(base_closed.D.L)
    L_D_open_tail   = tail_median(base_open.D.L)
    N_L_closed_tail = tail_median(base_closed.L.N)
    N_L_open_tail   = tail_median(base_open.L.N)

    assert np.isfinite(L_D_open_tail) and np.isfinite(L_D_closed_tail), "No finite L_D tail values."
    assert np.isfinite(N_L_open_tail) and np.isfinite(N_L_closed_tail), "No finite N_L tail values."

    assert L_D_open_tail >= L_D_closed_tail - 1e-8, \
        "Destination active labor should not be lower under μ-only opening."
    assert N_L_open_tail <= N_L_closed_tail + 1e-8, \
        "Origin population should not be higher under μ-only opening."
