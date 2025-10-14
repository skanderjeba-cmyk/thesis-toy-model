#!/usr/bin/env python3
"""
run_scenarios.py

Front-controller to run the existing Phase-1 harness (make_all_figures.py)
under different scenario overlay chains, and archive outputs into runs/.
Additionally (Step 4), it creates *canonical professor figures* by copying
the best-matching generated PNGs to canonical filenames:

  - fig_capital_dilution_<scenario>.png
  - fig_emissions_ratio_<scenario>.png
  - fig_wage_ratio_<scenario>.png           (optional)
  - fig_migration_diagnostics_<scenario>.png (optional)

Usage examples:

  # A. Symmetric closed
  python run_scenarios.py --scenarios symmetric_baseline_closed

  # B. K/L gap closed
  python run_scenarios.py --scenarios KL_gap_closed

  # C1. K/L gap + open migration (μ only)   (note the chain order)
  python run_scenarios.py --scenarios KL_gap_closed,open_migration_defaults,KL_gap_open_mu_only

  # C2. K/L gap + open migration (τ_H only)
  python run_scenarios.py --scenarios KL_gap_closed,open_migration_defaults,KL_gap_open_tauH_only

  # With a faster horizon for quick testing
  python run_scenarios.py --scenarios symmetric_baseline_closed --T 120
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import yaml

# Use your overlay validator to catch mistakes early
from model.params import load_params_with_overlays

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTROOT = REPO_ROOT / "runs"
PARAMS_PATH = REPO_ROOT / "params.yaml"
PHASE1_HARNESS = REPO_ROOT / "make_all_figures.py"


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge (overlay onto base), returning a new dict."""
    if not isinstance(base, dict):
        return overlay
    result = dict(base)
    for k, v in (overlay or {}).items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def git_commit_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "NA"
    except Exception:
        return "NA"


def iso_stamp_utc() -> str:
    # Example: 2025-10-10T1530Z
    return datetime.utcnow().strftime("%Y-%m-%dT%H%MZ")


def ensure_clean_root_outputs(keep: bool) -> None:
    """Remove root-level figures/ and results/ unless keep=True."""
    if keep:
        return
    for d in ("figures", "results"):
        p = REPO_ROOT / d
        if p.exists():
            shutil.rmtree(p)


def move_outputs_into(run_dir: Path) -> Dict[str, int]:
    """Move figures/ and results/ into run_dir, returning counts."""
    counts = {"figures": 0, "csv": 0, "txt": 0}
    fig_src = REPO_ROOT / "figures"
    res_src = REPO_ROOT / "results"
    fig_dst = run_dir / "figures"
    res_dst = run_dir / "results"
    fig_dst.mkdir(parents=True, exist_ok=True)
    res_dst.mkdir(parents=True, exist_ok=True)

    if fig_src.exists():
        for item in fig_src.iterdir():
            shutil.move(str(item), str(fig_dst / item.name))
        shutil.rmtree(fig_src)
        counts["figures"] = len(list(fig_dst.glob("*.png")))

    if res_src.exists():
        for item in res_src.iterdir():
            shutil.move(str(item), str(res_dst / item.name))
        shutil.rmtree(res_src)
        counts["csv"] = len(list(res_dst.glob("*.csv")))
        counts["txt"] = len(list(res_dst.glob("*.txt")))
    return counts


# ---------- Step 4 helpers: choose & copy canonical figures ----------

def _score_filename(name_lower: str, keywords: List[str]) -> int:
    """Simple keyword score for fuzzy selection among figure names."""
    return sum(1 for kw in keywords if kw in name_lower)


def _best_match(fig_dir: Path, keywords: List[str]) -> Optional[Path]:
    """Pick the best matching PNG in fig_dir for the given keyword list."""
    candidates = list(fig_dir.glob("*.png"))
    if not candidates:
        return None
    scored = []
    for p in candidates:
        s = _score_filename(p.name.lower(), keywords)
        if s > 0:
            scored.append((s, -len(p.name), p))  # prefer more matches; break ties by shorter name
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][2]


def create_canonical_professor_figs(run_dir: Path, scen_slug: str, merged_cfg: Dict[str, Any]) -> Dict[str, str]:
    """
    Copy the best-matching generated figures into canonical filenames expected for the pack.
    Returns a dict target_name -> source (or 'MISSING').
    """
    fig_dir = run_dir / "figures"
    results: Dict[str, str] = {}

    # Targets and their indicative keywords (all lowercased; OR policy)
    targets = {
        f"fig_capital_dilution_{scen_slug}.png": [
            "capital", "y_l", "y-l", "yperL", "y per l", "output_per_worker", "output-per-worker", "productivity", "yl"
        ],
        f"fig_emissions_ratio_{scen_slug}.png": [
            "emission", "emissions", "ratio", "m_over_e", "m/e", "recorded", "E_t", "m_e", "me"
        ],
        f"fig_wage_ratio_{scen_slug}.png": [
            "wage", "varrho", "w_d", "w_l", "wage_ratio", "wageratio"
        ],
        # Heuristic; if a dedicated diagnostics panel exists, we’ll catch it; otherwise we’ll pick the most “migration” looking plot
        f"fig_migration_diagnostics_{scen_slug}.png": [
            "migration", "m_t", "flow", "mig", "population", "n_l", "l_d", "varrho"
        ],
    }

    # Normalize keyword tokens to lowercase
    for k in list(targets.keys()):
        targets[k] = [kw.lower() for kw in targets[k]]

    for canonical_name, keywords in targets.items():
        src = _best_match(fig_dir, keywords)
        if src is None:
            results[canonical_name] = "MISSING"
        else:
            dst = fig_dir / canonical_name
            shutil.copy2(src, dst)
            results[canonical_name] = src.name

    return results


def append_professor_figs_summary(run_dir: Path, mapping: Dict[str, str]) -> None:
    lines = ["", "Canonical professor figures:"]
    for tgt, src in mapping.items():
        if src == "MISSING":
            lines.append(f"  - {Path(tgt).name}: MISSING (no close match found among generated figures)")
        else:
            lines.append(f"  - {Path(tgt).name}: copied from {src}")
    with (run_dir / "run_summary.txt").open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------- Summary writer ----------

def write_run_summary(run_dir: Path, scen_slug: str, overlays: List[str], T_used: int, counts: Dict[str, int]) -> None:
    summary = [
        "=== Scenario Run Summary ===",
        f"Timestamp (UTC): {iso_stamp_utc()}",
        f"Repo: {REPO_ROOT}",
        f"Git commit: {git_commit_hash()}",
        f"Scenario slug: {scen_slug}",
        f"Overlays (in order): {', '.join(overlays)}",
        f"Horizon T: {T_used}",
        "",
        "Artifacts:",
        f"  figures/: {counts.get('figures', 0)} .png files",
        f"  results/: {counts.get('csv', 0)} .csv, {counts.get('txt', 0)} .txt",
        "",
        "Notes:",
        "  - Outputs are self-contained under this run directory.",
        "  - The active params.yaml in repo root has been restored to its original contents.",
        "",
    ]
    (run_dir / "run_summary.txt").write_text("\n".join(summary), encoding="utf-8")


# ---------- Main ----------

def main() -> int:
    ap = argparse.ArgumentParser(description="Scenario runner that archives outputs into runs/ and prepares canonical figures")
    ap.add_argument(
        "--scenarios",
        required=True,
        help="Comma-separated overlay names (files must exist under scenarios/), applied in order.",
    )
    ap.add_argument(
        "--outroot",
        default=str(DEFAULT_OUTROOT),
        help=f"Root output folder (default: {DEFAULT_OUTROOT})",
    )
    ap.add_argument(
        "--T",
        type=int,
        default=None,
        help="Optional horizon override; applied on top of merged YAML before the run.",
    )
    ap.add_argument(
        "--keep-root-outputs",
        action="store_true",
        help="Do not delete or move figures/ and results/ in repo root after the run.",
    )
    args = ap.parse_args()

    overlay_names = [x.strip() for x in args.scenarios.split(",") if x.strip()]
    if not overlay_names:
        ap.error("Provide at least one overlay via --scenarios name[,name2,...]")

    # Resolve overlay paths and check existence
    overlay_paths = [REPO_ROOT / "scenarios" / f"{name}.yaml" for name in overlay_names]
    missing = [str(p) for p in overlay_paths if not p.exists()]
    if missing:
        print("ERROR: overlay file(s) not found:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 2

    outroot = Path(args.outroot).resolve()
    outroot.mkdir(parents=True, exist_ok=True)

    scen_slug = "__".join(overlay_names)  # readable & sortable
    run_dir = outroot / f"{iso_stamp_utc()}_{scen_slug}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build merged YAML (base + overlays)
    base_cfg = load_yaml(PARAMS_PATH)
    merged_cfg = dict(base_cfg)
    for p in overlay_paths:
        merged_cfg = deep_merge(merged_cfg, load_yaml(p))

    # Optional T override
    if args.T is not None:
        merged_cfg.setdefault("simulation", {})
        merged_cfg["simulation"]["T"] = int(args.T)

    # 2) Validate via your loader (raises if invalid)
    try:
        _ = load_params_with_overlays(str(PARAMS_PATH), [str(p) for p in overlay_paths])
        # If T override was provided, we still write merged_cfg for the harness call below.
    except Exception as e:
        print(f"ERROR: overlay validation failed: {e}", file=sys.stderr)
        return 3

    # 3) Swap params.yaml (with backup), run harness, move outputs, restore.
    backup_path = REPO_ROOT / f"params.backup.{iso_stamp_utc()}.yaml"
    try:
        # Backup current params.yaml
        shutil.copy2(PARAMS_PATH, backup_path)

        # Write merged_cfg to params.yaml for the harness
        save_yaml(PARAMS_PATH, merged_cfg)

        # Clean root outputs unless user wants to keep them
        ensure_clean_root_outputs(keep=args.keep_root_outputs)

        # Run Phase-1 harness (unchanged)
        print("=== Scenario run started ===")
        print(f"Scenario overlays: {overlay_names}")
        print(f"Writing merged params to: {PARAMS_PATH.name}")
        print(f"Calling harness: {PHASE1_HARNESS.name}")
        sys.stdout.flush()

        subprocess.run([sys.executable, str(PHASE1_HARNESS)], check=True, cwd=str(REPO_ROOT))

        print("Harness finished; moving outputs into the run directory...")
        counts = move_outputs_into(run_dir)

        # Determine T used (read back from merged_cfg)
        T_used = int(merged_cfg.get("simulation", {}).get("T", base_cfg.get("simulation", {}).get("T", 200)))

        write_run_summary(run_dir, scen_slug, overlay_names, T_used, counts)

        # ---------- Step 4: create canonical professor figures ----------
        canonical_map = create_canonical_professor_figs(run_dir, scen_slug, merged_cfg)
        append_professor_figs_summary(run_dir, canonical_map)

        print("=== Scenario run finished successfully ===")
        print(f"Run directory: {run_dir}")
        return 0

    finally:
        # Restore original params.yaml
        if backup_path.exists():
            shutil.copy2(backup_path, PARAMS_PATH)
            try:
                backup_path.unlink()
            except Exception:
                pass
        print(f"params.yaml restored to original contents.")


if __name__ == "__main__":
    raise SystemExit(main())
