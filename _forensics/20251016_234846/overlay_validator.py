import sys, json, pathlib, hashlib
from typing import Dict, Any
try:
    import yaml
except Exception as e:
    print("[ERR] PyYAML not available:", e)
    sys.exit(0)

repo = pathlib.Path(r"C:\Users\skand\thesis-toy-model")
sc = repo / "scenarios"
params = repo / "params.yaml"
ok = True
report_lines = []
def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k,v in d.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    else:
        out[prefix]=d
    return out

def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

allowed = set([
  "globals.alpha","globals.delta","globals.phi_M","globals.g_A","globals.q_star","globals.eta",
  "countries.D.s","countries.D.f","countries.D.epsilon","countries.D.kappa","countries.D.xi","countries.D.theta_bar","countries.D.zeta",
  "countries.L.s","countries.L.f","countries.L.epsilon","countries.L.kappa","countries.L.xi","countries.L.theta_bar","countries.L.zeta",
  "migration.mu","migration.tau_H","migration.m_bar","migration.mig_slack",
  "init.K_D","init.K_L","init.N_D","init.N_L","init.A0_D","init.A0_L","init.M0",
  "sim.T"
])

# Expected one-knob overlays by filename (very strict):
rules = {
  "fertility_gap_L.yaml": set(["countries.L.f"]),
  "KL_gap_open_mu_only.yaml": set(["migration.mu"]),
  "KL_gap_open_tauH_only.yaml": set(["migration.tau_H"]),
  "open_migration_defaults.yaml": set(["migration.mu","migration.tau_H","migration.m_bar","migration.mig_slack"]),
  "KL_gap_closed.yaml": set(["init.K_D","init.K_L"]),
  "symmetric_baseline_closed.yaml": None, # may touch symmetric params; we just list
}

if sc.exists():
    overlays = sorted([p for p in sc.glob("*.yml")] + [p for p in sc.glob("*.yaml")])
else:
    overlays = []
    report_lines.append("[WARN] No scenarios/ folder.")

if params.exists():
    base = load_yaml(params)
else:
    base = {}
    report_lines.append("[WARN] params.yaml missing.")

for p in overlays:
    name = p.name
    data = load_yaml(p) or {}
    flat = flatten(data)
    touched = sorted(flat.keys())
    # Allowed-key filter
    bad_keys = [k for k in touched if k not in allowed]
    if bad_keys:
        ok = False
        report_lines.append(f"[FAIL] {name}: contains non-whitelisted keys -> {bad_keys}")
    # One-knob rule (if specified)
    want = rules.get(name)
    if want is not None:
        # Ignore if overlay is empty (still a warning)
        if not touched:
            report_lines.append(f"[WARN] {name}: overlay is empty")
        extra = [k for k in touched if k not in want]
        if extra:
            ok = False
            report_lines.append(f"[FAIL] {name}: expected only {sorted(want)} but also found {extra}")
        else:
            report_lines.append(f"[OK]   {name}: keys {touched}")
    else:
        report_lines.append(f"[INFO] {name}: keys {touched}")

if not overlays:
    report_lines.append("[WARN] No overlay files found (*.yml|*.yaml)")

print("\n".join(report_lines))
print("\n=== OVERLAY VALIDATOR SUMMARY ===")
print("Status:", "OK" if ok else "ISSUES FOUND")
