"""scripts/preflight_hardware.py - offline checks before anything is sent to IBM. Needs NO token, no network.

    python scripts/preflight_hardware.py
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

res = {"pass": 0, "fail": 0, "warn": 0}


def ok(m):
    res["pass"] += 1
    print("  PASS ", m)


def bad(m):
    res["fail"] += 1
    print("  FAIL ", m)


def warn(m):
    res["warn"] += 1
    print("  WARN ", m)


import config  # noqa: E402

SEEDS, PEN, P, SHOTS = (11, 22, 33), 2.0, 1, 2048
path = ROOT / "results" / "raw" / "hardware_params.json"

print("== 1. hardware_params.json")
if not path.exists():
    bad("results/raw/hardware_params.json missing")
    sys.exit(1)
entries = json.load(open(path))
for s in SEEDS:
    m = [e for e in entries if e["seed"] == s and e["penalty"] == PEN and e["p"] == P]
    if len(m) != 1:
        bad("seed %d: %d matching entries (need exactly 1)" % (s, len(m)))
        continue
    e = m[0]
    g, b = e["params"]["gammas"], e["params"]["betas"]
    fine = len(g) == P and len(b) == P and all(np.isfinite(g + b))
    (ok if fine else bad)("seed %d: %d gamma / %d beta, finite" % (s, len(g), len(b)))
    (ok if e.get("budget") == getattr(config, "COST_BUDGET", None) else bad)(
        "seed %d: budget %s matches config" % (s, e.get("budget")))
    (ok if e.get("max_pauli_weight") is None else warn)(
        "seed %d: max_pauli_weight=%s (None = exact circuit)" % (s, e.get("max_pauli_weight")))

print("== 2. Circuits vs exact simulation")
from hardware_inputs import load_run, evaluate  # noqa: E402
from qaoa import qaoa_probs  # noqa: E402
from hamiltonian import hamiltonian_diag  # noqa: E402
from qiskit import transpile  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402
from qiskit_aer.primitives import SamplerV2  # noqa: E402

runs = {}
for s in SEEDS:
    try:
        r = load_run(penalty=PEN, p=P, sweep_file="hardware_params.json", seed=s)
    except Exception as ex:  # noqa: BLE001
        bad("seed %d: load_run failed: %s" % (s, ex))
        continue
    runs[s] = r
    cost = np.real(hamiltonian_diag(r["H"]))
    exact = qaoa_probs(cost, r["gammas"], r["betas"])
    sv = Statevector(r["circuit"].remove_final_measurements(inplace=False)).probabilities()
    d = np.abs(sv - exact).max()
    (ok if d < 1e-9 else bad)("seed %d: circuit matches exact simulator (max diff %.1e)" % (s, d))
    stored = r["sim_metrics"]["p_best"]
    now = float(exact[int(r["sim_metrics"]["best_design"], 2)])
    (ok if abs(stored - now) < 1e-6 else bad)("seed %d: stored P(best) %.4f matches recomputed %.4f" % (s, stored, now))

print("== 3. Sampling at %d shots (noise-free Aer)" % SHOTS)
for s, r in runs.items():
    qc = transpile(r["circuit"], AerSimulator(), optimization_level=1)
    counts = SamplerV2(seed=s).run([qc], shots=SHOTS).result()[0].data.meas.get_counts()
    m = evaluate(counts, r)
    exp = r["sim_metrics"]["p_feasible"]
    (ok if abs(m["p_feasible"] - exp) < 0.06 else bad)(
        "seed %d: P(feasible) %.3f vs exact %.3f; P(best) %.3f" % (s, m["p_feasible"], exp, m["p_best"]))
    (ok if m["best_design"] == "000111" else warn)("seed %d: best design reported %s" % (s, m["best_design"]))

print("== 4. Scripts compile and cannot submit by accident")
for f in ("experiments/run_hardware.py", "experiments/run_noisy_sim.py", "src/ibm_backend.py"):
    p = ROOT / f
    if not p.exists():
        warn(f + " not present")
        continue
    r = subprocess.run([sys.executable, "-m", "py_compile", str(p)], capture_output=True, text=True)
    (ok if r.returncode == 0 else bad)(f + " compiles" + ("" if r.returncode == 0 else ": " + r.stderr[-200:]))
hw = ROOT / "experiments" / "run_hardware.py"
if hw.exists():
    t = hw.read_text()
    (ok if "--submit" in t else bad)("run_hardware.py requires an explicit --submit flag")
    (ok if "--retrieve" in t else warn)("run_hardware.py has --retrieve")
    (ok if re.search(r"SHOTS\s*=\s*2048", t) else warn)("run_hardware.py SHOTS is 2048")

print("== 5. Secrets")
hits = []
for p in list(ROOT.glob("src/*.py")) + list(ROOT.glob("experiments/*.py")) + list(ROOT.glob("*.sh")) + \
        list(ROOT.glob("results/raw/*.json")) + list(ROOT.glob("docs/*.md")):
    for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
        if re.search(r"[A-Za-z0-9_\-]{40,}", line) and re.search(r"(token|key|apikey|secret)", line, re.I):
            hits.append("%s:%d" % (p.relative_to(ROOT), i))
(ok if not hits else bad)("no token-like strings in source/results/docs" if not hits else "possible secret at " + ", ".join(hits))
gi = (ROOT / ".gitignore").read_text() if (ROOT / ".gitignore").exists() else ""
(ok if re.search(r"^\.env\b", gi, re.M) else bad)(".env is in .gitignore")

print("== 6. Compiled size on a real IBM chip layout (offline fake backend)")
try:
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    fb = FakeSherbrooke()
    for s, r in list(runs.items())[:1]:
        for lvl in (1, 3):
            qc = transpile(r["circuit"], fb, optimization_level=lvl, seed_transpiler=42)
            ops = qc.count_ops()
            two = sum(v for k, v in ops.items() if k in ("ecr", "cx", "cz", "rzz"))
            print("          seed %d, opt level %d: depth %d, two-qubit gates %d" % (s, lvl, qc.depth(), two))
            if lvl == 3 and two > 150:
                warn("%d two-qubit gates: expect heavy noise on hardware; consider pairwise (--max-weight 2)" % two)
            elif lvl == 3:
                ok("compiled two-qubit count %d" % two)
except Exception as ex:  # noqa: BLE001
    warn("fake backend not available (%s): ask the CompE for his compiled depth/2q count" % type(ex).__name__)

print("\nRESULT: %d passed, %d warnings, %d FAILED" % (res["pass"], res["warn"], res["fail"]))
sys.exit(1 if res["fail"] else 0)
