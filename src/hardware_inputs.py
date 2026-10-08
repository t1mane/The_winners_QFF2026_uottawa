"""hardware_inputs.py - everything a hardware or noisy-simulator run needs, rebuilt from the saved sweep.

    from hardware_inputs import load_run, evaluate, top_designs
    run = load_run(penalty=1.0, p=1)                                   # exact problem, depth 1
    run = load_run(penalty=1.0, p=2, sweep_file="sim_sweep_w2.json")   # pairwise-approximation circuit

    run["circuit"]  -> measured Qiskit circuit: transpile it for your backend, then run it
    counts = {...}  -> what the backend returns, e.g. result[0].data.meas.get_counts()
    evaluate(counts, run)    -> P(best), P(feasible), ... judged on the TRUE problem
    top_designs(counts, run) -> most frequent designs with names, for the demo

Bit order: the keys in `counts` are used as-is (int(key, 2)); do NOT reverse them.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
from hamiltonian import (load_scores, design_labels, cost_hamiltonian,  # noqa: E402
                         truncate_hamiltonian)
from qaoa import build_circuit, summarise  # noqa: E402


def load_run(penalty=1.0, p=1, sweep_file="sim_sweep.json", seed=None):
    """Return the circuit, tuned angles and problem data for one (penalty, depth[, seed]) setting.

    seed: pick the entry tuned with this optimiser seed (needed for hardware_params.json, which
          holds several seeds per setting). None = first matching entry.
    """
    entries = json.load(open(ROOT / "results" / "raw" / sweep_file))
    match = [e for e in entries if abs(e["penalty"] - penalty) < 1e-9 and e["p"] == p
             and (seed is None or e.get("seed") == seed)]
    if not match:
        avail = sorted({(e["penalty"], e["p"], e.get("seed")) for e in entries})
        raise ValueError("no entry for penalty=%s p=%s seed=%s in %s; available (penalty, p, seed): %s"
                         % (penalty, p, seed, sweep_file, avail))
    e = match[0]
    budget = e["budget"]
    score, feasible = load_scores(budget=budget)
    H, _ = cost_hamiltonian(score, feasible, penalty)
    if e.get("max_pauli_weight") is not None:
        H = truncate_hamiltonian(H, e["max_pauli_weight"])
    gammas, betas = e["params"]["gammas"], e["params"]["betas"]
    return {"penalty": penalty, "p": p, "seed": e.get("seed"), "budget": budget,
            "max_pauli_weight": e.get("max_pauli_weight"),
            "gammas": gammas, "betas": betas, "H": H,
            "circuit": build_circuit(H, gammas, betas, measure=True),
            "score": score, "feasible": feasible, "labels": design_labels(),
            "shots": e.get("shots", config.SHOTS), "sim_metrics": e["metrics"]}


def counts_to_probs(counts, n_qubits=6):
    probs = np.zeros(2 ** n_qubits)
    total = sum(counts.values())
    for key, v in counts.items():
        probs[int(key.replace(" ", ""), 2)] += v / total
    return probs


def evaluate(counts, run):
    """Quality of a measured distribution (hardware, noisy sim, or ideal sim)."""
    m = summarise(counts_to_probs(counts), run["score"], run["feasible"])
    m["best_label"] = run["labels"][int(m["best_design"], 2)]
    return m


def top_designs(counts, run, k=5):
    """[(bitstring, design name, count, feasible?)] for the k most frequent outcomes."""
    out = []
    for key, v in sorted(counts.items(), key=lambda kv: -kv[1])[:k]:
        i = int(key.replace(" ", ""), 2)
        out.append((key, run["labels"][i], v, bool(run["feasible"][i])))
    return out