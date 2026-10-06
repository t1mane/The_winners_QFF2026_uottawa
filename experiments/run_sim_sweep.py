"""run_sim_sweep.py - QAOA depth x penalty sweep on the exact simulator.

Run from the repo root:   python experiments/run_sim_sweep.py
Writes results/raw/sim_sweep.json  (read by plotting.py and by the hardware runner,
which reuses the saved gammas/betas so no tuning is spent on real QPU minutes).

Settings come from config.py: SEED, SHOTS, QAOA_DEPTHS, PENALTY_VALUES and
(optional) COST_BUDGET in USD/m2 - designs above the budget count as infeasible.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
from hamiltonian import (load_scores, design_labels, cost_hamiltonian,  # noqa: E402
                         truncate_hamiltonian, hamiltonian_diag)
from qaoa import optimise, qaoa_probs, summarise, sample_counts  # noqa: E402

N_RESTARTS = 20


def main():
    budget = getattr(config, "COST_BUDGET", None)
    max_w = getattr(config, "MAX_PAULI_WEIGHT", None)   # None = exact problem; 2 = pairwise approximation
    score, feasible = load_scores(budget=budget)
    labels = design_labels()
    print("Cost budget:", budget, "| max Pauli weight:", max_w,
          "| feasible designs: %d of %d" % (feasible.sum(), len(feasible)))

    results = []
    best_printed = False
    print("%-3s %-8s %-9s %-9s %-9s %-8s" % ("p", "penalty", "P(best)", "P(feas)", "P(top3)", "lift"))
    for penalty in config.PENALTY_VALUES:
        H, cost = cost_hamiltonian(score, feasible, penalty)
        if max_w is not None:                       # optional: cheaper, approximate circuit
            H = truncate_hamiltonian(H, max_w)
            cost = hamiltonian_diag(H)
        prev = None
        for p in config.QAOA_DEPTHS:
            # Parameter reuse: start from the depth p-1 optimum plus an extra layer with
            # gamma = beta = 0 (which is the identity), so depth p can only match or beat p-1.
            init = None
            if prev is not None and len(prev["gammas"]) == p - 1:
                init = prev["gammas"] + [0.0] + prev["betas"] + [0.0]
            opt = optimise(cost, p, n_restarts=N_RESTARTS, seed=config.SEED, init=init)
            prev = opt
            probs = qaoa_probs(cost, opt["gammas"], opt["betas"])
            m = summarise(probs, score, feasible)
            m["best_label"] = labels[int(m["best_design"], 2)]
            if not best_printed:
                print("Exact best feasible design: %s -> %s" % (m["best_design"], m["best_label"]))
                best_printed = True
            # shot-limited version: what a real run with config.SHOTS would see
            counts = sample_counts(probs, config.SHOTS, seed=config.SEED)
            m["shot_p_best"] = counts.get(m["best_design"], 0) / config.SHOTS
            results.append({"penalty": penalty, "p": p, "seed": config.SEED, "shots": config.SHOTS,
                            "budget": budget, "max_pauli_weight": max_w, "n_restarts": N_RESTARTS,
                            "params": {"gammas": opt["gammas"], "betas": opt["betas"]},
                            "expected_cost": opt["value"], "total_evals": opt["total_evals"],
                            "restart_values": opt["restart_values"], "metrics": m})
            print("%-3d %-8.2f %-9.3f %-9.3f %-9.3f %-8.1f" % (
                p, penalty, m["p_best"], m["p_feasible"], m["p_top3"], m["lift_vs_random_feasible"]))

    name = "sim_sweep.json" if max_w is None else "sim_sweep_w%d.json" % max_w
    out = ROOT / "results" / "raw" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()