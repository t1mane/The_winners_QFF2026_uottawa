"""make_hardware_params.py - tuned QAOA angles (gammas/betas) for hardware runs, one entry per seed.

The seed controls the optimiser's random restarts (restart 0 is always the same linear-ramp guess).
Angles are plain numbers, so they are portable: a run on any machine or on a real IBM chip
uses exactly these values. The metrics stored with them are exact (noise-free simulation).

Usage, from the repo root with the venv active:
    python experiments/make_hardware_params.py                    # penalty 2, depth 1, seeds 11 22 33
    python experiments/make_hardware_params.py --penalty 2 --depth 1 --seeds 11 22 33 --max-weight 2

Writes / updates results/raw/hardware_params.json. Entries use the same schema as sim_sweep.json, so
src/hardware_inputs.py can load them:
    run = load_run(penalty=2.0, p=1, sweep_file="hardware_params.json", seed=11)
This file is separate from sim_sweep.json on purpose: plotting.py groups sim_sweep.json by penalty,
and extra penalties/seeds there would add stray lines to the figures.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
from hamiltonian import (load_scores, design_labels, cost_hamiltonian,  # noqa: E402
                         truncate_hamiltonian, hamiltonian_diag)
from qaoa import optimise, qaoa_probs, summarise  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--penalty", type=float, default=2.0)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    ap.add_argument("--max-weight", type=int, default=None, help="2 = pairwise-approximation circuit")
    ap.add_argument("--restarts", type=int, default=20)
    ap.add_argument("--out", default="results/raw/hardware_params.json")
    a = ap.parse_args()

    budget = getattr(config, "COST_BUDGET", None)
    score, feasible = load_scores(budget=budget)
    labels = design_labels()
    H, cost = cost_hamiltonian(score, feasible, a.penalty)
    if a.max_weight is not None:
        H = truncate_hamiltonian(H, a.max_weight)
        cost = hamiltonian_diag(H)
    print("budget:", budget, "| feasible: %d of %d" % (feasible.sum(), len(feasible)),
          "| penalty:", a.penalty, "| depth:", a.depth, "| max Pauli weight:", a.max_weight)

    new = []
    for seed in a.seeds:
        opt = optimise(cost, a.depth, n_restarts=a.restarts, seed=seed)
        probs = qaoa_probs(cost, opt["gammas"], opt["betas"])
        m = summarise(probs, score, feasible)
        m["best_label"] = labels[int(m["best_design"], 2)]
        new.append({"penalty": a.penalty, "p": a.depth, "seed": seed, "shots": config.SHOTS,
                    "budget": budget, "max_pauli_weight": a.max_weight, "n_restarts": a.restarts,
                    "params": {"gammas": opt["gammas"], "betas": opt["betas"]},
                    "expected_cost": opt["value"], "total_evals": opt["total_evals"],
                    "restart_values": opt["restart_values"], "metrics": m})
        print("seed %-3d gammas=%s betas=%s | exact P(best)=%.3f P(feasible)=%.3f (random %.3f)" % (
            seed, [round(x, 4) for x in opt["gammas"]], [round(x, 4) for x in opt["betas"]],
            m["p_best"], m["p_feasible"], m["random_p_best"]))

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    old = json.load(open(out)) if out.exists() else []
    keys = set((e["penalty"], e["p"], e["seed"], e["max_pauli_weight"]) for e in new)
    kept = [e for e in old if (e["penalty"], e["p"], e["seed"], e.get("max_pauli_weight")) not in keys]
    with open(out, "w") as f:
        json.dump(kept + new, f, indent=1)
    print("saved", out, "(%d entries)" % (len(kept) + len(new)))


if __name__ == "__main__":
    main()