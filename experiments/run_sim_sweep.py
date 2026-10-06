"""run_sim_sweep.py - QAOA depth x penalty sweep on the exact simulator.

Run from the repo root:   python experiments/run_sim_sweep.py
Writes results/raw/sim_sweep.json  (read by plotting.py and by the hardware runner,
which reuses the saved gammas/betas so no tuning is spent on real QPU minutes).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
from hamiltonian import load_scores, cost_vector  # noqa: E402
from qaoa import optimise, qaoa_probs, summarise, sample_counts  # noqa: E402

N_RESTARTS = 20


def main():
    score, feasible = load_scores()
    results = []
    print("%-3s %-8s %-9s %-9s %-9s %-8s" % ("p", "penalty", "P(best)", "P(feas)", "P(top3)", "lift"))
    for penalty in config.PENALTY_VALUES:
        cost = cost_vector(score, feasible, penalty)
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
            # shot-limited version: what a real run with config.SHOTS would see
            counts = sample_counts(probs, config.SHOTS, seed=config.SEED)
            m["shot_p_best"] = counts.get(m["best_design"], 0) / config.SHOTS
            results.append({"penalty": penalty, "p": p, "seed": config.SEED, "shots": config.SHOTS,
                            "n_restarts": N_RESTARTS, "params": {"gammas": opt["gammas"], "betas": opt["betas"]},
                            "expected_cost": opt["value"], "total_evals": opt["total_evals"],
                            "restart_values": opt["restart_values"], "metrics": m})
            print("%-3d %-8.1f %-9.3f %-9.3f %-9.3f %-8.1f" % (
                p, penalty, m["p_best"], m["p_feasible"], m["p_top3"], m["lift_vs_random_feasible"]))

    out = ROOT / "results" / "raw" / "sim_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()