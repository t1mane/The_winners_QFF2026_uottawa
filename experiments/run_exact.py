"""Exact optimum, random feasible draws, and the two greedy baselines.

Run from the repository root:

    python experiments/run_exact.py

Reads the budget and the draw count from config.py. Writes
results/raw/exact_baselines.json. Sampling seeds are 11, 22 and 33.
"""

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
import encoding  # noqa: E402
from baselines import (  # noqa: E402
    BASELINE_SEEDS,
    exact_enumeration,
    greedy_absorption_first,
    greedy_coordinate,
    random_feasible,
)
from metrics import (  # noqa: E402
    counts_from_samples,
    exact_reference,
    failure_frequency,
    first_optimum_draw,
    prefix_curve,
    summarize_samples,
)

OUTPUT_PATH = ROOT / "results" / "raw" / "exact_baselines.json"
AUDIT_PENALTIES = (0.0, 0.25, 1.0, 2.0, 5.0, 20.0)
SAVED_ENERGY_PENALTIES = (0.25, 2.0)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _publish(design, scores):
    if design is None:
        return None
    published = encoding.public_design(design)
    published["normalized_score"] = float(scores[design["index"]])
    return published


def _rank_of(design, ranked):
    if design is None:
        return None
    for position, candidate in enumerate(ranked, start=1):
        if candidate["index"] == design["index"]:
            return position
    return None


def _method_block(name, result, designs, reference, scores, ranked):
    design = result["design"]
    samples = [] if design is None else [design["index"]]
    summary = summarize_samples(samples, designs, reference=reference)
    return {
        "method": name,
        "rule": result["rule"],
        "n_queries": result["n_queries"],
        "elapsed_s": result["elapsed_s"],
        "reads_every_design": result["reads_every_design"],
        "feasible_rank": _rank_of(design, ranked),
        "design": _publish(design, scores),
        "metrics": summary,
    }


def _fmt(value, digits=6):
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def main():
    budget = getattr(config, "COST_BUDGET", None)
    n_draws = int(getattr(config, "SHOTS", 4096))
    designs = encoding.load_designs(budget=budget)
    for design in designs:
        material_id, geometry_id, layer_id, valid = encoding.decode(design["index"])
        if encoding.encode(material_id, geometry_id, layer_id) != design["index"] or not valid:
            print("encoding round-trip failed")
            return 1

    scores, low, high = encoding.normalized_scores(designs, scale="feasible_minmax")
    reference = exact_reference(designs)
    exact = exact_enumeration(designs)
    if exact["design"] is None:
        print("no feasible design at the configured budget")
        return 1
    if exact["design"]["bitstring"] != reference["best_bitstring"]:
        print("exact enumeration disagrees with the metric reference")
        return 1

    audits = []
    for scale in ("feasible_minmax", "catalogue_max"):
        for penalty in AUDIT_PENALTIES:
            audits.append(encoding.audit_penalty(designs, penalty, scale=scale))
    energy_tables = {
        f"M={penalty:g}|feasible_minmax": encoding.build_cost_table(designs, penalty)["energy"].tolist()
        for penalty in SAVED_ENERGY_PENALTIES
    }

    absorption = greedy_absorption_first(designs)
    coordinate = greedy_coordinate(designs)
    random_runs = []
    for seed in BASELINE_SEEDS:
        draw = random_feasible(designs, n_draws=n_draws, seed=seed)
        summary = summarize_samples(draw["samples"], designs, reference=reference)
        curve = prefix_curve(draw["samples"], designs, reference=reference)
        random_runs.append({
            "seed": seed,
            "n_draws": n_draws,
            "pool_size": draw["pool_size"],
            "n_pool_queries": draw["n_pool_queries"],
            "pool_seconds": draw["pool_seconds"],
            "draw_seconds": draw["draw_seconds"],
            "elapsed_s": draw["elapsed_s"],
            "first_optimum_draw": first_optimum_draw(draw["samples"], reference["optimum_indices"]),
            "counts": counts_from_samples(draw["samples"], designs),
            "metrics": summary,
            "prefix_curve": curve,
        })

    ranked = exact["ranked"]
    payload = {
        "description": (
            "Classical reference for the frozen six-qubit catalogue. "
            "Exact enumeration maximises lifetime-averaged efficiency subject to "
            "process compatibility and the cost budget. Random baselines draw "
            "uniformly with replacement from the feasible designs. The absorption-first "
            "and coordinate greeds are deterministic and do not rank every design."
        ),
        "code_commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "scores_sha256": _sha256(encoding.SCORES_PATH),
        "catalogue_sha256": _sha256(ROOT / "data" / "catalogue.json"),
        "budget_usd_m2": budget,
        "n_designs": len(designs),
        "n_categorical": sum(1 for design in designs if design["category_valid"]),
        "n_compatible": sum(1 for design in designs if design["compatible"]),
        "n_within_budget": sum(1 for design in designs if design["within_budget"]),
        "n_feasible": sum(1 for design in designs if design["feasible"]),
        "score_scale": {
            "name": "feasible_minmax",
            "lo": low,
            "hi": high,
            "definition": (
                "r = (efficiency - min feasible efficiency) / "
                "(max feasible efficiency - min feasible efficiency). "
                "The best feasible design has r = 1. This matches src/hamiltonian.py."
            ),
            "catalogue_max_definition": (
                "r = efficiency / max efficiency over all 64 designs, so r is in [0, 1]. "
                "Recorded for the penalty check. QAOA training uses feasible_minmax."
            ),
        },
        "sampling": {
            "seeds": list(BASELINE_SEEDS),
            "draws_per_seed": n_draws,
            "optimizer_seed_in_config": getattr(config, "SEED", None),
            "note": (
                "Sampling seeds drive numpy Generator.choice on the feasible pool. "
                "config.SEED is the QAOA optimizer seed and is not used here."
            ),
        },
        "exact": _method_block("exact_enumeration", exact, designs, reference, scores, ranked),
        "ranked_feasible": [_publish(design, scores) for design in ranked],
        "absorption_first": {
            **_method_block("absorption_first", absorption, designs, reference, scores, ranked),
            "used_fallback": absorption["used_fallback"],
            "search_order": absorption["search_order"],
            "geometry_absorption_proxy": absorption["geometry_absorption_proxy"],
        },
        "coordinate_greedy": {
            **_method_block("coordinate_greedy", coordinate, designs, reference, scores, ranked),
            "trace": coordinate["trace"],
        },
        "random_feasible": random_runs,
        "random_failure_frequency": failure_frequency([run["metrics"] for run in random_runs]),
        "penalty_audit": audits,
        "penalty_threshold": {
            "feasible_minmax": encoding.penalty_threshold(designs, scale="feasible_minmax"),
            "catalogue_max": encoding.penalty_threshold(designs, scale="catalogue_max"),
        },
        "cost_tables": energy_tables,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2)
        handle.write("\n")

    optimum = exact["design"]
    print(
        f"Exact optimum: {optimum['bitstring']}  "
        f"{optimum['material']} / {optimum['geometry']} / {optimum['passivation']}"
    )
    print(
        f"  efficiency {_fmt(optimum['efficiency'])}   cost {_fmt(optimum['cost'], 4)} USD/m2"
        f"   feasible {payload['n_feasible']}/{payload['n_designs']}"
        f"   queries {exact['n_queries']}   seconds {_fmt(exact['elapsed_s'], 6)}"
    )
    for label, result in (("Absorption-first", absorption), ("Coordinate greedy", coordinate)):
        design = result["design"]
        summary = summarize_samples([design["index"]], designs, reference=reference)
        print(
            f"{label}: {design['bitstring']}  "
            f"{design['material']} / {design['geometry']} / {design['passivation']}"
        )
        print(
            f"  efficiency {_fmt(design['efficiency'])}"
            f"   gap_pp {_fmt(summary['efficiency_gap_percentage_points'], 4)}"
            f"   rank {_rank_of(design, ranked)}"
            f"   queries {result['n_queries']}"
        )
    chance = 1.0 / payload["n_feasible"]
    print(f"Random feasible, {n_draws} draws, seeds {list(BASELINE_SEEDS)}")
    print(f"  chance p_best = {_fmt(chance)} (1/{payload['n_feasible']})")
    for run in random_runs:
        metrics = run["metrics"]
        print(
            f"  seed {run['seed']}: p_best={_fmt(metrics['p_best'])}"
            f"  top3={_fmt(metrics['top_k_hit_rate']['3'])}"
            f"  mean_r={_fmt(metrics['mean_feasible_score'])}"
            f"  best_eff={_fmt(metrics['best_feasible_efficiency'])}"
            f"  gap_pp={_fmt(metrics['efficiency_gap_percentage_points'], 4)}"
            f"  first_hit={run['first_optimum_draw']}"
        )
    print("Penalty audit (minimizer, matches physical optimum, infeasible gap)")
    for audit in audits:
        print(
            f"  {audit['scale']} M={audit['penalty']:g}: "
            f"{audit['global_minimizer_bitstring']}"
            f"  match={audit['matches_physical_optimum']}"
            f"  gap={_fmt(audit['infeasible_energy_gap'], 4)}"
        )
    print(f"saved {OUTPUT_PATH}")

    for audit in audits:
        if audit["penalty"] == 2.0 and (
            not audit["matches_physical_optimum"] or not audit["all_infeasible_worse"]
        ):
            print("M=2 does not keep the physical optimum on every scale")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
