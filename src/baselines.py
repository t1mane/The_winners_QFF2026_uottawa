"""Classical references for the frozen 64-design catalogue.

Exact enumeration reads every design and maximises lifetime-averaged efficiency
among designs that are process-compatible and inside the budget. Ties break
toward lower cost, then lower state index.

Uniform random draws with replacement from that feasible pool. Building the
pool costs one feasibility read per catalogue row. Sampling seeds are 11, 22
and 33, stated here so the same draws can be reused across quantum settings.
They are sampling seeds, not the QAOA optimizer seed.

Two deterministic greeds, neither of which ranks the full catalogue:

    absorption_first
        Order materials by the catalogue record efficiency (high first),
        geometries by the optical proxy (1 - reflectance) * path_enhancement
        (high first), and passivations by S_on_cSi (low first). Ties in each
        list break toward lower step cost, then lower category id.
        Accept the first candidate that the score table marks feasible.
        If the walk finds none, fall back to the lowest feasible state index.

    coordinate_greedy
        Start at the lowest feasible state index. Then, in category order
        material, geometry, passivation, try all four choices of that category
        while holding the others fixed. Keep the feasible choice with the
        highest efficiency, then lower cost, then lower category index.
        One pass only. The current feasible design is always one of the four
        candidates, so the pass stays feasible.

Every candidate whose feasibility is read counts as one query, including a
design read again on a later step.
"""

import json
import time
from pathlib import Path

import numpy as np

import encoding

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE_PATH = ROOT / "data" / "catalogue.json"

BASELINE_SEEDS = (11, 22, 33)
DEFAULT_DRAWS = 4096
CATEGORY_FIELDS = ("material_id", "geometry_id", "layer_id")


def enumerate_best(scores, feasible):
    """Return (index, score) of the best feasible entry, or (None, None).

    On an equal score the lower index wins. Infeasible entries are ignored
    even when their score is larger.
    """
    best_index = None
    best_score = None
    for index, (score, ok) in enumerate(zip(scores, feasible)):
        if not ok:
            continue
        if best_score is None or score > best_score:
            best_index = index
            best_score = score
    return best_index, best_score


def exact_enumeration(designs):
    """Maximise efficiency over the feasible rows. Reads every design once."""
    ordered = encoding.require_ordered(designs)
    started = time.perf_counter()
    best = None
    for design in ordered:
        if not design["feasible"]:
            continue
        rank = (design["efficiency"], -design["cost"], -design["index"])
        if best is None or rank > best[0]:
            best = (rank, design)
    ranked = sorted(
        (design for design in ordered if design["feasible"]),
        key=lambda design: (-design["efficiency"], design["cost"], design["index"]),
    )
    elapsed = time.perf_counter() - started
    return {
        "method": "exact_enumeration",
        "rule": (
            "Maximise lifetime-averaged efficiency among process-compatible designs "
            "within the budget. Ties break toward lower cost, then lower state index."
        ),
        "design": None if best is None else best[1],
        "ranked": ranked,
        "n_queries": len(ordered),
        "n_feasible": len(ranked),
        "elapsed_s": elapsed,
        "reads_every_design": True,
    }


def random_feasible(designs, n_draws=DEFAULT_DRAWS, seed=11):
    """Draw uniformly with replacement from the feasible state indices.

    The pool is sorted by ascending state index before Generator.choice, so a
    seed names one sample sequence.
    """
    if isinstance(n_draws, bool) or not isinstance(n_draws, (int, np.integer)):
        raise TypeError("n_draws must be an integer")
    n_draws = int(n_draws)
    if n_draws < 1:
        raise ValueError("n_draws must be at least 1")
    ordered = encoding.require_ordered(designs)
    started = time.perf_counter()
    pool = [design["index"] for design in ordered if design["feasible"]]
    pool_seconds = time.perf_counter() - started
    if not pool:
        raise ValueError("the feasible pool is empty, so random feasible draws are undefined")
    draw_started = time.perf_counter()
    generator = np.random.default_rng(seed)
    draws = generator.choice(np.asarray(pool, dtype=np.int64), size=n_draws, replace=True)
    draw_seconds = time.perf_counter() - draw_started
    return {
        "method": "random_feasible",
        "rule": (
            "Uniform with replacement over the feasible designs. "
            "The pool is built by reading every catalogue row once, in index order."
        ),
        "seed": int(seed),
        "samples": [int(sample) for sample in draws.tolist()],
        "pool": pool,
        "pool_size": len(pool),
        "n_draws": n_draws,
        "n_pool_queries": len(ordered),
        "pool_seconds": pool_seconds,
        "draw_seconds": draw_seconds,
        "elapsed_s": pool_seconds + draw_seconds,
    }


def _load_catalogue(catalogue=None, catalogue_path=None):
    if catalogue is not None:
        return catalogue
    path = Path(catalogue_path) if catalogue_path is not None else CATALOGUE_PATH
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _absorption_proxy(geometry):
    values = geometry["values"]
    return (1.0 - float(values["reflectance"])) * float(values["path_enhancement"])


def greedy_absorption_first(designs, catalogue=None, catalogue_path=None):
    """First feasible design in the absorption-first category order."""
    ordered = encoding.require_ordered(designs)
    catalogue = _load_catalogue(catalogue, catalogue_path)
    by_index = {design["index"]: design for design in ordered}
    materials = sorted(
        catalogue["materials"],
        key=lambda item: (
            -float(item["values"]["record_efficiency"]),
            float(item["values"]["base_cost_usd_m2"]),
            int(item["id"]),
        ),
    )
    geometries = sorted(
        catalogue["geometries"],
        key=lambda item: (
            -_absorption_proxy(item),
            float(item["values"]["cost_usd_m2"]),
            int(item["id"]),
        ),
    )
    passivations = sorted(
        catalogue["passivations"],
        key=lambda item: (
            float(item["values"]["S_on_cSi_cm_per_s"]),
            float(item["values"]["cost_usd_m2"]),
            int(item["id"]),
        ),
    )
    started = time.perf_counter()
    n_queries = 0
    chosen = None
    used_fallback = False
    for material in materials:
        for geometry in geometries:
            for passivation in passivations:
                index = encoding.encode(material["id"], geometry["id"], passivation["id"])
                if index not in by_index:
                    raise ValueError(f"design table is missing state {index}")
                n_queries += 1
                candidate = by_index[index]
                if candidate["feasible"]:
                    chosen = candidate
                    break
            if chosen is not None:
                break
        if chosen is not None:
            break
    if chosen is None:
        used_fallback = True
        for candidate in ordered:
            n_queries += 1
            if candidate["feasible"]:
                chosen = candidate
                break
    elapsed = time.perf_counter() - started
    return {
        "method": "absorption_first",
        "rule": (
            "Walk materials by catalogue record efficiency, geometries by "
            "(1 - reflectance) * path_enhancement, and passivations by lower "
            "S_on_cSi. Accept the first design the score table marks feasible. "
            "Each checked candidate counts as one query. The walk does not "
            "compare efficiencies."
        ),
        "design": chosen,
        "n_queries": n_queries,
        "elapsed_s": elapsed,
        "used_fallback": used_fallback,
        "reads_every_design": False,
        "search_order": {
            "materials": [item["name"] for item in materials],
            "geometries": [item["name"] for item in geometries],
            "passivations": [item["name"] for item in passivations],
        },
        "geometry_absorption_proxy": [
            {"name": item["name"], "proxy": _absorption_proxy(item)} for item in geometries
        ],
    }


def greedy_coordinate(designs):
    """One coordinate-descent pass from the lowest feasible state index."""
    ordered = encoding.require_ordered(designs)
    by_ids = {
        (design["material_id"], design["geometry_id"], design["layer_id"]): design
        for design in ordered
    }
    started = time.perf_counter()
    n_queries = 0
    start = None
    for design in ordered:
        n_queries += 1
        if design["feasible"]:
            start = design
            break
    if start is None:
        return {
            "method": "coordinate_greedy",
            "rule": "No feasible start exists.",
            "design": None,
            "n_queries": n_queries,
            "elapsed_s": time.perf_counter() - started,
            "trace": [],
            "reads_every_design": False,
        }
    trace = [{"stage": "start", "bitstring": start["bitstring"], "n_queries": n_queries}]
    current = [start["material_id"], start["geometry_id"], start["layer_id"]]
    for category, field in enumerate(CATEGORY_FIELDS):
        best_record = None
        best_key = None
        for value in range(encoding.N_OPTIONS):
            trial = current[:]
            trial[category] = value
            n_queries += 1
            candidate = by_ids[tuple(trial)]
            if not candidate["feasible"]:
                continue
            rank = (candidate["efficiency"], -candidate["cost"], -value)
            if best_key is None or rank > best_key:
                best_key = rank
                best_record = candidate
        if best_record is None:
            raise RuntimeError("coordinate greedy lost the feasible design it started from")
        current = [
            best_record["material_id"],
            best_record["geometry_id"],
            best_record["layer_id"],
        ]
        trace.append({
            "stage": field,
            "bitstring": best_record["bitstring"],
            "n_queries": n_queries,
        })
    elapsed = time.perf_counter() - started
    return {
        "method": "coordinate_greedy",
        "rule": (
            "Start at the lowest feasible state index. For material, then geometry, "
            "then passivation, try all four values and keep the feasible one with "
            "highest efficiency, then lower cost, then lower category index. "
            "One pass. Each try counts as one query, including the start scan."
        ),
        "design": by_ids[tuple(current)],
        "n_queries": n_queries,
        "elapsed_s": elapsed,
        "trace": trace,
        "reads_every_design": False,
    }
