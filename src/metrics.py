"""Sample metrics for feasible-design selection.

Quality is reported on the physical efficiency and on the feasible min-max
score r used by the QAOA cost table. The penalized training energy is not a
comparison metric: it changes when the penalty changes.

An unsuccessful sample (no feasible outcome) has feasible fraction 0 and leaves
best, mean and both gaps as None. Those gaps are not filled with zero and the
exact optimum is not substituted.

p_best is the fraction of outcomes equal to the tie-broken exact optimum.
optimum_hit_rate also counts other feasible designs within 1e-9 efficiency of
that optimum. top-k uses the ranked feasible list: higher efficiency, then
lower cost, then lower index. Chance is reported two ways: 1/N over all
encoded states, and 1/n_feasible for a uniform draw from the feasible pool.
"""

import numbers

import encoding

OPTIMUM_TOL = 1e-9
DEFAULT_TOP_K = (1, 3)
DEFAULT_PREFIXES = (1, 4, 16, 64, 256, 4096)


def exact_reference(designs, top_k=DEFAULT_TOP_K, tol=OPTIMUM_TOL):
    """Physical optimum and the top-k index sets on the feasible min-max scale."""
    ordered = encoding.require_ordered(designs)
    feasible = [design for design in ordered if design["feasible"]]
    reference = {
        "n_designs": len(ordered),
        "n_feasible": len(feasible),
        "tol": float(tol),
    }
    if not feasible:
        reference.update({
            "best_index": None,
            "best_bitstring": None,
            "best_efficiency": None,
            "best_cost": None,
            "best_score": None,
            "lo": None,
            "hi": None,
            "optimum_indices": [],
            "top_k_indices": {str(k): [] for k in top_k},
            "ranked_bitstrings": [],
        })
        return reference
    scores, low, high = encoding.normalized_scores(ordered, scale="feasible_minmax")
    ranked = sorted(
        feasible,
        key=lambda design: (-design["efficiency"], design["cost"], design["index"]),
    )
    best = ranked[0]
    optimum = [
        design["index"]
        for design in feasible
        if design["efficiency"] >= best["efficiency"] - tol
    ]
    reference.update({
        "best_index": best["index"],
        "best_bitstring": best["bitstring"],
        "best_efficiency": best["efficiency"],
        "best_cost": best["cost"],
        "best_score": float(scores[best["index"]]),
        "lo": float(low),
        "hi": float(high),
        "optimum_indices": optimum,
        "top_k_indices": {
            str(k): [design["index"] for design in ranked[: min(int(k), len(ranked))]]
            for k in top_k
        },
        "ranked_bitstrings": [design["bitstring"] for design in ranked],
    })
    return reference


def _index_of(key, ordered):
    if isinstance(key, str):
        text = key.replace(" ", "")
        for design in ordered:
            if design["bitstring"] == text:
                return design["index"]
        raise ValueError(f"unknown bitstring {key!r}")
    if isinstance(key, bool) or not isinstance(key, numbers.Integral):
        raise TypeError(f"sample key must be a state index or bitstring, got {key!r}")
    index = int(key)
    if index < 0 or index >= len(ordered):
        raise ValueError(f"state index {index} is outside 0..{len(ordered) - 1}")
    return index


def _weights(samples_or_counts, ordered):
    """Return (weight per index, total weight, ordered samples or None)."""
    n_states = len(ordered)
    if isinstance(samples_or_counts, dict):
        weights = [0.0] * n_states
        total = 0.0
        for key, count in samples_or_counts.items():
            if isinstance(count, bool) or not isinstance(count, numbers.Real) or count < 0:
                raise ValueError(f"count for {key!r} must be a non-negative number")
            weights[_index_of(key, ordered)] += float(count)
            total += float(count)
        return weights, total, None
    if not isinstance(samples_or_counts, (str, bytes)) and hasattr(samples_or_counts, "shape"):
        values = [float(value) for value in samples_or_counts]
        if len(values) != n_states:
            raise ValueError(f"weight vector length {len(values)} != {n_states}")
        if any(value < 0 for value in values):
            raise ValueError("weights must be non-negative")
        return values, float(sum(values)), None
    if isinstance(samples_or_counts, (str, bytes)):
        raise TypeError("pass a sequence of samples, a counts dict, or a weight vector")
    samples = list(samples_or_counts)
    weights = [0.0] * n_states
    for sample in samples:
        weights[_index_of(sample, ordered)] += 1.0
    return weights, float(len(samples)), samples


def _as_count(value):
    if abs(value - round(value)) < 1e-9:
        return int(round(value))
    return value


def _check_flags(design):
    expected = bool(design["compatible"] and design["within_budget"] and design["category_valid"])
    if bool(design["feasible"]) != expected:
        raise ValueError(f"feasibility flags disagree at index {design['index']}")


def summarize_samples(samples_or_counts, designs, reference=None, top_k=DEFAULT_TOP_K, tol=OPTIMUM_TOL):
    """Summarise counts, an ordered sample list, or a length-N weight vector.

    Ordered lists keep shot order for prefix curves. Dict counts and weight
    vectors have no order. Weights may be shot counts or probabilities.
    """
    ordered = encoding.require_ordered(designs)
    for design in ordered:
        _check_flags(design)
    if reference is None:
        reference = exact_reference(ordered, top_k=top_k, tol=tol)
    weights, total, _samples = _weights(samples_or_counts, ordered)
    n_designs = reference["n_designs"]
    n_feasible_designs = reference["n_feasible"]
    chance_uniform = 1.0 / n_designs
    chance_feasible = None if n_feasible_designs == 0 else 1.0 / n_feasible_designs

    violation = {
        "infeasible": 0.0,
        "incompatible": 0.0,
        "over_budget": 0.0,
        "incompatible_and_over_budget": 0.0,
        "invalid_category": 0.0,
    }
    for design, weight in zip(ordered, weights):
        if weight == 0:
            continue
        if not design["category_valid"]:
            violation["invalid_category"] += weight
        if not design["compatible"]:
            violation["incompatible"] += weight
        if not design["within_budget"]:
            violation["over_budget"] += weight
        if not design["compatible"] and not design["within_budget"]:
            violation["incompatible_and_over_budget"] += weight
        if not design["feasible"]:
            violation["infeasible"] += weight

    empty = {
        "n_samples": _as_count(total),
        "n_feasible_samples": 0,
        "n_feasible_designs": n_feasible_designs,
        "feasible_fraction": 0.0,
        "unsuccessful": True,
        "best_feasible_bitstring": None,
        "best_feasible_index": None,
        "best_feasible_material": None,
        "best_feasible_geometry": None,
        "best_feasible_passivation": None,
        "best_feasible_efficiency": None,
        "best_feasible_score": None,
        "mean_feasible_score": None,
        "objective_gap": None,
        "efficiency_gap": None,
        "efficiency_gap_percentage_points": None,
        "p_best": 0.0,
        "optimum_hit_rate": 0.0,
        "top_k_hit_rate": {key: 0.0 for key in reference["top_k_indices"]},
        "chance_p_best_uniform": chance_uniform,
        "chance_p_best_feasible": chance_feasible,
        "lift_vs_uniform": 0.0,
        "lift_vs_random_feasible": 0.0 if chance_feasible is not None else None,
        "violation_counts": {key: _as_count(value) for key, value in violation.items()},
    }
    if total <= 0:
        return empty

    scores = None
    if n_feasible_designs:
        scores, _low, _high = encoding.normalized_scores(ordered, scale="feasible_minmax")

    feasible_weight = 0.0
    weighted_score = 0.0
    best_sample = None
    for design, weight in zip(ordered, weights):
        if weight <= 0 or not design["feasible"]:
            continue
        feasible_weight += weight
        weighted_score += weight * scores[design["index"]]
        rank = (design["efficiency"], -design["cost"], -design["index"])
        if best_sample is None or rank > best_sample[0]:
            best_sample = (rank, design)

    p_best = 0.0
    if reference["best_index"] is not None:
        p_best = weights[reference["best_index"]] / total
    optimum_mass = sum(weights[index] for index in reference["optimum_indices"]) / total
    top_k_hit = {
        key: sum(weights[index] for index in indices) / total
        for key, indices in reference["top_k_indices"].items()
    }
    summary = {
        "n_samples": _as_count(total),
        "n_feasible_samples": _as_count(feasible_weight),
        "n_feasible_designs": n_feasible_designs,
        "feasible_fraction": feasible_weight / total,
        "unsuccessful": feasible_weight <= 0,
        "p_best": p_best,
        "optimum_hit_rate": optimum_mass,
        "top_k_hit_rate": top_k_hit,
        "chance_p_best_uniform": chance_uniform,
        "chance_p_best_feasible": chance_feasible,
        "lift_vs_uniform": p_best / chance_uniform,
        "lift_vs_random_feasible": None if chance_feasible is None else p_best / chance_feasible,
        "violation_counts": {key: _as_count(value) for key, value in violation.items()},
    }
    if feasible_weight <= 0 or best_sample is None:
        summary.update({
            "best_feasible_bitstring": None,
            "best_feasible_index": None,
            "best_feasible_material": None,
            "best_feasible_geometry": None,
            "best_feasible_passivation": None,
            "best_feasible_efficiency": None,
            "best_feasible_score": None,
            "mean_feasible_score": None,
            "objective_gap": None,
            "efficiency_gap": None,
            "efficiency_gap_percentage_points": None,
        })
        return summary

    chosen = best_sample[1]
    best_score = float(scores[chosen["index"]])
    efficiency_gap = reference["best_efficiency"] - chosen["efficiency"]
    summary.update({
        "best_feasible_bitstring": chosen["bitstring"],
        "best_feasible_index": chosen["index"],
        "best_feasible_material": chosen.get("material"),
        "best_feasible_geometry": chosen.get("geometry"),
        "best_feasible_passivation": chosen.get("passivation"),
        "best_feasible_efficiency": chosen["efficiency"],
        "best_feasible_score": best_score,
        "mean_feasible_score": weighted_score / feasible_weight,
        "objective_gap": reference["best_score"] - best_score,
        "efficiency_gap": efficiency_gap,
        "efficiency_gap_percentage_points": 100.0 * efficiency_gap,
    })
    return summary


def prefix_curve(samples, designs, prefixes=DEFAULT_PREFIXES, reference=None, top_k=DEFAULT_TOP_K):
    """Metrics on sample prefixes. The input order is the draw order.

    Prefixes longer than the sample list are omitted. A dict of counts has no
    draw order, so this function requires a sequence.
    """
    if isinstance(samples, dict) or hasattr(samples, "shape"):
        raise TypeError("prefix_curve needs an ordered sample sequence")
    ordered_samples = list(samples)
    reference = reference or exact_reference(designs, top_k=top_k)
    rows = []
    for prefix in prefixes:
        length = int(prefix)
        if length < 1 or length > len(ordered_samples):
            continue
        row = summarize_samples(ordered_samples[:length], designs, reference=reference)
        row["prefix"] = length
        rows.append(row)
    return rows


def first_optimum_draw(samples, optimum_indices):
    """1-based position of the first sample in `optimum_indices`, or None."""
    wanted = set(optimum_indices)
    for offset, sample in enumerate(samples):
        if isinstance(sample, bool) or not isinstance(sample, numbers.Integral):
            raise TypeError("first_optimum_draw expects integer state indices")
        if int(sample) in wanted:
            return offset + 1
    return None


def counts_from_samples(samples, designs=None):
    """Shot counts keyed by bitstring when designs are given, otherwise by index."""
    counts = {}
    labels = None
    if designs is not None:
        labels = {design["index"]: design["bitstring"] for design in designs}
    for sample in samples:
        if isinstance(sample, bool) or not isinstance(sample, numbers.Integral):
            raise TypeError("counts_from_samples expects integer state indices")
        key = labels[int(sample)] if labels is not None else int(sample)
        counts[key] = counts.get(key, 0) + 1
    return counts


def failure_frequency(run_summaries):
    """Fraction of runs whose sample summary is unsuccessful."""
    if not run_summaries:
        raise ValueError("failure_frequency needs at least one run")
    n_unsuccessful = sum(1 for summary in run_summaries if summary["unsuccessful"])
    n_runs = len(run_summaries)
    return {
        "n_runs": n_runs,
        "n_unsuccessful": n_unsuccessful,
        "failure_frequency": n_unsuccessful / n_runs,
    }
