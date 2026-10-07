"""Classical encoding, baseline and metric checks.

Run from the repository root:

    python -m pytest tests/test_encoding.py -q
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import config  # noqa: E402
import physics  # noqa: E402
from baselines import (  # noqa: E402
    BASELINE_SEEDS,
    enumerate_best,
    exact_enumeration,
    greedy_absorption_first,
    greedy_coordinate,
    random_feasible,
)
from encoding import (  # noqa: E402
    N_STATES,
    audit_penalty,
    bitstring_to_index,
    build_cost_table,
    build_unit_cost_table,
    decode,
    encode,
    index_to_bitstring,
    load_designs,
    penalty_threshold,
    qubit_bits,
)
from metrics import (  # noqa: E402
    exact_reference,
    failure_frequency,
    first_optimum_draw,
    prefix_curve,
    summarize_samples,
)

SCORES = ROOT / "data" / "scores.json"
BUDGET = 51.0


def _raw():
    with open(SCORES, encoding="utf-8") as handle:
        return json.load(handle)["designs"]


def _independent_best(raw, budget):
    best_key = None
    best_rank = None
    for key, row in raw.items():
        within = True if budget is None else row["cost"] <= budget + 1e-9
        if row["feasible"] and within:
            rank = (row["efficiency"], -row["cost"], -int(key, 2))
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_key = key
    return best_key


def _toy_designs():
    specs = [
        (0.50, 1.0, True, True),
        (1.00, 2.0, True, True),
        (0.80, 3.0, False, True),
        (0.00, 4.0, True, False),
    ]
    designs = []
    for index, (efficiency, cost, compatible, within_budget) in enumerate(specs):
        designs.append({
            "index": index,
            "bitstring": format(index, "02b"),
            "material_id": 0,
            "geometry_id": 0,
            "layer_id": index,
            "material": "m",
            "geometry": "g",
            "passivation": "p",
            "efficiency": efficiency,
            "cost": cost,
            "compatible": compatible,
            "within_budget": within_budget,
            "feasible": compatible and within_budget,
            "category_valid": True,
        })
    return designs


def _grid(efficiency_of, cost_of, feasible_of):
    designs = []
    for index in range(N_STATES):
        material_id, geometry_id, layer_id, valid = decode(index)
        feasible = bool(feasible_of(material_id, geometry_id, layer_id))
        designs.append({
            "index": index,
            "bitstring": index_to_bitstring(index),
            "material_id": material_id,
            "geometry_id": geometry_id,
            "layer_id": layer_id,
            "material": str(material_id),
            "geometry": str(geometry_id),
            "passivation": str(layer_id),
            "efficiency": float(efficiency_of(material_id, geometry_id, layer_id)),
            "cost": float(cost_of(material_id, geometry_id, layer_id)),
            "compatible": True,
            "within_budget": feasible,
            "feasible": feasible,
            "category_valid": valid,
        })
    return designs


@pytest.fixture(scope="module")
def designs():
    if not SCORES.exists():
        pytest.skip("data/scores.json is missing")
    return load_designs(budget=BUDGET)


def _reconstruct_diagonal(energy):
    size = len(energy)
    coefficients = []
    for mask in range(size):
        total = 0.0
        for state in range(size):
            parity = (mask & state).bit_count() & 1
            total += energy[state] * (1 - 2 * parity)
        coefficients.append(total / size)
    diagonal = []
    for state in range(size):
        total = 0.0
        for mask in range(size):
            parity = (mask & state).bit_count() & 1
            total += coefficients[mask] * (1 - 2 * parity)
        diagonal.append(total)
    return diagonal


# ---------------------------------------------------------------- encoding
def test_roundtrip_covers_every_state():
    seen = set()
    for material_id, geometry_id, layer_id in (
        (m, g, p) for m in range(4) for g in range(4) for p in range(4)
    ):
        index = encode(material_id, geometry_id, layer_id)
        assert decode(index) == (material_id, geometry_id, layer_id, True)
        text = index_to_bitstring(index)
        assert text == f"{material_id:02b}{geometry_id:02b}{layer_id:02b}"
        assert bitstring_to_index(text) == index == int(text, 2)
        assert physics.parse_design(text) == (material_id, geometry_id, layer_id)
        assert physics.design_to_bitstring(material_id, geometry_id, layer_id) == text
        seen.add(index)
    assert seen == set(range(N_STATES))


def test_catalogue_example_and_qiskit_bit_order():
    assert decode(bitstring_to_index("000111")) == (0, 1, 3, True)
    assert qubit_bits(7) == [1, 1, 1, 0, 0, 0]
    text = index_to_bitstring(7)
    bits = qubit_bits(7)
    assert text[0] == str(bits[5])
    assert text[-1] == str(bits[0])
    assert qubit_bits(0b100000) == [0, 0, 0, 0, 0, 1]
    assert decode(0b100000)[0] == 2


def test_out_of_range_and_bad_types():
    assert decode(-1) == (None, None, None, False)
    assert decode(64) == (None, None, None, False)
    with pytest.raises(TypeError):
        decode(True)
    with pytest.raises(ValueError):
        encode(0, 0, 4)
    with pytest.raises(ValueError):
        bitstring_to_index("00011")
    with pytest.raises(ValueError):
        index_to_bitstring(64)


def test_frozen_catalogue_uses_all_64_codes(designs):
    assert len(designs) == 64
    assert all(design["category_valid"] for design in designs)
    assert sum(design["feasible"] for design in designs) == 15
    assert sum(design["compatible"] for design in designs) == 48
    assert sum(not design["within_budget"] for design in designs) == 49


# ---------------------------------------------------------------- feasibility
def test_feasibility_matches_the_score_file_without_using_the_cost_table(designs):
    raw = _raw()
    by_bit = {design["bitstring"]: design for design in designs}
    for key, row in raw.items():
        design = by_bit[key]
        compatible = bool(row["feasible"])
        within = row["cost"] <= BUDGET + 1e-9
        assert design["compatible"] is compatible
        assert design["within_budget"] is within
        assert design["feasible"] is (compatible and within)
    assert by_bit["000111"]["feasible"] is True
    assert by_bit["001111"]["compatible"] is True
    assert by_bit["001111"]["within_budget"] is False
    assert by_bit["001111"]["feasible"] is False
    assert raw["110001"]["feasible"] is False
    assert by_bit["110001"]["compatible"] is False
    assert by_bit["110001"]["efficiency"] == 0.0


def test_penalty_does_not_change_which_designs_are_feasible(designs):
    flags = [design["feasible"] for design in designs]
    for penalty in (0.0, 0.25, 2.0):
        build_cost_table(designs, penalty)
        assert [design["feasible"] for design in designs] == flags


def test_exact_optimum_matches_an_independent_scan(designs):
    raw = _raw()
    assert _independent_best(raw, BUDGET) == "000111"
    found = exact_enumeration(designs)
    assert found["design"]["bitstring"] == "000111"
    assert found["design"]["efficiency"] == raw["000111"]["efficiency"]
    assert found["design"]["cost"] == raw["000111"]["cost"]
    assert found["n_queries"] == 64
    assert found["n_feasible"] == 15
    open_budget = load_designs(budget=None)
    assert exact_enumeration(open_budget)["design"]["bitstring"] == _independent_best(raw, None)
    assert _independent_best(raw, None) == "001111"


def test_experiment_budget_matches_config(designs):
    assert config.COST_BUDGET == BUDGET
    configured = load_designs(budget=config.COST_BUDGET)
    assert [design["feasible"] for design in configured] == [design["feasible"] for design in designs]


# ---------------------------------------------------------------- cost table
def test_cost_table_matches_independent_arithmetic(designs):
    feasible_eff = [design["efficiency"] for design in designs if design["feasible"]]
    low, high = min(feasible_eff), max(feasible_eff)
    penalty = 2.0
    manual = []
    for design in designs:
        score = (design["efficiency"] - low) / (high - low)
        manual.append(-score + (0.0 if design["feasible"] else penalty))
    table = build_cost_table(designs, penalty)
    assert max(abs(left - right) for left, right in zip(manual, table["energy"])) < 1e-12
    feasible_energy = min(table["energy"][index] for index, design in enumerate(designs) if design["feasible"])
    assert feasible_energy == pytest.approx(-1.0)


def test_walsh_hadamard_reconstructs_the_diagonal(designs):
    energy = build_cost_table(designs, 2.0)["energy"]
    rebuilt = _reconstruct_diagonal(energy)
    assert max(abs(left - right) for left, right in zip(energy, rebuilt)) < 1e-10


def test_unit_interval_penalty_theorem(designs):
    unit = build_unit_cost_table(designs, 2.0)
    assert unit["normalized_score"].min() >= -1e-12
    assert unit["normalized_score"].max() <= 1.0 + 1e-12
    strong = audit_penalty(designs, 2.0, scale="catalogue_max")
    assert strong["all_infeasible_worse"] is True
    assert strong["matches_physical_optimum"] is True
    assert strong["physical_optimum_bitstring"] == "000111"
    weak = audit_penalty(designs, 0.0, scale="catalogue_max")
    assert weak["global_minimizer_bitstring"] == "001111"
    assert weak["matches_physical_optimum"] is False
    assert weak["all_infeasible_worse"] is False


def test_training_scale_penalty_threshold(designs):
    threshold = penalty_threshold(designs, scale="feasible_minmax")
    assert 0.0 < threshold < 0.25
    assert audit_penalty(designs, 0.0)["global_minimizer_bitstring"] == "001111"
    assert audit_penalty(designs, 0.25)["matches_physical_optimum"] is True
    assert audit_penalty(designs, 2.0)["all_infeasible_worse"] is True
    assert audit_penalty(designs, 2.0)["matches_physical_optimum"] is True


def test_cost_table_matches_hamiltonian_module(designs):
    pytest.importorskip("qiskit")
    from hamiltonian import cost_vector, load_scores

    score, feasible = load_scores(budget=BUDGET)
    for penalty in (0.25, 2.0, 5.0):
        table = build_cost_table(designs, penalty)
        other = cost_vector(score, feasible, penalty)
        assert max(abs(left - right) for left, right in zip(table["energy"], other)) < 1e-10


# ---------------------------------------------------------------- baselines
def test_absorption_first_is_feasible_and_suboptimal(designs):
    result = greedy_absorption_first(designs)
    assert result["design"]["bitstring"] == "001100"
    assert result["design"]["feasible"] is True
    assert result["n_queries"] == 24
    assert result["used_fallback"] is False
    assert result["n_queries"] < 64
    assert result["design"]["efficiency"] < exact_enumeration(designs)["design"]["efficiency"]
    assert result["search_order"]["materials"][:2] == ["Perovskite", "c-Si"]
    assert result["search_order"]["geometries"][0] == "Nanowire array"
    assert result["search_order"]["passivations"][-1] == "None"


def test_coordinate_greedy_is_feasible_and_suboptimal(designs):
    result = greedy_coordinate(designs)
    assert result["design"]["bitstring"] == "010011"
    assert result["design"]["feasible"] is True
    assert result["n_queries"] == 13
    assert [step["bitstring"] for step in result["trace"]] == [
        "000000",
        "010000",
        "010000",
        "010011",
    ]
    optimum = exact_enumeration(designs)["design"]["efficiency"]
    assert result["design"]["efficiency"] < optimum
    assert result["design"]["efficiency"] > greedy_absorption_first(designs)["design"]["efficiency"]


def test_coordinate_greedy_tie_breaks_toward_lower_cost():
    designs = _grid(
        lambda material, geometry, layer: 0.5 if layer in (1, 2) else 0.1,
        lambda material, geometry, layer: 5.0 if layer == 1 else 1.0,
        lambda material, geometry, layer: True,
    )
    result = greedy_coordinate(designs)
    assert (result["design"]["material_id"], result["design"]["geometry_id"], result["design"]["layer_id"]) == (0, 0, 2)


def test_random_feasible_is_reproducible_and_stays_feasible(designs):
    assert BASELINE_SEEDS == (11, 22, 33)
    first = random_feasible(designs, n_draws=4096, seed=11)
    second = random_feasible(designs, n_draws=4096, seed=11)
    third = random_feasible(designs, n_draws=4096, seed=22)
    assert first["samples"] == second["samples"]
    assert first["samples"] != third["samples"]
    assert first["pool_size"] == 15
    assert first["n_pool_queries"] == 64
    assert len(first["samples"]) == 4096
    assert all(designs[sample]["feasible"] for sample in first["samples"])
    assert len(set(first["samples"])) < 4096
    reference = exact_reference(designs)
    summary = summarize_samples(first["samples"], designs, reference=reference)
    assert summary["feasible_fraction"] == 1.0
    assert summary["p_best"] == pytest.approx(1.0 / 15, abs=0.02)
    assert summary["top_k_hit_rate"]["3"] == pytest.approx(3.0 / 15, abs=0.02)
    assert summary["objective_gap"] == pytest.approx(0.0)
    assert summary["unsuccessful"] is False
    curve = prefix_curve(first["samples"], designs, reference=reference)
    assert curve[-1]["prefix"] == 4096
    assert curve[-1]["p_best"] == pytest.approx(summary["p_best"])
    assert first_optimum_draw(first["samples"], reference["optimum_indices"]) == next(
        offset + 1 for offset, sample in enumerate(first["samples"]) if sample == reference["best_index"]
    )


def test_enumerate_best_ignores_infeasible_and_keeps_the_lower_index():
    assert enumerate_best([0.2, 0.4, 0.4], [True, True, True]) == (1, 0.4)
    assert enumerate_best([0.2, 0.9], [True, False]) == (0, 0.2)
    assert enumerate_best([0.4, 0.4], [False, False]) == (None, None)


def test_synthetic_scan_finds_the_planted_optimum():
    from scaling_study import synthetic_instance

    scores, feasible, planted = synthetic_instance(6, seed=2026)
    assert enumerate_best(scores, feasible)[0] == planted == 21


# ---------------------------------------------------------------- metrics
def test_metrics_on_a_hand_calculated_count_table():
    designs = _toy_designs()
    summary = summarize_samples({0: 1, 1: 2, 2: 1}, designs)
    assert summary["n_samples"] == 4
    assert summary["n_feasible_samples"] == 3
    assert summary["feasible_fraction"] == pytest.approx(0.75)
    assert summary["unsuccessful"] is False
    assert summary["best_feasible_bitstring"] == "01"
    assert summary["best_feasible_score"] == pytest.approx(1.0)
    assert summary["mean_feasible_score"] == pytest.approx(2.0 / 3.0)
    assert summary["objective_gap"] == pytest.approx(0.0)
    assert summary["efficiency_gap"] == pytest.approx(0.0)
    assert summary["p_best"] == pytest.approx(0.5)
    assert summary["optimum_hit_rate"] == pytest.approx(0.5)
    assert summary["top_k_hit_rate"]["1"] == pytest.approx(0.5)
    assert summary["top_k_hit_rate"]["3"] == pytest.approx(0.75)
    assert summary["chance_p_best_uniform"] == pytest.approx(0.25)
    assert summary["chance_p_best_feasible"] == pytest.approx(0.5)
    assert summary["lift_vs_uniform"] == pytest.approx(2.0)
    assert summary["lift_vs_random_feasible"] == pytest.approx(1.0)
    assert summary["violation_counts"]["infeasible"] == 1
    assert summary["violation_counts"]["incompatible"] == 1
    assert summary["violation_counts"]["over_budget"] == 0

    missed = summarize_samples([0, 0, 0, 2], designs)
    assert missed["best_feasible_score"] == pytest.approx(0.0)
    assert missed["mean_feasible_score"] == pytest.approx(0.0)
    assert missed["objective_gap"] == pytest.approx(1.0)
    assert missed["efficiency_gap"] == pytest.approx(0.5)
    assert missed["efficiency_gap_percentage_points"] == pytest.approx(50.0)
    assert missed["p_best"] == 0.0
    assert missed["top_k_hit_rate"]["1"] == 0.0
    assert missed["top_k_hit_rate"]["3"] == pytest.approx(0.75)


def test_zero_feasible_samples_leave_gaps_undefined():
    designs = _toy_designs()
    summary = summarize_samples({2: 5, 3: 5}, designs)
    assert summary["feasible_fraction"] == 0.0
    assert summary["unsuccessful"] is True
    assert summary["best_feasible_score"] is None
    assert summary["mean_feasible_score"] is None
    assert summary["objective_gap"] is None
    assert summary["efficiency_gap"] is None
    assert summary["efficiency_gap_percentage_points"] is None
    assert summary["p_best"] == 0.0
    assert summary["optimum_hit_rate"] == 0.0
    assert summary["violation_counts"] == {
        "infeasible": 10,
        "incompatible": 5,
        "over_budget": 5,
        "incompatible_and_over_budget": 0,
        "invalid_category": 0,
    }


def test_prefix_curve_preserves_sample_order():
    designs = _toy_designs()
    rows = prefix_curve([1, 0, 1], designs, prefixes=(1, 2, 3))
    assert [row["prefix"] for row in rows] == [1, 2, 3]
    assert rows[0]["p_best"] == pytest.approx(1.0)
    assert rows[1]["p_best"] == pytest.approx(0.5)
    assert rows[2]["p_best"] == pytest.approx(2.0 / 3.0)
    assert first_optimum_draw([1, 0, 1], [1]) == 1
    assert first_optimum_draw([0, 0], [1]) is None


def test_failure_frequency_counts_unsuccessful_runs():
    report = failure_frequency([
        {"unsuccessful": False},
        {"unsuccessful": True},
        {"unsuccessful": False},
    ])
    assert report == {"n_runs": 3, "n_unsuccessful": 1, "failure_frequency": pytest.approx(1 / 3)}
    with pytest.raises(ValueError):
        failure_frequency([])
