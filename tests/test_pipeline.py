"""tests/test_pipeline.py - end-to-end checks for the physics -> QAOA pipeline.

Run from the repo root (venv active):    pytest -q tests/test_pipeline.py
All tests passing means: data is complete, the quantum cost operator matches the physics
scores, the Qiskit circuit matches the fast simulator, bit order is correct end to end
(through Aer), and QAOA actually learns something. Takes about 10-20 seconds.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
from hamiltonian import (load_scores, design_labels, cost_hamiltonian,  # noqa: E402
                         truncate_hamiltonian)
from qaoa import build_circuit, qaoa_probs, optimise, summarise  # noqa: E402

SCORES = ROOT / "data" / "scores.json"
CATALOGUE = ROOT / "data" / "catalogue.json"
BUDGET = getattr(config, "COST_BUDGET", None)

pytestmark = pytest.mark.skipif(not SCORES.exists(),
                                reason="data/scores.json missing - run: python src/physics.py")


def _rows():
    raw = json.load(open(SCORES))
    return raw["designs"] if "designs" in raw else raw


@pytest.fixture(scope="module")
def data():
    score, feas = load_scores(budget=BUDGET)
    return score, feas


# ------------------------------------------------------------------ data
def test_table_has_all_64_six_bit_designs():
    rows = _rows()
    assert len(rows) == 64
    assert all(len(k) == 6 and set(k) <= {"0", "1"} for k in rows)


def test_some_but_not_all_designs_feasible(data):
    _, feas = data
    assert 0 < feas.sum() < 64


def test_loader_feasibility_matches_json():
    rows = _rows()
    if "efficiency" not in next(iter(rows.values())):
        pytest.skip("placeholder format has no cost column")
    _, feas = load_scores(budget=BUDGET)
    for key, r in rows.items():
        expected = bool(r["feasible"]) and (BUDGET is None or r["cost"] <= BUDGET + 1e-9)
        assert feas[int(key, 2)] == expected, key


def test_exact_best_matches_independent_search(data):
    rows = _rows()
    if "efficiency" not in next(iter(rows.values())):
        pytest.skip("placeholder format")
    score, feas = data
    ok = {k: r["efficiency"] for k, r in rows.items()
          if r["feasible"] and (BUDGET is None or r["cost"] <= BUDGET + 1e-9)}
    best_independent = max(ok, key=ok.get)
    H, cost = cost_hamiltonian(score, feas, penalty=5.0)
    assert format(int(np.argmin(cost)), "06b") == best_independent


def test_bit_layout_matches_catalogue():
    """'000111' must decode to material 0, geometry 1, passivation 3 as the catalogue says."""
    if not CATALOGUE.exists():
        pytest.skip("data/catalogue.json missing")
    cat = json.load(open(CATALOGUE))
    labels = design_labels()
    expect = "%s / %s / %s" % (cat["materials"][0]["name"], cat["geometries"][1]["name"],
                               cat["passivations"][3]["name"])
    assert labels[int("000111", 2)] == expect


def test_scores_json_is_up_to_date_with_physics():
    """Catches a stale scores.json: regenerates from physics.py and compares."""
    if not (ROOT / "data" / "am15g.csv").exists() or not CATALOGUE.exists():
        pytest.skip("needs data/am15g.csv and data/catalogue.json")
    if "efficiency" not in next(iter(_rows().values())):
        pytest.skip("placeholder format")
    import physics
    fresh = physics.build_scores()["designs"]
    for key, r in _rows().items():
        assert abs(fresh[key]["efficiency"] - r["efficiency"]) < 1e-5, key
        assert abs(fresh[key]["cost"] - r["cost"]) < 1e-3, key


# ------------------------------------------------------------------ quantum side
@pytest.mark.parametrize("penalty", [0.25, 5.0])
def test_hamiltonian_reproduces_cost_table(data, penalty):
    score, feas = data
    H, cost = cost_hamiltonian(score, feas, penalty)
    diag = np.real(np.diag(H.to_matrix()))
    assert np.abs(diag - cost).max() < 1e-9


def test_qiskit_circuit_matches_fast_simulator(data):
    from qiskit.quantum_info import Statevector
    score, feas = data
    H, cost = cost_hamiltonian(score, feas, 5.0)
    g, b = [0.7, 1.3], [0.4, 0.9]
    p_circuit = Statevector(build_circuit(H, g, b, measure=False)).probabilities()
    assert np.abs(p_circuit - qaoa_probs(cost, g, b)).max() < 1e-9


def test_aer_counts_use_same_bit_order_as_scores(data):
    """End-to-end bit-order check: sample the circuit on Aer and compare to the exact
    distribution looked up with int(counts_key, 2) - the same lookup scores.json uses."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2
    score, feas = data
    H, cost = cost_hamiltonian(score, feas, 5.0)
    g, b = [0.7, 1.3], [0.4, 0.9]
    exact = qaoa_probs(cost, g, b)
    qc = transpile(build_circuit(H, g, b, measure=True), AerSimulator(), optimization_level=1)
    shots = 40000
    counts = SamplerV2(seed=1).run([qc], shots=shots).result()[0].data.meas.get_counts()
    emp = np.zeros(64)
    for k, v in counts.items():
        assert len(k) == 6
        emp[int(k, 2)] = v / shots
    assert 0.5 * np.abs(emp - exact).sum() < 0.04          # total-variation distance


def test_qaoa_beats_uniform_guessing(data):
    score, feas = data
    H, cost = cost_hamiltonian(score, feas, 1.0)
    opt = optimise(cost, 1, n_restarts=5, maxiter=150, seed=0)
    assert opt["value"] < cost.mean() - 0.02               # lower average cost than random bitstrings


def test_extra_layer_never_hurts_with_parameter_reuse(data):
    score, feas = data
    H, cost = cost_hamiltonian(score, feas, 1.0)
    o1 = optimise(cost, 1, n_restarts=3, maxiter=150, seed=0)
    init = o1["gammas"] + [0.0] + o1["betas"] + [0.0]
    o2 = optimise(cost, 2, n_restarts=1, maxiter=150, seed=0, init=init)
    assert o2["value"] <= o1["value"] + 1e-9


def test_summarise_on_uniform_distribution(data):
    score, feas = data
    probs = np.full(64, 1 / 64)
    m = summarise(probs, score, feas)
    assert m["p_best"] == pytest.approx(1 / 64)
    assert m["p_feasible"] == pytest.approx(feas.sum() / 64)
    assert m["lift_vs_random_feasible"] == pytest.approx(feas.sum() / 64)


def test_truncation_removes_high_order_terms(data):
    score, feas = data
    H, _ = cost_hamiltonian(score, feas, 1.0)
    Ht = truncate_hamiltonian(H, 2)
    assert len(Ht) < len(H)
    assert all(lab.count("Z") <= 2 for lab in Ht.paulis.to_labels())
