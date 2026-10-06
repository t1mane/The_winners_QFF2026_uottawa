"""qaoa.py - QAOA circuit, fast exact simulator, optimiser and metrics.

Two views of the SAME algorithm:
  * build_circuit(...)  -> a Qiskit circuit (for Aer / noisy sim / real IBM hardware)
  * qaoa_probs(...)     -> a tiny numpy statevector simulator (fast, used to tune parameters)
tests/test code checks they agree, so tune with numpy, then run the circuit on hardware.

QAOA depth p:  |psi> = prod_{l=1..p} [ exp(-i beta_l * sum_k X_k) exp(-i gamma_l * C) ] H^n |0>
"""
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate


# ---------------------------------------------------------------- Qiskit circuit
def build_circuit(H, gammas, betas, measure=True):
    """QAOA circuit for cost operator H (SparsePauliOp, all-Z terms so Trotter is exact)."""
    n = H.num_qubits
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for g, b in zip(gammas, betas):
        qc.append(PauliEvolutionGate(H, time=float(g)), range(n))  # exp(-i g H)
        for q in range(n):
            qc.rx(2.0 * float(b), q)                               # exp(-i b X)
    if measure:
        qc.measure_all()
    return qc


# ---------------------------------------------------------------- fast numpy simulator
def qaoa_state(cost, gammas, betas):
    n = int(round(np.log2(len(cost))))
    psi = np.full(len(cost), 1.0 / np.sqrt(len(cost)), dtype=complex)
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * cost)
        c, s = np.cos(b), -1j * np.sin(b)
        rx = np.array([[c, s], [s, c]])
        psi = psi.reshape((2,) * n)
        for axis in range(n):
            psi = np.moveaxis(np.tensordot(rx, psi, axes=([1], [axis])), 0, axis)
        psi = psi.reshape(-1)
    return psi


def qaoa_probs(cost, gammas, betas):
    return np.abs(qaoa_state(cost, gammas, betas)) ** 2


def expectation(cost, gammas, betas):
    return float(np.dot(qaoa_probs(cost, gammas, betas), cost))


# ---------------------------------------------------------------- optimiser
def _split(x):
    p = len(x) // 2
    return x[:p], x[p:]


def linear_ramp(p, gamma_max=1.0, beta_max=0.8):
    """Standard good starting point: gamma ramps up, beta ramps down across layers."""
    t = (np.arange(p) + 0.5) / p
    return np.concatenate([gamma_max * t, beta_max * (1 - t)])


def optimise(cost, p, n_restarts=20, seed=42, maxiter=300, init=None):
    """Tune (gammas, betas) to minimise <C>. Returns dict with best params and every restart.

    Restart 0 is the linear ramp (or `init` if given, e.g. layer p-1 params for reuse);
    the rest are random. COBYLA is gradient-free, which is what you would also use
    on noisy hardware.
    """
    rng = np.random.default_rng(seed)
    starts = [linear_ramp(p) if init is None else np.asarray(init, dtype=float)]
    for _ in range(n_restarts - 1):
        starts.append(np.concatenate([rng.uniform(0, 2 * np.pi, p), rng.uniform(0, np.pi, p)]))

    runs = []
    for x0 in starts:
        trace = []

        def f(x):
            val = expectation(cost, *_split(x))
            trace.append(val)
            return val

        res = minimize(f, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.5})
        runs.append({"x": res.x.tolist(), "value": float(res.fun), "n_evals": len(trace),
                     "trace": [float(t) for t in trace]})
    best = min(runs, key=lambda r: r["value"])
    g, b = _split(np.array(best["x"]))
    return {"p": p, "gammas": g.tolist(), "betas": b.tolist(), "value": best["value"],
            "restart_values": [r["value"] for r in runs], "best_trace": best["trace"],
            "total_evals": int(sum(r["n_evals"] for r in runs))}


# ---------------------------------------------------------------- metrics
def summarise(probs, score, feasible):
    """Quality + feasibility of the QAOA output distribution (exact, shot-noise free)."""
    s = (score - score[feasible].min()) / (score[feasible].max() - score[feasible].min())
    best = int(np.argmax(np.where(feasible, s, -np.inf)))
    p_feas = float(probs[feasible].sum())
    n_feas = int(feasible.sum())
    order = np.argsort(-np.where(feasible, s, -np.inf))
    top3 = order[:3]
    return {
        "best_design": format(best, "0%db" % int(np.log2(len(probs)))),
        "p_best": float(probs[best]),
        "p_feasible": p_feas,
        "p_top3": float(probs[top3].sum()),
        "mean_score_given_feasible": float(np.dot(probs[feasible], s[feasible]) / p_feas) if p_feas > 0 else 0.0,
        "random_p_best": 1.0 / n_feas,                       # uniform over feasible designs
        "uniform_p_best": 1.0 / len(probs),                  # uniform over all bitstrings
        "lift_vs_random_feasible": float(probs[best] * n_feas),
    }


def sample_counts(probs, shots, seed=0):
    """Draw shot-limited counts from a probability vector (mimics a finite-shot run)."""
    rng = np.random.default_rng(seed)
    n = int(round(np.log2(len(probs))))
    draws = rng.choice(len(probs), size=shots, p=probs / probs.sum())
    counts = {}
    for d in draws:
        key = format(int(d), "0%db" % n)
        counts[key] = counts.get(key, 0) + 1
    return counts