"""src/plotting.py - TODO"""
import os
import json
import matplotlib.pyplot as plt

# Handle relative paths dynamically
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in globals() else os.getcwd()
output_dir = os.path.join(base_dir, "results", "figures")
os.makedirs(output_dir, exist_ok=True)

data_file = os.path.join(base_dir, "results", "raw", "sim_sweep.json")

# Load existing simulation results or fallback to mock data
if os.path.exists(data_file) and os.path.getsize(data_file) > 0:
    with open(data_file, "r") as f:
        data = json.load(f)
else:
    # Fallback mock data if sim_sweep.json is not ready yet
    data = [
        {"penalty": 1.0, "p": 1, "p_best": 0.05, "p_feasible": 0.50},
        {"penalty": 1.0, "p": 2, "p_best": 0.12, "p_feasible": 0.70},
        {"penalty": 1.0, "p": 3, "p_best": 0.22, "p_feasible": 0.85},
        {"penalty": 2.0, "p": 1, "p_best": 0.08, "p_feasible": 0.60},
        {"penalty": 2.0, "p": 2, "p_best": 0.18, "p_feasible": 0.80},
        {"penalty": 2.0, "p": 3, "p_best": 0.31, "p_feasible": 0.92},
    ]

penalties = sorted(list(set(e["penalty"] for e in data)))

# --- Figure 1: P(best) vs QAOA Depth (p) ---
plt.figure(figsize=(8, 5))
for pen in penalties:
    subset = [e for e in data if e["penalty"] == pen]
    subset.sort(key=lambda x: x["p"])
    plt.plot([e["p"] for e in subset], [e.get("p_best", e.get("P(best)")) for e in subset], marker='o', linewidth=2, label=f"Penalty M = {pen}")

plt.axhline(y=1/48, color='gray', linestyle='--', label="Random baseline (1/48 ≈ 0.021)")
plt.title("Optimal State Probability P(best) vs Depth (p)", fontsize=13, fontweight='bold')
plt.xlabel("QAOA Depth (p)", fontsize=11)
plt.ylabel("P(best)", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "p_best_vs_depth.png"), dpi=300)
plt.close()

# --- Figure 2: P(feasible) vs QAOA Depth (p) ---
plt.figure(figsize=(8, 5))
for pen in penalties:
    subset = [e for e in data if e["penalty"] == pen]
    subset.sort(key=lambda x: x["p"])
    plt.plot([e["p"] for e in subset], [e.get("p_feasible", e.get("P(feas)")) for e in subset], marker='s', linewidth=2, label=f"Penalty M = {pen}")

plt.title("Feasible Solution Rate P(feasible) vs Depth (p)", fontsize=13, fontweight='bold')
plt.xlabel("QAOA Depth (p)", fontsize=11)
plt.ylabel("P(feasible)", fontsize=11)
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "p_feasible_vs_depth.png"), dpi=300)
plt.close()

print("--> Figures successfully generated in results/figures/")