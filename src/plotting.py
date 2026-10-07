"""plotting.py - figures from the QAOA sweep results.

Run from the repo root:   python src/plotting.py
Reads   results/raw/sim_sweep.json        (exact problem; written by experiments/run_sim_sweep.py)
        results/raw/sim_sweep_w2.json     (optional: pairwise-approximation circuit)
Writes  results/figures/p_best_vs_depth.png
        results/figures/p_feasible_vs_depth.png
        results/figures/p_best_exact_vs_pairwise.png   (only if the w2 file exists)
        results/figures/sweep_table.csv                (the same numbers as a table)

The random baselines are read from the results, never hard-coded:
  P(best)     random = 1 / (number of feasible designs)
  P(feasible) random = (number of feasible designs) / 64   (a uniformly random 6-bit string)
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
FIG = ROOT / "results" / "figures"

# Validated categorical palette, fixed order (one colour per penalty value, ascending).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK2, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dc", "#8a8983"


def load(name):
    path = RAW / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def by_penalty(entries):
    """{penalty: [(depth, metrics), ...]} sorted by depth."""
    out = {}
    for e in entries:
        out.setdefault(e["penalty"], []).append((e["p"], e["metrics"]))
    return {k: sorted(v, key=lambda t: t[0]) for k, v in sorted(out.items())}


def n_feasible(entries):
    return round(1.0 / entries[0]["metrics"]["random_p_best"])


def spread(ys, gap):
    """Nudge label heights apart so direct labels do not overlap."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def style(ax, ylabel, depths):
    ax.set_facecolor(SURFACE)
    ax.set_xlabel("QAOA depth p (number of layers)", color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_xticks(depths)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def line_chart(groups, key, baseline, baseline_label, title, subtitle, ylabel, outfile, ymax=None):
    depths = sorted({d for rows in groups.values() for d, _ in rows})
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax, ylabel, depths)
    ax.axhline(baseline, color=BASE, linestyle=(0, (5, 4)), linewidth=1.6, label=baseline_label, zorder=1)

    ends = []
    for i, (pen, rows) in enumerate(groups.items()):
        color = SERIES[i % len(SERIES)]
        xs = [d for d, _ in rows]
        ys = [m[key] for _, m in rows]
        ax.plot(xs, ys, color=color, linewidth=2.2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label="penalty = %g" % pen, zorder=3)
        ends.append((xs[-1], ys[-1], color, pen))

    top = ymax if ymax else max(m[key] for rows in groups.values() for _, m in rows) * 1.18
    ax.set_ylim(0, top)
    ax.set_xlim(min(depths) - 0.25, max(depths) + 0.85)
    ys_lab = spread([e[1] for e in ends], top * 0.06)
    for (x, y, color, pen), yl in zip(ends, ys_lab):
        ax.annotate("%g" % pen, xy=(x, y), xytext=(x + 0.12, yl), color=INK, fontsize=9,
                    va="center", arrowprops=dict(arrowstyle="-", color=color, lw=0.9, shrinkA=4, shrinkB=2))

    ax.set_title(title, loc="left", color=INK, fontsize=12.5, fontweight="bold", pad=22)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left", ncol=1)
    fig.tight_layout()
    fig.savefig(outfile, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", outfile.relative_to(ROOT))


def compare_chart(exact, pairwise, outfile, n_feas):
    """Exact vs pairwise-approximation circuit at the same penalty values (P(best))."""
    depths = sorted({d for rows in exact.values() for d, _ in rows})
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax, "Probability of sampling the best design", depths)
    ax.axhline(1.0 / n_feas, color=BASE, linestyle=(0, (5, 4)), linewidth=1.6,
               label="random guess (1/%d = %.1f%%)" % (n_feas, 100.0 / n_feas), zorder=1)
    shared = [p for p in exact if p in pairwise][:3]
    for i, pen in enumerate(shared):
        color = SERIES[i]
        for rows, ls, mk, tag in ((exact[pen], "-", "o", "exact"), (pairwise[pen], (0, (3, 2)), "s", "pairwise")):
            ax.plot([d for d, _ in rows], [m["p_best"] for _, m in rows], color=color, linestyle=ls,
                    linewidth=2.2, marker=mk, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label="penalty %g, %s" % (pen, tag), zorder=3)
    top = max(m["p_best"] for g in (exact, pairwise) for rows in g.values() for _, m in rows) * 1.18
    ax.set_ylim(0, top)
    ax.set_xlim(min(depths) - 0.25, max(depths) + 0.25)
    ax.set_title("Cheaper pairwise circuit vs exact circuit", loc="left", color=INK, fontsize=12.5,
                 fontweight="bold", pad=22)
    ax.text(0, 1.03, "Solid = exact cost function, dashed = only 1- and 2-qubit terms (far fewer gates). "
            "Scored on the true problem.", transform=ax.transAxes, color=INK2, fontsize=8.2)
    ax.legend(frameon=False, fontsize=7.8, labelcolor=INK2, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(outfile, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", outfile.relative_to(ROOT))


def write_table(entries, outfile):
    with open(outfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["penalty", "depth_p", "P_best", "P_feasible", "P_top3", "lift_vs_random"])
        for e in sorted(entries, key=lambda e: (e["penalty"], e["p"])):
            m = e["metrics"]
            w.writerow([e["penalty"], e["p"], round(m["p_best"], 4), round(m["p_feasible"], 4),
                        round(m["p_top3"], 4), round(m["lift_vs_random_feasible"], 2)])
    print("wrote", outfile.relative_to(ROOT))


def main():
    entries = load("sim_sweep.json")
    if entries is None:
        sys.exit("results/raw/sim_sweep.json not found - run: python experiments/run_sim_sweep.py")
    FIG.mkdir(parents=True, exist_ok=True)
    groups = by_penalty(entries)
    nf = n_feasible(entries)
    budget = entries[0].get("budget")
    sub = "%d feasible designs of 64%s. Dashed line = picking a feasible design at random." % (
        nf, "" if budget is None else " (cost budget $%g/m2)" % budget)

    line_chart(groups, "p_best", 1.0 / nf, "random guess (1/%d = %.1f%%)" % (nf, 100.0 / nf),
               "More QAOA layers generally find the best design more often",
               sub, "Probability of sampling the best design", FIG / "p_best_vs_depth.png")
    line_chart(groups, "p_feasible", nf / 64.0, "random 6-bit string (%d/64 = %.0f%%)" % (nf, 100.0 * nf / 64),
               "A moderate penalty keeps QAOA on valid designs",
               "Share of samples that satisfy the compatibility and cost rules. Dashed line = random bit string.",
               "Probability of sampling a feasible design", FIG / "p_feasible_vs_depth.png", ymax=1.08)
    write_table(entries, FIG / "sweep_table.csv")

    w2 = load("sim_sweep_w2.json")
    if w2:
        compare_chart(groups, by_penalty(w2), FIG / "p_best_exact_vs_pairwise.png", nf)


if __name__ == "__main__":
    main()