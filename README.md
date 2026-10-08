 # Quantum Radiophotovoltaic Converter Benchmark

QAOA design-selection benchmark for quantum radiophotovoltaic converter optimization.

## 1. Problem Overview
This project benchmarks the Quantum Approximate Optimization Algorithm (QAOA) for optimal design selection in quantum radiophotovoltaic converter systems under operational and physical constraints.

## 2. Getting Started & Execution

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Check the classical encoding, baselines, and metrics** (about 15 seconds once Qiskit is installed; the Hamiltonian comparison is skipped if Qiskit is absent):
   ```bash
   python -m pytest tests/test_encoding.py -q
   ```
3. **Rebuild the classical reference** (under a second) and the catalogue-size timing (about a second):
   ```bash
   python experiments/run_exact.py
   python experiments/scaling_study.py
   ```
   These write [`results/raw/exact_baselines.json`](results/raw/exact_baselines.json) and [`results/raw/scaling_study.json`](results/raw/scaling_study.json). The budget and the 4,096-draw sample size come from `config.py` (`COST_BUDGET = 51.0` USD/m², `SHOTS = 4096`). Classical sampling seeds are 11, 22, and 33. `config.SEED` (42) is the QAOA optimizer seed and is not used by these draws.
4. **QAOA sweep** (Timane), which scores samples against the same feasible optimum:
   ```bash
   python experiments/run_sim_sweep.py
   ```

The saved classical file was produced with Python 3.9.6 and NumPy 2.0.2 on macOS. The same designs, query counts, and best-design probabilities were reproduced with Python 3.13.1, NumPy 2.5.3, and Qiskit 2.5.2. Data hashes in the results file are SHA-256 of the score and catalogue files with Windows CRLF normalized to LF, so the hash does not change between those checkouts.

## 3. Data and assumptions used by the classical reference

The classical code does not generate material properties. It reads the frozen table [`data/scores.json`](data/scores.json), produced by `src/physics.py` from [`data/catalogue.json`](data/catalogue.json) and the AM1.5G spectrum in `data/am15g.csv`. Provenance, units, and model assumptions for that table are in [`docs/assumptions.md`](docs/assumptions.md) and in the catalogue sources. Regenerating the table, if the inputs are present, is:

```bash
python src/physics.py
```

Each of the 64 rows is one design: 4 absorbers × 4 surface geometries × 4 passivation options, including passivation “None”. Efficiency is the fraction of incident sunlight converted to electricity, averaged over the 25-year service life in the physics model. Cost is USD per square metre of module on the NREL 2020 minimum-sustainable-price basis recorded in the catalogue, not a market price.

At the budget in `config.py`:

| Count | Meaning | Number |
|---|---|---:|
| Categorical designs | Every 6-bit code is a real material / geometry / passivation triple | 64 |
| Process-compatible | The passivation step is within the absorber’s temperature limit. This is the `feasible` field inside `scores.json` | 48 |
| Within budget | Module cost ≤ 51.0 USD/m² | 15 |
| Feasible for optimization | Process-compatible and within budget | 15 |

The 16 process-incompatible designs are also over the budget, so the feasible set is exactly the 15 designs at or under 51.0 USD/m². A design on the budget boundary is kept: c-Si / micro-pyramids / thermal SiO₂ costs 51.0 USD/m² and is feasible.

## 4. Encoding and the cost used for comparison

`src/encoding.py` packs each design into six bits, most significant bit first:

```text
bitstring = material(2) | geometry(2) | passivation(2)
state index s = int(bitstring, 2) = 16*material + 4*geometry + passivation
```

Qubit q holds bit q of s. A Qiskit count string is already in this order: the leftmost character is qubit 5 and the rightmost character is qubit 0. Counts are looked up with `int(key, 2)`. The catalogue example `000111` is material 0 (c-Si), geometry 1 (micro-pyramids), passivation 3 (Al₂O₃ ALD).

Two scores are recorded. Training and the reported objective gap use the same scale as `src/hamiltonian.py`:

```text
r = (efficiency - min feasible efficiency) / (max feasible efficiency - min feasible efficiency)
```

The best feasible design has r = 1. On this table the feasible efficiencies run from 0.117442 to 0.261381. An over-budget design can sit slightly above 1 because the scale is fixed by the feasible rows only. The diagonal energy is

```text
E = -r + M * (1 if the design is infeasible else 0)
```

The identity contribution is kept in E. A second scale, efficiency divided by the maximum efficiency among all 64 designs, puts every r in [0, 1]. It is used only to check the penalty. QAOA training uses the feasible min-max scale.

`tests/test_encoding.py` checks the round trip of all 64 states, agreement with `physics.parse_design`, feasibility read directly from the score file, the Walsh–Hadamard rebuild of the 64 energies to 1e-10, and, when Qiskit is installed, equality with `hamiltonian.cost_vector` to 1e-10.

## 5. Classical methods

`src/baselines.py` implements three references. None of them changes the score table.

**Exact enumeration** reads all 64 rows and maximises lifetime-averaged efficiency among feasible designs. Ties would break toward lower cost, then lower state index. On this table the winning efficiency is unique. This is the reference optimum. Building it costs 64 feasibility and score reads and, in the saved run, about 0.03 ms.

**Uniform feasible sampling** draws with replacement from the feasible index list, sorted by state index, using `numpy.random.Generator.choice`. Three runs use seeds 11, 22, and 33, with 4,096 draws each, the same sample size as `config.SHOTS`. Constructing the pool costs one read of each catalogue row. These seeds are sampling seeds. They are separate from the QAOA optimizer seed.

**Absorption-first greedy** does not rank efficiencies. It walks materials by the catalogue record efficiency (highest first), geometries by the optical proxy `(1 - reflectance) * path_enhancement` (highest first), and passivations by surface recombination velocity on c-Si (lowest first). Ties in each list break toward lower step cost, then lower category id. The first candidate that the score table marks feasible is accepted. On this budget that walk checks 24 candidates.

**Coordinate greedy** starts at the lowest feasible state index and makes one pass in the order material, geometry, passivation. For each category it tries all four values, holds the other two fixed, and keeps the feasible choice with the highest efficiency, then lower cost, then lower category index. The current feasible design is always one of the four tries, so the pass stays feasible. The saved run uses 13 queries, including the start scan.

`src/metrics.py` scores a sample list, a count dictionary, or a length-64 weight vector. The reported quality metrics are on physical efficiency and on r. The penalized energy E is not used to compare runs, because E changes when M changes.

| Metric | Definition |
|---|---|
| Feasible fraction | Feasible outcomes divided by all outcomes |
| p(best) | Fraction of outcomes equal to the tie-broken exact optimum |
| Optimum-hit rate | Fraction within 1e-9 efficiency of that optimum. On this table the optimum is unique, so this matches p(best) |
| Top-k hit rate | Fraction landing in the k best feasible designs (k = 1 and k = 3) |
| Chance, all bitstrings | 1/64 |
| Chance, feasible pool | 1/15 for this budget |
| Mean feasible score | Shot-weighted mean of r over feasible outcomes |
| Objective gap | 1 − r of the best feasible outcome in the sample |
| Efficiency gap | Optimum efficiency minus that outcome’s efficiency, also given in percentage points |

If a sample contains no feasible design, the feasible fraction is 0 and the best score, mean score, and both gaps are left empty. They are not filled with zero, and the exact optimum is not substituted. Prefix curves keep draw order at 1, 4, 16, 64, 256, and 4,096 samples.

## 6. What was run

Completed and saved:

- Exact ranking of the 15 feasible designs at 51.0 USD/m².
- Three random-feasible runs, 4,096 draws each.
- Both greeds, with query counts.
- A penalty audit at M = 0, 0.25, 1, 2, 5, and 20 on both score scales. The study’s training penalties include 0.25, 1, 5, and 20.
- A diagonal-scan timing from 6 to 18 qubits on a synthetic planted objective, plus a timing of the real 64-design enumeration. The synthetic rows are a timing probe. They are not additional solar cells.

`experiments/scaling_study.py` is a stretch check of enumeration cost. It is not part of the design-selection comparison.

## 7. Results

Full rows, sample counts, and prefix curves are in [`results/raw/exact_baselines.json`](results/raw/exact_baselines.json). Timings are in [`results/raw/scaling_study.json`](results/raw/scaling_study.json).

**Exact optimum.** Bitstring `000111`: c-Si / micro-pyramids / Al₂O₃ (ALD). Lifetime-averaged efficiency 0.261381 (26.1381% of incident sunlight). Cost 50.0 USD/m². Normalized score r = 1. With the budget removed, the highest efficiency in the table is `001111`, c-Si / nanocone array / Al₂O₃ (ALD), at 0.263198 and 52.4 USD/m². That cell is process-compatible and over the 51.0 USD/m² cap, so it is infeasible for the optimization.

The next feasible designs are c-Si / flat + double-layer ARC / Al₂O₃ at 0.251245 and 49.9 USD/m² (`000011`), and c-Si / micro-pyramids / thermal SiO₂ at 0.247537 and 51.0 USD/m² (`000101`).

**Greedy designs.**

| Method | Design | Efficiency | Gap | Rank among 15 | Queries |
|---|---|---:|---:|---:|---:|
| Exact | `000111` c-Si / micro-pyramids / Al₂O₃ | 0.261381 | 0 | 1 | 64 |
| Coordinate greedy | `010011` CdTe / flat + ARC / Al₂O₃ | 0.217112 | 4.43 percentage points | 7 | 13 |
| Absorption-first | `001100` c-Si / nanocones / no coating | 0.120987 | 14.04 percentage points | 13 | 24 |

The absorption walk reaches unpassivated nanocones because the highest record-efficiency absorber that can meet the budget is c-Si, the strongest optical proxy that still fits is the nanocone, and Al₂O₃ on that geometry costs 52.4 USD/m². The coordinate pass starts at bare flat c-Si (`000000`, efficiency 0.117442, the worst feasible cell) and stops at CdTe / flat / Al₂O₃. A second pass is not part of the method.

**Random feasible draws.** Chance for one draw is 1/15 = 0.066667 for the single best design and 3/15 = 0.200 for the top three. All three seeds returned a feasible fraction of 1, because the sampler only draws from the feasible pool. Each 4,096-draw sample contains the optimum, so the efficiency gap of the best draw is 0. The per-draw hit rate stays at chance:

| Seed | p(best) | Top-3 hit rate | Mean feasible r | First draw of the optimum |
|---:|---:|---:|---:|---:|
| 11 | 0.060303 | 0.189453 | 0.534151 | 4 |
| 22 | 0.068359 | 0.196533 | 0.531871 | 23 |
| 33 | 0.068359 | 0.202393 | 0.531777 | 9 |

No random run failed. Failure frequency across the three runs is 0.

**Penalty.** On the training scale, M must be greater than 0.01262 before every infeasible energy is strictly worse than the best feasible energy. M = 0 selects `001111`. M = 0.25, 1, 2, 5, and 20 all select `000111`, and every infeasible state is strictly more expensive. On the unit-interval scale the corresponding threshold is 0.00690, and the same M values select the physical optimum. The saved energies for M = 0.25 and M = 2 are in the results file.

**Enumeration cost.** The frozen 64-design ranking took about 0.08 ms in the saved timing run. A synthetic linear scan of a planted diagonal objective took about 3 µs at 6 qubits (64 states) and about 10 ms at 18 qubits (262,144 states). Every planted optimum was found. At this catalogue size, exact enumeration is the reference solver. The sampling study asks whether a method’s distribution prefers the good designs, which is a separate question from wall-clock time.

The QAOA sweep in [`results/raw/sim_sweep.json`](results/raw/sim_sweep.json) names the same best feasible design, `000111`, and the same feasible chance, 1/15.

## 8. Discussion and limitations

The exact list is the right reference for this frozen table: 64 scores, one budget, and a unique feasible winner. It does not generalize to a larger materials library or to a different budget. Changing `COST_BUDGET` changes which rows are feasible and can change the winner. The unconstrained winner is already a different cell, 0.18 percentage points higher and 1.4 USD/m² over the cap.

Uniform draws from 15 feasible designs will almost always include the optimum somewhere in 4,096 samples. A gap of zero on the best draw is the expected behaviour of that sampler. The informative comparisons are p(best) and the top-3 hit rate, which sit on the chance lines 1/15 and 3/15, and the mean feasible score, about 0.53 on a scale whose optimum is 1.

Both greeds return a legal design and both miss the optimum. Absorption-first misses because its order uses catalogue record efficiency and an optical proxy, then accepts the first feasible triple. High light-trapping with no passivation is a poor electrical design in this model. Coordinate descent misses because one pass from bare flat c-Si never reaches micro-pyramids. Query counts of 13 and 24 are the cost of those rules. They are not an exhaustive search.

The penalty check is specific to these efficiencies. The sufficient condition “M > 1 when every r is in [0, 1]” holds on the unit-interval scale. On the training scale the over-budget nanocone cell has r = 1.01262, so the instance threshold is 0.01262. Both M = 0.25 and M = 2 clear that threshold here. A catalogue whose best infeasible cell were much more efficient than every feasible cell would need a larger M.

Efficiencies and costs are taken from `scores.json` as frozen. The classical code does not refit them. Gaps are in the units of that table. They are not a measured outdoor energy yield.

## 9. Conclusions

Under the 51.0 USD/m² cap, the unique best feasible design is c-Si with micro-pyramids and Al₂O₃ (ALD): efficiency 0.261381 at 50.0 USD/m². Exact enumeration finds it by reading the whole table.

A uniform draw from the 15 feasible designs finds that cell with probability 1/15. The three 4,096-draw runs land at 0.060 to 0.068, around that chance rate, and their top-3 hit rates are 0.189 to 0.202, around 3/15. Each of those samples also contains the optimum at least once, so the best draw in a 4,096-draw sample has an efficiency gap of zero. The mean normalized score of the feasible draws is about 0.53.

The absorption-first rule and one coordinate-greedy pass are feasible and incomplete. They land at ranks 13 and 7, short of the optimum by 14.04 and 4.43 percentage points, after 24 and 13 queries.

For the cost actually used in training, any penalty above 0.01262, including every positive penalty in `config.py`, makes `000111` the unique minimizer. M = 0 does not: it selects the over-budget nanocone cell.

A useful next measurement, once more QAOA samples are in, is to put p(best), top-3 hit rate, and mean feasible r next to the three rows above, at the same 4,096-shot budget. Those are the comparisons this catalogue can support. Beating the exact list on wall-clock time is not one of them: ranking these 64 designs takes well under a millisecond.

## 10. Sources and contributions

The score table, spectrum, and cited material parameters belong to the physics model in `src/physics.py`, `data/catalogue.json`, `data/am15g.csv`, and `docs/assumptions.md`. The classical code only reads that table. Random draws use NumPy’s `Generator.choice`. The cost diagonal is the same formula as `src/hamiltonian.py`.

| Person | Role in this repository | Where the work lives |
|---|---|---|
| Safa | Physics model, catalogue, and assumptions | `src/physics.py`, `data/`, `docs/assumptions.md` |
| Alex | Six-qubit encoding, exact and greedy baselines, sample metrics, penalty audit, enumeration timing | `src/encoding.py`, `src/baselines.py`, `src/metrics.py`, `experiments/run_exact.py`, `experiments/scaling_study.py`, `tests/test_encoding.py` |
| Timane | QAOA circuit, training, and simulator sweep | `src/hamiltonian.py`, `src/qaoa.py`, `experiments/run_sim_sweep.py`, `docs/methods.md` |
| Yassine | Environment, noisy simulation, and hardware execution | `requirements.txt`, `src/ibm_backend.py`, `experiments/run_noisy_sim.py`, `experiments/run_hardware.py`, `docs/hardware_notes.md` |
| Yaniss | Repository assembly, figures, and the shared run script | `src/plotting.py`, `run_all.sh`, `scripts/check_all.sh` |
