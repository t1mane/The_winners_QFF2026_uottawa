# QNanoVolt: physics model, assumptions and limitations

Owner: Safa. Files: `data/catalogue.json`, `data/am15g.csv`, `src/physics.py`, `data/scores.json`.

## 1. What is being optimised

A single-junction solar cell under standard sunlight (ASTM G173 AM1.5G, 1000 W/m², 25 °C). Each design picks three things (2 bits each, 6 qubits, 64 designs):

| Choice | 00 | 01 | 10 | 11 |
|---|---|---|---|---|
| Material | c-Si | CdTe | CIGS | Perovskite |
| Geometry | Flat + double-layer ARC | Micro-pyramids | Nanowire array | Nanocone array |
| Passivation | None | Thermal SiO₂ | SiNx:H (PECVD) | Al₂O₃ (ALD) |

Bitstring layout: `material | geometry | passivation`, most significant bit first. `000111` = c-Si, micro-pyramids, Al₂O₃.

**Qiskit prints bitstrings little-endian (qubit 0 is the rightmost character).** The keys in `scores.json` use the layout above, so reverse Qiskit's strings before looking them up.

## 2. Outputs and units

- `efficiency`: fraction (0 to 1) of sunlight converted to electricity, averaged over a 25-year service life. This is the number to maximise.
- `efficiency_initial`: the same on day one, before degradation.
- `cost`: USD per m² of module, on the basis of NREL's 2020 minimum sustainable price. These are not current market prices.
- `feasible`: `false` when the passivation step is hotter than the absorber survives. Efficiency is then 0. This applies to 16 of the 64 designs.
- `eff_norm`, `cost_norm`: min-max normalised to [0, 1] across the 64 designs.

## 3. Model chain

1. **Absorptance** `a(E) = (1 − R) · (1 − exp(−X · α(E) · w))` above the bandgap, 0 below. `R` is the texture's reflectance, `X` its path-length enhancement, `w` the absorber thickness.
2. **Photocurrent** `J_gen = q ∫ a(E) · Φ_sun(E) dE`, with the tabulated AM1.5G photon flux.
3. **Radiative limit** `J0_rad = q ∫ a(E) · Φ_blackbody(E, 25 °C) dE` (detailed balance, as in Shockley and Queisser).
4. **Surface recombination** `S_eff = area_factor · [c · S_passivated + (1 − c) · S_bare]`. `c` is the fraction of the textured surface the layer actually coats.
5. **Lifetime** `1/τ_eff = 1/τ_bulk + 1/τ_surf`, with `τ_surf = w/S_eff + (4/D)(w/π)²`.
6. **Collection** `η_c = (L/w) · tanh(w/L)`, with `L = √(D · τ_eff)`.
7. **Voltage** `Voc = Voc_rad − ΔV + (kT/q) · ln(τ_eff / τ_eff,reference)`.
8. **Efficiency** `η₀ = FF(Voc) · Voc · J_gen · η_c / P_sun`, with Green's empirical fill factor.
9. **Service life** `η = η₀ ·` average of `(1 − d)^t` over 25 years, for a degradation rate `d`.

Cost is material + geometry step + passivation step.

**Calibration.** `ΔV` is the one fit parameter per material. It is set so that a reference design reproduces the certified record cell: pyramids + Al₂O₃ for c-Si, flat + Al₂O₃ for the three thin films. All other designs of that material are scored relative to it.

| Material | Record matched | Model Voc | Model Jsc | ΔV | 25-year retention |
|---|---|---|---|---|---|
| c-Si | 27.81 % | 0.765 V | 42.5 mA/cm² | 119 mV | 0.940 |
| CdTe | 23.10 % | 1.028 V | 25.4 mA/cm² | 158 mV | 0.940 |
| CIGS | 23.64 % | 0.703 V | 39.7 mA/cm² | 198 mV | 0.912 |
| Perovskite | 28.00 % | 1.214 V | 25.7 mA/cm² | 73 mV | 0.700 |

**Self-check.** For a perfect absorber, steps 2 and 3 give a limit of 33.76 % at 1.34 eV. The published value is 33.7 % at 1.34 eV.

## 4. Where the numbers come from

Every parameter in `catalogue.json` has a source string beginning with one of four labels.

- **measured**: reported in the cited source. This covers the spectrum, the silicon absorption table, all four passivation velocities on c-Si, the bare-surface velocities for Si and CdTe, the degradation rates for c-Si and CdTe, the five record efficiencies, the four material costs, the three texture reflectances, the nanowire and nanocone surface areas, and the nanowire path enhancement.
- **geometry**: follows from the shape (pyramid area √3, two passes in a flat cell).
- **calibration**: `ΔV` per material, the CdTe absorption strength (1 µm absorbs 92 %), and the c-Si base cost.
- **estimate**: our own assumption. The ones that matter are listed in section 6.

## 5. Results

Run `python src/physics.py --analyze` to reproduce these.

- **Best with no budget:** `001111` c-Si / nanocones / Al₂O₃, efficiency 0.2632, cost 52.4 USD/m².
- **Best within 52 USD/m²:** `000111` c-Si / pyramids / Al₂O₃, efficiency 0.2614, cost 50.0 USD/m². 20 designs fit this budget.
- **Best within 49.5 USD/m²:** `000010` c-Si / flat / SiNx, efficiency 0.2237, cost 49.4 USD/m². 6 designs fit this budget.
- 48 designs are feasible. No two of them have the same efficiency.

Pareto front (each row is the best design at or below its cost):

| Bitstring | Design | Efficiency | Cost (USD/m²) |
|---|---|---|---|
| `000000` | c-Si / flat / none | 0.1174 | 48.4 |
| `000100` | c-Si / pyramids / none | 0.1202 | 48.5 |
| `010000` | CdTe / flat / none | 0.1607 | 49.2 |
| `000010` | c-Si / flat / SiNx | 0.2237 | 49.4 |
| `000011` | c-Si / flat / Al₂O₃ | 0.2512 | 49.9 |
| `000111` | c-Si / pyramids / Al₂O₃ | 0.2614 | 50.0 |
| `001111` | c-Si / nanocones / Al₂O₃ | 0.2632 | 52.4 |

Best design per material: c-Si 0.2632, CdTe 0.2171 (flat / Al₂O₃, 50.7), CIGS 0.2155 (nanocones / Al₂O₃, 81.3), perovskite 0.2012 (nanocones / Al₂O₃, 65.3).

**Suggested main budget: 52 USD/m².** It excludes the unconstrained optimum by 0.4 USD/m², so the penalty term decides the answer. The top two designs differ by 0.7 % in efficiency, so top-k hit rate is a fairer metric than best-design probability alone. 49.5 USD/m² is a good second budget: the answer changes to a much cheaper passivation.

What the model says, in plain terms:

- **Silicon wins at almost every budget.** The one exception is a 0.2 USD/m² window around 49.2, where unpassivated CdTe leads. Silicon has the highest record efficiency and the lowest module price. This matches the market: c-Si was about 94 % of shipments in 2019 (NREL).
- **Passivation matters more than texture.** On c-Si, going from no passivation to Al₂O₃ doubles the efficiency (0.12 to 0.26). Going from flat to pyramids adds about one point.
- **Texture only pays if the passivation follows the shape.** Nanocones with conformal ALD Al₂O₃ are the best design. The same nanocones under directional PECVD SiNx fall to 0.14.
- **Nanowires never win.** Their surface area is 21 times a flat surface, and the extra recombination outweighs the light trapping.
- **Perovskite starts highest and ends lowest.** 28.0 % on day one for the reference design, but at best 0.201 averaged over 25 years at 3 %/year.

## 6. How solid the results are

`python src/physics.py --sensitivity 52` changes one estimate at a time across a plausible range and reports whether the winners move. 23 parameters were tested.

- **c-Si with Al₂O₃ wins in every case.** No change to any thin-film estimate, degradation rate or lifetime moves either winner.
- **Nanocones versus pyramids is a close call.** The nanocone's lead comes from its reflectance, where we applied a black-silicon measurement (1.21 %). At 3 % reflectance, pyramids win outright. With the same reflectance and path enhancement as pyramids, the two are level (0.2615 against 0.2614).
- **The budget winner changes only if nanocones are cheap enough to fit.** At a nanocone step cost of 1 USD/m², or an Al₂O₃ step cost of 0.8 USD/m², the nanocone design fits within 52 USD/m² and wins.
- **PECVD coverage does not affect the winners.** Setting it to 100 % on all textures, or lowering it to 80 % on cones and 50 % on wires, changes neither.
- **Perovskite calibrated on the 22.1 % mini-module** instead of the 0.05 cm² cell peaks at 0.158 and changes neither winner.

## 7. Limitations

- **It is a closed-form chain, not a device simulation.** It reproduces trends. Absolute efficiencies away from the reference designs are extrapolations. The nanocone c-Si design comes out at 28.0 % on day one, above the 27.81 % record. That is a model output, not a claim that such a cell exists. The best measured silicon nanocone cell is 13.7 % on a 10 µm film.
- **Record cells are matched to module costs.** Efficiencies are calibrated on laboratory cells and costs on full modules. Real modules are several points less efficient than record cells.
- **The passivation options are defined on silicon.** Their surface velocities were measured on c-Si. For the thin films they are mapped, on a log scale, between each material's bare and best surface. Real thin-film cells are passivated by their own junction layers, so this axis is the weakest part of the model for CdTe, CIGS and perovskite.
- **Step costs are estimates.** Material costs come from NREL. The texture and passivation step costs (0.5 to 4 USD/m²) do not. The Pareto front has gaps of 0.1 to 0.5 USD/m², which is smaller than the uncertainty in those estimates.
- **Costs are from 2020** and exclude scale, yield and trade effects. Perovskite and CIGS costs are for small production volumes.
- **Textures are treated as a property of the front surface.** A pyramid several µm tall cannot be etched into a 0.5 µm perovskite film. For thin films, read it as a textured substrate with a conformal absorber.
- **Reflectance is tied to the geometry**, as if each texture carried its own anti-reflection coating, whatever the passivation choice.
- **Collection is by diffusion only.** Thin-film cells also collect by drift in the junction field. A CdTe bulk lifetime below about 20 ns cannot reproduce the record cell in this model.
- **Perovskite lifetime is assumed.** No 25-year field data exists. 3 %/year sits inside published outdoor results, which run from under 1 %/year to 22 % in the first year.
- **Feasibility is a temperature rule only.** Thermal SiO₂ (1050 °C) is ruled out on all three thin films, and SiNx (300 °C) on perovskite. The 400 °C limit for CdTe and CIGS is our estimate.
- **No quantum advantage is claimed.** With 64 designs, exhaustive search is instant. The project benchmarks QAOA against exact enumeration.

## 8. Why it matters (slide text)

A solar cell is chosen from a design space that grows exponentially with every added option: material, surface texture, passivation, thickness, coatings. Each choice trades efficiency against manufacturing cost, and the choices interact. A nanotexture that traps more light also adds surface, and it only pays off if the passivation layer can coat that surface. We encode this trade-off as a cost function over bitstrings, built from published measurements, and ask whether QAOA can find the best design within a manufacturing budget. At 6 qubits every answer can be checked exactly, which makes this an honest small-scale benchmark of QAOA depth, penalty strength and hardware noise.

## 9. References

- ASTM G173-03 reference spectrum (AM1.5G), as distributed with the pvlib Python package.
- W. Shockley and H. J. Queisser, *J. Appl. Phys.* 32, 510 (1961). Limit of 33.7 % at 1.34 eV: https://ossila.com/pages/radiative-efficiency-limit
- M. A. Green and M. J. Keevers, "Optical properties of intrinsic silicon at 300 K," *Prog. Photovolt.* 3, 189 (1995). Table: https://www.pveducation.org/pvcdrom/materials/optical-properties-of-silicon
- R. Kotipalli et al., "Passivation effects of atomic-layer-deposited aluminum oxide," *EPJ Photovoltaics* 4, 45107 (2013). https://www.epj-pv.org/10.1051/epjpv/2013023/pdf
- H. Mäckel and A. Cuevas, "Determination of the surface recombination velocity of unpassivated silicon from spectral photoconductance measurements," *Proc. 3rd World Conf. Photovoltaic Energy Conversion*, 71 (2003).
- H. Savin et al., "Black silicon solar cells with interdigitated back-contacts achieve 22.1% efficiency," *Nat. Nanotechnol.* (2015). https://doi.org/10.1038/nnano.2015.89
- G. von Gastrow et al., "Analysis of the atomic layer deposited Al₂O₃ field-effect passivation in black silicon," *Sol. Energy Mater. Sol. Cells*. https://www.recercat.cat/handle/2072/286411
- E. Garnett and P. Yang, "Light Trapping in Silicon Nanowire Solar Cells," *Nano Lett.* (2010).
- S. Jeong, M. D. McGehee and Y. Cui, "All-back-contact ultra-thin silicon nanocone solar cells with 13.7% power conversion efficiency," *Nat. Commun.* 4, 2950 (2013). https://doi.org/10.1038/ncomms3950
- "Front side antireflection concepts for silicon solar cells with diffractive rear side structures," Fraunhofer ISE (2014). https://publica.fraunhofer.de/handle/publica/385714
- A. Nawabjan et al., "A Front Surface Optimization Study for Photovoltaic Application," *TELKOMNIKA* 16(4). http://doi.org/10.12928/telkomnika.v16i4.9059
- B. L. Smith et al., *Photovoltaic (PV) Module Technologies: 2020 Benchmark Costs and Technology Evolution Framework Results*, NREL/TP-7A40-78173 (2021). https://www.nrel.gov/docs/fy22osti/78173.pdf
- D. C. Jordan and S. R. Kurtz, "Photovoltaic Degradation Rates: An Analytical Review," *Prog. Photovolt.* 21, 12 (2013).
- D. C. Jordan et al., "Compendium of photovoltaic degradation rates," *Prog. Photovolt.* (2016). https://www.osti.gov/biblio/1259256
- M. A. Green et al., Solar Cell Efficiency Tables, versions 66 and 68, as reported by pv magazine (2026). https://www.pv-magazine.com/2026/06/08/all-solar-cell-efficiencies-at-a-glance-updated-7/
- Uppsala University, CIGS record of 23.64 % (2024). https://www.uu.se/en/press/press-releases/2024/2024-02-26-uppsala-university-sets-new-world-record-for-cigs-solar-cells
- EPFL thin-film PV lecture notes, CdTe and CIGS (CdTe record 23.1 %, 1 µm absorbs more than 92 %).
- "Minority-Carrier Lifetime and Surface Recombination Velocity in Single-Crystal CdTe," *IEEE J. Photovolt.* 5, 366 (2015); "Surface Passivation of CdTe Single Crystals," *IEEE J. Photovolt.* 5, 382 (2015).
- T. P. Weiss et al., "Time-resolved photoluminescence on double graded Cu(In,Ga)Se₂," *Sci. Technol. Adv. Mater.* 20, 313 (2019). https://doi.org/10.1080/14686996.2019.1586583
- "Surface passivation of organometal halide perovskites by atomic layer deposition," *Nanoscale Adv.* (2021). https://doi.org/10.1039/d1na00075f
- "Report on the relevance of perovskite module outdoor ageing performance and indoor UV degradation trend," *Nanoscale Adv.* (2025). https://doi.org/10.1039/d5na00622h
- imec perovskite mini-module field test (2025). https://bits-chips.com/?p=204975
- Bandgaps: "Efficiency Limits in Photovoltaics: Case of Single Junction Solar Cells," *Facta Universitatis, Ser. Electron. Energ.*; arXiv:1904.06033, Table II.
- Inspiration only, no values used: N. P. Irvin et al., *Solar RRL* 10, e70360 (2026).