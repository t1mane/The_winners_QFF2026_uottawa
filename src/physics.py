"""
physics.py  -  QNanoVolt (Safa: physics + story)

Scores every solar-cell design (material x geometry x passivation) under
standard sunlight (ASTM G173 AM1.5G, 1000 W/m2, 25 C).

    score(design) -> {"efficiency": float in [0,1], "cost": float (USD/m2)}

`efficiency` is the average over the module's service life, so a material
that degrades quickly is penalised. `design` can be a 6-character bitstring
("000111"), a (material, geometry, passivation) tuple of indices, or a dict
with those three keys.

Bit layout (most significant bit first):  material(2) | geometry(2) | passivation(2)
WARNING: Qiskit prints bitstrings little-endian (qubit 0 is the RIGHTMOST
character). scores.json keys are in the layout above, so reverse Qiskit's
strings before looking them up.

Model chain (docs/assumptions.md explains each step):
    1. absorptance        a(E) = (1 - R) * (1 - exp(-X * alpha(E) * w))   for E > Eg
    2. photocurrent       J_gen = q * integral a(E) * AM1.5G photon flux
    3. radiative limit    J0_rad = q * integral a(E) * blackbody(300 K) flux   (detailed balance)
    4. surface            S_eff = area_factor * [c * S_passivated + (1 - c) * S_bare]
    5. lifetime           1/tau_eff = 1/tau_bulk + 1/tau_surf,  tau_surf = w/S_eff + (4/D)(w/pi)^2
    6. collection         eta_c = (L/w) tanh(w/L),  L = sqrt(D * tau_eff)
    7. voltage            Voc = Voc_rad - dV_material + (kT/q) ln(tau_eff / tau_eff_reference)
    8. efficiency         eta_0 = FF(Voc) * Voc * J_gen * eta_c / P_sun
    9. service life       eta = eta_0 * mean over 25 years of (1 - d)^t

dV_material is the one fit parameter per material. It is set so the reference
design (the best-passivated standard geometry) reproduces the certified
record cell for that material.

Run `python src/physics.py`               to (re)generate data/scores.json
Run `python src/physics.py --analyze`     to print the landscape diagnostics
Run `python src/physics.py --sensitivity` to vary every estimated parameter
No third-party packages are needed.
"""

import copy
import csv
import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE_PATH = ROOT / "data" / "catalogue.json"
SPECTRUM_PATH = ROOT / "data" / "am15g.csv"
SCORES_PATH = ROOT / "data" / "scores.json"

Q = 1.602176634e-19        # C
H = 6.62607015e-34         # J s
C = 2.99792458e8           # m/s
KB = 1.380649e-23          # J/K
HC_EV_NM = 1239.841984     # eV nm


# --------------------------------------------------------------------------- #
# Loading and decoding
# --------------------------------------------------------------------------- #
def load_catalogue(path=CATALOGUE_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_spectrum(path=SPECTRUM_PATH):
    """Return (wavelength_nm, irradiance_W_m2_nm) lists for AM1.5G."""
    wl, irr = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            wl.append(float(row[0]))
            irr.append(float(row[1]))
    return wl, irr


def parse_design(design):
    """Return (material_idx, geometry_idx, passivation_idx)."""
    if isinstance(design, str):
        b = design.replace(" ", "")
        if len(b) != 6 or set(b) - {"0", "1"}:
            raise ValueError(f"bitstring must be 6 bits of 0/1, got {design!r}")
        return int(b[0:2], 2), int(b[2:4], 2), int(b[4:6], 2)
    if isinstance(design, dict):
        return design["material"], design["geometry"], design["passivation"]
    m, g, p = design
    return int(m), int(g), int(p)


def design_to_bitstring(m, g, p):
    return f"{m:02b}{g:02b}{p:02b}"


def _trapz(y, x):
    return sum(0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i]) for i in range(len(x) - 1))


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class Model:
    """Holds a catalogue and the spectrum, and scores designs."""

    def __init__(self, catalogue=None, spectrum=None):
        self.cat = catalogue if catalogue is not None else load_catalogue()
        self.wl, self.irr = spectrum if spectrum is not None else load_spectrum()
        k = self.cat["constants"]
        self.T = k["temperature_K"]["value"]
        self.years = k["service_life_years"]["value"]
        self.kT = KB * self.T / Q                               # eV
        self.p_sun = _trapz(self.irr, self.wl)                  # W/m2
        # photon flux per nm: irradiance / photon energy
        self.flux = [i * w * 1e-9 / (H * C) for i, w in zip(self.irr, self.wl)]
        self.energy = [HC_EV_NM / w for w in self.wl]           # eV
        # blackbody photon flux at the cell temperature, per nm (Boltzmann limit)
        self.bb = []
        for w in self.wl:
            lam = w * 1e-9
            x = H * C / (lam * KB * self.T)
            self.bb.append(2.0 * math.pi * C / lam ** 4 * math.exp(-x) * 1e-9 if x < 700 else 0.0)
        self.alpha0 = self._calibrate_alpha0()
        self._dv = {}

    # -- absorption --------------------------------------------------------- #
    def _alpha_si(self, wl_nm):
        t = self.cat["silicon_absorption"]
        xs, ys = t["wavelength_nm"], t["alpha_per_cm"]
        if wl_nm <= xs[0]:
            return ys[0]
        if wl_nm >= xs[-1]:
            return ys[-1]
        for i in range(len(xs) - 1):
            if xs[i] <= wl_nm <= xs[i + 1]:
                f = (wl_nm - xs[i]) / (xs[i + 1] - xs[i])
                return math.exp(math.log(ys[i]) * (1 - f) + math.log(ys[i + 1]) * f)

    def alpha(self, mat, wl_nm, alpha0=None):
        """Absorption coefficient in cm^-1; zero below the bandgap."""
        e = HC_EV_NM / wl_nm
        if e <= mat["bandgap_eV"]:
            return 0.0
        if mat["absorption"] == "table":
            return self._alpha_si(wl_nm)
        a0 = (self.alpha0 if alpha0 is None else alpha0) * mat.get("alpha0_relative_to_CdTe", 1.0)
        return a0 * math.sqrt(e - mat["bandgap_eV"])

    def _calibrate_alpha0(self):
        """alpha0 such that the target thickness absorbs the target fraction in one pass."""
        tgt = self.cat["direct_gap_absorption"]["calibration_target"]
        mat = next(m["values"] for m in self.cat["materials"] if m["name"] == tgt["material"])
        w_cm = tgt["thickness_um"] * 1e-4
        above = [f if e > mat["bandgap_eV"] else 0.0 for f, e in zip(self.flux, self.energy)]
        total = _trapz(above, self.wl)

        def frac(a0):
            y = [f * (1 - math.exp(-self.alpha(mat, w, a0) * w_cm)) for f, w in zip(above, self.wl)]
            return _trapz(y, self.wl) / total

        lo, hi = 1e3, 1e7
        for _ in range(80):
            mid = math.sqrt(lo * hi)
            lo, hi = (mid, hi) if frac(mid) < tgt["single_pass_absorbed_fraction"] else (lo, mid)
        return math.sqrt(lo * hi)

    def absorptance(self, mat, geo):
        w_cm = mat["thickness_um"] * 1e-4
        x, r = geo["path_enhancement"], geo["reflectance"]
        return [(1 - r) * (1 - math.exp(-x * self.alpha(mat, w) * w_cm)) for w in self.wl]

    # -- recombination ------------------------------------------------------ #
    def surface_velocity(self, mat, geo, pas, g_idx):
        """Effective front-surface recombination velocity in cm/s."""
        passivations = self.cat["passivations"]
        s_si_bare = passivations[0]["values"]["S_on_cSi_cm_per_s"]
        s_si_best = min(p["values"]["S_on_cSi_cm_per_s"] for p in passivations)
        # quality of this layer on a log scale: 0 = bare, 1 = best layer measured on c-Si
        q = math.log(s_si_bare / pas["S_on_cSi_cm_per_s"]) / math.log(s_si_bare / s_si_best)
        s_bare, s_best = mat["S_bare_cm_per_s"], mat["S_best_cm_per_s"]
        s_pass = s_bare ** (1 - q) * s_best ** q
        c = pas["conformality"][g_idx]
        return geo["area_factor"] * (c * s_pass + (1 - c) * s_bare)

    def lifetime(self, mat, s_eff):
        w_cm = mat["thickness_um"] * 1e-4
        tau_surf = w_cm / s_eff + (4.0 / mat["D_cm2_per_s"]) * (w_cm / math.pi) ** 2
        return 1.0 / (1.0 / mat["tau_bulk_s"] + 1.0 / tau_surf)

    @staticmethod
    def fill_factor(voc, kt):
        """Green's empirical ideal fill factor FF0(Voc)."""
        v = voc / kt
        return (v - math.log(v + 0.72)) / (v + 1.0)

    def retention(self, mat):
        """Average of (1 - d)^t over the service life."""
        d = mat["degradation_pct_per_year"] / 100.0
        if d <= 0:
            return 1.0
        k = -math.log(1.0 - d)
        return (1.0 - math.exp(-k * self.years)) / (k * self.years)

    # -- one design, before calibration ------------------------------------- #
    def _raw(self, m, g, p):
        mat = self.cat["materials"][m]["values"]
        geo = self.cat["geometries"][g]["values"]
        pas = self.cat["passivations"][p]["values"]
        w_cm = mat["thickness_um"] * 1e-4
        a = self.absorptance(mat, geo)
        j_gen = Q * _trapz([x * f for x, f in zip(a, self.flux)], self.wl)      # A/m2
        j0_rad = Q * _trapz([x * b for x, b in zip(a, self.bb)], self.wl)       # A/m2
        s_eff = self.surface_velocity(mat, geo, pas, g)
        tau = self.lifetime(mat, s_eff)
        ldiff = math.sqrt(mat["D_cm2_per_s"] * tau)
        eta_c = (ldiff / w_cm) * math.tanh(w_cm / ldiff)
        j_sc = j_gen * eta_c
        voc_rad = self.kT * math.log(j_sc / j0_rad + 1.0)
        return {"j_gen": j_gen, "j_sc": j_sc, "voc_rad": voc_rad, "tau": tau,
                "s_eff": s_eff, "eta_c": eta_c}

    def _eff_from(self, raw, voc):
        if voc <= 0:
            return 0.0, 0.0
        ff = self.fill_factor(voc, self.kT)
        return ff * voc * raw["j_sc"] / self.p_sun, ff

    def reference(self, m):
        """(geometry, passivation) the record cell is mapped onto."""
        mat = self.cat["materials"][m]["values"]
        feas = [p for p in range(4) if self.feasible(m, p)[0]]
        best_p = min(feas, key=lambda p: self.cat["passivations"][p]["values"]["S_on_cSi_cm_per_s"])
        return mat["reference_geometry"], best_p

    def voltage_deficit(self, m):
        """dV (in V) so the reference design reproduces the record efficiency."""
        if m in self._dv:
            return self._dv[m]
        g, p = self.reference(m)
        raw = self._raw(m, g, p)
        target = self.cat["materials"][m]["values"]["record_efficiency"]
        if self._eff_from(raw, raw["voc_rad"])[0] < target:
            raise ValueError(
                f"{self.cat['materials'][m]['name']}: the radiative limit of the reference "
                f"design is below the record efficiency; check the absorption inputs")
        lo, hi = 0.0, raw["voc_rad"]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if self._eff_from(raw, raw["voc_rad"] - mid)[0] > target else (lo, mid)
        self._dv[m] = (0.5 * (lo + hi), raw["tau"])
        return self._dv[m]

    def feasible(self, m, p):
        mat = self.cat["materials"][m]
        pas = self.cat["passivations"][p]
        if pas["values"]["process_temp_C"] > mat["values"]["max_process_temp_C"]:
            return False, (f"{pas['name']} needs {pas['values']['process_temp_C']} C; "
                           f"{mat['name']} tolerates {mat['values']['max_process_temp_C']} C")
        return True, ""

    # -- public ------------------------------------------------------------- #
    def score(self, design, detail=False):
        m, g, p = parse_design(design)
        cat = self.cat
        mat = cat["materials"][m]["values"]
        cost = (mat["base_cost_usd_m2"] + cat["geometries"][g]["values"]["cost_usd_m2"]
                + cat["passivations"][p]["values"]["cost_usd_m2"])
        ok, why = self.feasible(m, p)
        if not ok:
            out = {"efficiency": 0.0, "cost": cost}
            if detail:
                out["detail"] = {"feasible": False, "reason": why, "efficiency_initial": 0.0,
                                 "Voc_V": 0.0, "Jsc_mA_cm2": 0.0, "FF": 0.0,
                                 "S_eff_cm_per_s": None, "tau_eff_s": None,
                                 "collection": 0.0, "retention": self.retention(mat)}
            return out
        dv, tau_ref = self.voltage_deficit(m)
        raw = self._raw(m, g, p)
        voc = raw["voc_rad"] - dv + self.kT * math.log(raw["tau"] / tau_ref)
        eff0, ff = self._eff_from(raw, voc)
        ret = self.retention(mat)
        out = {"efficiency": max(0.0, min(1.0, eff0 * ret)), "cost": cost}
        if detail:
            out["detail"] = {"feasible": True, "reason": "", "efficiency_initial": eff0,
                             "Voc_V": max(voc, 0.0), "Jsc_mA_cm2": raw["j_sc"] / 10.0, "FF": ff,
                             "S_eff_cm_per_s": raw["s_eff"], "tau_eff_s": raw["tau"],
                             "collection": raw["eta_c"], "retention": ret,
                             "voltage_deficit_V": dv}
        return out

    def describe(self, design):
        m, g, p = parse_design(design)
        c = self.cat
        return c["materials"][m]["name"], c["geometries"][g]["name"], c["passivations"][p]["name"]

    def sq_limit(self, bandgap_ev):
        """Shockley-Queisser efficiency for a perfect absorber: a self-check of steps 2-3."""
        a = [1.0 if e > bandgap_ev else 0.0 for e in self.energy]
        j = Q * _trapz([x * f for x, f in zip(a, self.flux)], self.wl)
        j0 = Q * _trapz([x * b for x, b in zip(a, self.bb)], self.wl)
        best = 0.0
        voc = self.kT * math.log(j / j0 + 1.0)
        for i in range(2000):
            v = voc * i / 2000
            best = max(best, v * (j - j0 * (math.exp(v / self.kT) - 1.0)))
        return best / self.p_sun


_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = Model()
    return _MODEL


def score(design, detail=False):
    """Return {'efficiency': 0..1, 'cost': USD/m2} (plus internals if detail=True)."""
    return _model().score(design, detail)


def describe(design):
    return _model().describe(design)


def all_designs():
    return [design_to_bitstring(m, g, p)
            for m, g, p in itertools.product(range(4), range(4), range(4))]


def build_scores(model=None):
    model = model or _model()
    rows = {}
    for b in all_designs():
        s = model.score(b, detail=True)
        d = s["detail"]
        names = model.describe(b)
        rows[b] = {
            "material": names[0], "geometry": names[1], "passivation": names[2],
            "efficiency": round(s["efficiency"], 6),
            "cost": round(s["cost"], 4),
            "feasible": d["feasible"],
            "efficiency_initial": round(d["efficiency_initial"], 6),
            "Voc_V": round(d["Voc_V"], 6),
            "Jsc_mA_cm2": round(d["Jsc_mA_cm2"], 4),
        }
        if not d["feasible"]:
            rows[b]["infeasible_reason"] = d["reason"]
    effs = [r["efficiency"] for r in rows.values()]
    costs = [r["cost"] for r in rows.values()]
    e0, e1, c0, c1 = min(effs), max(effs), min(costs), max(costs)
    for r in rows.values():
        r["eff_norm"] = round((r["efficiency"] - e0) / (e1 - e0), 6)
        r["cost_norm"] = round((r["cost"] - c0) / (c1 - c0), 6)
    return {
        "meta": {
            "units": {
                "efficiency": "fraction (0-1) of sunlight converted to electricity, averaged over a 25-year service life",
                "efficiency_initial": "fraction (0-1) on day one, before degradation",
                "cost": "USD per m2 of module, NREL 2020 minimum-sustainable-price basis; not market prices",
                "eff_norm / cost_norm": "min-max normalised to [0,1] over the 64 designs",
                "feasible": "false when the passivation step is hotter than the absorber survives; efficiency is then 0",
            },
            "bit_layout": model.cat["bit_layout"],
            "spectrum": "ASTM G173-03 AM1.5G, 1000 W/m2, 25 C",
            "normalisation": {"eff_min": e0, "eff_max": e1, "cost_min": c0, "cost_max": c1},
            "generated_by": "src/physics.py (deterministic, no randomness)",
        },
        "designs": rows,
    }


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def _label(v):
    return f"{v['material']} / {v['geometry']} / {v['passivation']}"


def best_under(designs, budget):
    feas = [(b, v) for b, v in designs.items() if v["cost"] <= budget + 1e-9]
    return max(feas, key=lambda kv: kv[1]["efficiency"]) if feas else None


def pareto(designs):
    front, best_e = [], -1.0
    for b, v in sorted(designs.items(), key=lambda kv: (kv[1]["cost"], -kv[1]["efficiency"])):
        if v["efficiency"] > best_e + 1e-12:
            front.append((b, v))
            best_e = v["efficiency"]
    return front


def analyze(scores, model=None):
    model = model or _model()
    d = scores["designs"]
    print("\nSelf-check: Shockley-Queisser limit of a perfect absorber")
    for eg in (1.12, 1.34, 1.45, 1.55):
        print(f"  Eg = {eg:.2f} eV -> {100 * model.sq_limit(eg):.2f} %")
    print("\nCalibration (reference design reproduces the record cell)")
    for m, mat in enumerate(model.cat["materials"]):
        g, p = model.reference(m)
        s = model.score((m, g, p), detail=True)["detail"]
        print(f"  {mat['name']:<10} ref = {model.cat['geometries'][g]['name']} + "
              f"{model.cat['passivations'][p]['name']}: eta0 = {100 * s['efficiency_initial']:.2f} %, "
              f"Voc = {s['Voc_V']:.3f} V, Jsc = {s['Jsc_mA_cm2']:.1f} mA/cm2, FF = {s['FF']:.3f}, "
              f"dV = {1000 * s['voltage_deficit_V']:.0f} mV, retention = {s['retention']:.3f}")
    items = sorted(d.items(), key=lambda kv: -kv[1]["efficiency"])
    print(f"\nFeasible designs: {sum(v['feasible'] for v in d.values())} / {len(d)}")
    print(f"Best unconstrained: {items[0][0]} {_label(items[0][1])} "
          f"eff = {items[0][1]['efficiency']:.4f} cost = {items[0][1]['cost']}")
    print("\nAll designs by efficiency:")
    for b, v in items:
        print(f"  {b} {_label(v):<52} eff = {v['efficiency']:.4f}  eta0 = {v['efficiency_initial']:.4f}"
              f"  cost = {v['cost']:5.1f}")
    print("\nPareto front (cost up, efficiency up):")
    for b, v in pareto(d):
        print(f"  {b} {_label(v):<52} eff = {v['efficiency']:.4f}  cost = {v['cost']:5.1f}")
    print("\nBudget sweep -> best design within budget")
    for budget in sorted({v["cost"] for v in d.values()}):
        b, v = best_under(d, budget)
        n = sum(1 for x in d.values() if x["cost"] <= budget + 1e-9)
        print(f"  <= {budget:5.1f}: {b} {_label(v):<52} eff = {v['efficiency']:.4f}  n = {n}")
    live = [round(v["efficiency"], 4) for v in d.values() if v["feasible"]]
    print(f"\nDistinct efficiencies among feasible designs (4 dp): {len(set(live))} / {len(live)}")


SENSITIVITY = [
    # (label, path into the catalogue, low value, high value)
    ("perovskite degradation %/yr", ("materials", 3, "degradation_pct_per_year"), 1.0, 6.0),
    ("CIGS degradation %/yr", ("materials", 2, "degradation_pct_per_year"), 0.5, 1.0),
    ("c-Si bulk lifetime s", ("materials", 0, "tau_bulk_s"), 3e-4, 3e-3),
    ("c-Si bare surface cm/s", ("materials", 0, "S_bare_cm_per_s"), 1e4, 1e6),
    ("CdTe bulk lifetime s", ("materials", 1, "tau_bulk_s"), 2e-8, 3.6e-7),
    ("CdTe thickness um", ("materials", 1, "thickness_um"), 1.0, 3.0),
    ("CIGS best surface cm/s", ("materials", 2, "S_best_cm_per_s"), 1e2, 1e4),
    ("perovskite best surface cm/s", ("materials", 3, "S_best_cm_per_s"), 1e1, 1e3),
    ("perovskite bulk lifetime s", ("materials", 3, "tau_bulk_s"), 1e-7, 2e-6),
    ("perovskite absorption strength", ("materials", 3, "alpha0_relative_to_CdTe"), 2.0, 4.0),
    ("CIGS absorption strength", ("materials", 2, "alpha0_relative_to_CdTe"), 1.0, 2.5),
    ("pyramid path enhancement", ("geometries", 1, "path_enhancement"), 10, 50),
    ("nanocone path enhancement", ("geometries", 3, "path_enhancement"), 10, 50),
    ("nanocone reflectance", ("geometries", 3, "reflectance"), 0.0121, 0.03),
    ("pyramid reflectance", ("geometries", 1, "reflectance"), 0.015, 0.0309),
    ("nanowire area factor", ("geometries", 2, "area_factor"), 5, 21),
    ("nanowire step cost", ("geometries", 2, "cost_usd_m2"), 1.0, 8.0),
    ("nanocone step cost", ("geometries", 3, "cost_usd_m2"), 1.0, 6.0),
    ("pyramid step cost", ("geometries", 1, "cost_usd_m2"), 0.3, 1.5),
    ("thermal SiO2 step cost", ("passivations", 1, "cost_usd_m2"), 1.0, 5.0),
    ("SiNx step cost", ("passivations", 2, "cost_usd_m2"), 0.5, 2.0),
    ("Al2O3 step cost", ("passivations", 3, "cost_usd_m2"), 0.8, 3.0),
    ("Al2O3 on c-Si cm/s (thermal ALD)", ("passivations", 3, "S_on_cSi_cm_per_s"), 3, 17),
]


def sensitivity(budget):
    base_cat = load_catalogue()
    spectrum = load_spectrum()
    base = build_scores(Model(base_cat, spectrum))["designs"]
    b_unc = max(base.items(), key=lambda kv: kv[1]["efficiency"])[0]
    b_bud = best_under(base, budget)[0]
    print(f"\nBaseline: unconstrained best {b_unc} ({_label(base[b_unc])}); "
          f"best within {budget} USD/m2 {b_bud} ({_label(base[b_bud])})")
    print("One estimate changed at a time. '=' means the winner is unchanged.\n")
    for label, (sec, idx, key), lo, hi in SENSITIVITY:
        cells = []
        for val in (lo, hi):
            cat = copy.deepcopy(base_cat)
            cat[sec][idx]["values"][key] = val
            if sec == "materials" and idx == 0 and key == "S_bare_cm_per_s":
                cat["passivations"][0]["values"]["S_on_cSi_cm_per_s"] = val
            if sec == "passivations" and idx == 3 and key == "S_on_cSi_cm_per_s":
                cat["materials"][0]["values"]["S_best_cm_per_s"] = val
            try:
                d = build_scores(Model(cat, spectrum))["designs"]
            except ValueError:
                cells.append(f"{val:g}: cannot reproduce the record cell, so this value is ruled out")
                continue
            u = max(d.items(), key=lambda kv: kv[1]["efficiency"])[0]
            w = best_under(d, budget)[0]
            cells.append(f"{val:g}: unconstrained {'=' if u == b_unc else u + ' ' + _label(d[u])}, "
                         f"budget {'=' if w == b_bud else w + ' ' + _label(d[w])}")
        print(f"  {label:<34} {cells[0]}\n  {'':<34} {cells[1]}")


if __name__ == "__main__":
    scores = build_scores()
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    print(f"Wrote {SCORES_PATH}  ({len(scores['designs'])} designs)")
    if "--analyze" in sys.argv:
        analyze(scores)
    if "--sensitivity" in sys.argv:
        i = sys.argv.index("--sensitivity")
        budget = float(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 51.0
        sensitivity(budget)