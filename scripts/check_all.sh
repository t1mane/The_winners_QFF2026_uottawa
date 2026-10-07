#!/usr/bin/env bash
# scripts/check_all.sh - one-command health check of the whole repo (everything except real-hardware runs).
#
# Run from the repo root with the venv active:
#     ./scripts/check_all.sh            full check (re-runs every experiment; ~2-4 minutes)
#     QUICK=1 ./scripts/check_all.sh    skips the slow sweep + scaling study, reuses committed results
#
# Prints PASS / WARN / FAIL per step. FAIL = something is broken. WARN = look at it, may be fine
# (e.g. a file that is still an empty stub). Exit code is 1 if anything FAILED.

pass=0; fail=0; warn=0
ok()   { echo "  PASS  $1"; pass=$((pass+1)); }
bad()  { echo "  FAIL  $1"; fail=$((fail+1)); }
note() { echo "  WARN  $1"; warn=$((warn+1)); }
tailout() { echo "$1" | tail -12 | sed 's/^/          /'; }

step() {            # step "name" cmd...   -> PASS if exit 0, else FAIL with the last lines of output
  local name="$1"; shift
  local out
  if out=$("$@" 2>&1); then ok "$name"; else bad "$name"; tailout "$out"; fi
}
is_stub() { [ ! -f "$1" ] || [ "$(wc -l < "$1")" -lt 5 ]; }

if [ ! -f config.py ] || [ ! -d src ]; then echo "Run this from the repo root."; exit 2; fi

echo "== 1. Environment"
if command -v python >/dev/null 2>&1; then
  if python -c "import qiskit, qiskit_aer, numpy, scipy, matplotlib, pytest" 2>/dev/null; then
    ok "python + qiskit/aer/numpy/scipy/matplotlib/pytest import"
  else bad "a required package is missing - activate the venv and: pip install -r requirements.txt"; fi
else bad "'python' not found - activate the venv: source .venv/bin/activate"; exit 1; fi

echo "== 2. Git"
git fetch -q 2>/dev/null
if git rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
  behind=$(git rev-list --count HEAD..'@{u}' 2>/dev/null)
  if [ "${behind:-0}" -gt 0 ]; then note "you are $behind commit(s) behind origin - run: git pull"; else ok "up to date with origin"; fi
fi
dirty=$(git status --short | grep -v '^??' | wc -l | tr -d ' ')
if [ "$dirty" -gt 0 ]; then note "$dirty tracked file(s) modified locally (committed? intended?):"; git status --short | grep -v '^??' | sed 's/^/          /'; else ok "no uncommitted changes to tracked files"; fi

echo "== 3. Secrets"
if git ls-files | grep -Eq '(^|/)\.env$'; then bad ".env is tracked by git - remove it and REGENERATE the IBM key"; else ok ".env is not tracked"; fi
if git grep -I -n -i -E "(ibm_?quantum_?token|api_?key|apikey|token)[\"' ]*[=:][\"' ]*[A-Za-z0-9_-]{30,}" -- . ':!*.lock.txt' ':!data/am15g.csv' >/tmp/secret_hits.txt 2>/dev/null; then
  bad "possible API key committed:"; sed 's/^/          /' /tmp/secret_hits.txt | cut -c1-120 | head -5
else ok "no API-key-like strings in tracked files"; fi

echo "== 4. Physics model -> data/scores.json"
if [ -f src/physics.py ] && [ -f data/catalogue.json ] && [ -f data/am15g.csv ]; then
  step "python src/physics.py runs" python src/physics.py
  if git ls-files --error-unmatch data/scores.json >/dev/null 2>&1 && ! git diff --quiet data/scores.json; then
    note "regenerating changed data/scores.json (it was stale) - commit the new one"
  else ok "data/scores.json matches the physics model"; fi
  step "scores.json: 64 designs, some feasible" python - <<'EOF'
import json
d = json.load(open("data/scores.json")); d = d.get("designs", d)
assert len(d) == 64, len(d)
f = sum(1 for v in d.values() if v["feasible"]); assert 0 < f < 64, f
EOF
else bad "missing src/physics.py, data/catalogue.json or data/am15g.csv"; fi

echo "== 5. Tests"
step "pytest (all test files)" python -m pytest -q -x

echo "== 6. Experiments"
for f in experiments/run_exact.py experiments/run_sim_sweep.py experiments/scaling_study.py; do
  name=$(basename "$f")
  if is_stub "$f"; then note "$name is empty or a stub"; continue; fi
  if [ -n "$QUICK" ] && { [ "$name" = run_sim_sweep.py ] || [ "$name" = scaling_study.py ]; }; then note "$name skipped (QUICK=1)"; continue; fi
  step "$name runs" python "$f"
done

echo "== 7. Consistency between QAOA results, physics and config"
if [ -f results/raw/sim_sweep.json ]; then
  step "sim_sweep.json agrees with scores.json + config.py" python - <<'EOF'
import json, sys
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import config
from hamiltonian import load_scores
budget = getattr(config, "COST_BUDGET", None)
score, feas = load_scores(budget=budget)
best = format(max((i for i in range(len(feas)) if feas[i]), key=lambda i: score[i]), "06b")
n = int(feas.sum())
sw = json.load(open("results/raw/sim_sweep.json"))
errs = []
for e in sw:
    m = e["metrics"]
    if e.get("budget") != budget: errs.append("sweep budget %s != config budget %s" % (e.get("budget"), budget)); break
    if m["best_design"] != best: errs.append("sweep best %s != true best %s" % (m["best_design"], best)); break
    if abs(m["random_p_best"] - 1.0 / n) > 1e-9: errs.append("random baseline is not 1/%d" % n); break
if errs: print("; ".join(errs)); sys.exit(1)
print("best design %s, %d feasible, %d sweep entries OK" % (best, n, len(sw)))
EOF
else note "results/raw/sim_sweep.json missing"; fi
if [ -f results/raw/exact_baselines.json ]; then
  if grep -q '"000111"' results/raw/exact_baselines.json; then ok "exact_baselines.json mentions the best design 000111"
  else note "exact_baselines.json never mentions 000111 - check it uses the same \$ budget as config.py"; fi
fi

echo "== 8. Figures"
if is_stub src/plotting.py; then note "src/plotting.py is empty or a stub"; else
  step "python src/plotting.py runs" python src/plotting.py
  for f in results/figures/p_best_vs_depth.png results/figures/p_feasible_vs_depth.png; do
    if [ -f "$f" ] && [ "$(wc -c < "$f")" -gt 30000 ]; then ok "$(basename $f) exists and is not blank-sized"; else bad "$f missing or suspiciously small"; fi
  done
  if grep -n -E "1/48|0\.021" src/plotting.py >/dev/null; then bad "plotting.py still hard-codes the 1/48 baseline"; else ok "plotting.py has no hard-coded 1/48 baseline"; fi
fi

echo "== 9. One-command reproduction"
if [ -n "$QUICK" ]; then note "run_all.sh skipped (QUICK=1)"; elif [ -x run_all.sh ]; then step "./run_all.sh" ./run_all.sh; else bad "run_all.sh missing or not executable (chmod +x run_all.sh)"; fi

echo "== 10. Still to do"
for f in src/*.py experiments/*.py; do is_stub "$f" && echo "          stub: $f"; done
if [ -f README.md ] && [ "$(wc -l < README.md)" -lt 30 ]; then note "README.md is only $(wc -l < README.md) lines - judges read this first"; else ok "README.md has real content"; fi
for f in docs/assumptions.md docs/methods.md docs/hardware_notes.md; do
  if [ ! -f "$f" ] || [ "$(wc -l < "$f")" -lt 6 ]; then note "$f is empty or a heading only"; fi
done

echo
echo "RESULT: $pass passed, $warn warnings, $fail FAILED"
[ "$fail" -eq 0 ]