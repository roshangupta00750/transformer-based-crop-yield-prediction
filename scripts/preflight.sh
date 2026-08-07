#!/usr/bin/env bash
# Pre-flight for any julia2 submission. Run this BEFORE sbatch, every time.
#
# Exists because six separate runs in one week were lost to avoidable causes:
# a sbatch success-gate that tested for the wrong artefact, an `afterok`
# dependency on an array whose task had only cosmetically failed, a `--resume`
# that silently treated stale rows as complete, a guessed-at QoS name, and
# twice submitting a script whose fix had been committed but never rsynced.
# None of those were cluster problems and none produced any information.
#
# Usage:  bash scripts/preflight.sh scripts/<sbatch script>.sh
set -uo pipefail

SBATCH_FILE="${1:?usage: preflight.sh scripts/<sbatch>.sh}"
REMOTE=julia2
RROOT='~/crop_yield_benchmark'
LROOT="$(cd "$(dirname "$0")/.." && pwd)"
fails=0

# Resolve an interpreter that has the project deps. A bare `python` is not on
# PATH in a fresh non-interactive shell, which silently failed the test gate.
PY=""
for cand in "${CONDA_PREFIX:-/nonexistent}/bin/python" \
            "$HOME/miniconda3/envs/agri/bin/python" \
            "$(command -v python3 2>/dev/null)"; do
  if [ -x "$cand" ] && "$cand" -c "import pandas, torch" 2>/dev/null; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
  printf '  \033[31mFAIL\033[0m  no python with project deps found\n'; exit 1
fi
echo "     interpreter: $PY"

ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; fails=$((fails+1)); }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$1"; }

echo "=== 1. local tests ==="
if (cd "$LROOT" && "$PY" -m pytest -q >/tmp/pf_tests.log 2>&1); then
  ok "pytest ($(grep -oE '[0-9]+ passed' /tmp/pf_tests.log | tail -1))"
else
  bad "pytest — see /tmp/pf_tests.log"; tail -5 /tmp/pf_tests.log | sed 's/^/        /'
fi

# Compile only TRACKED sources. Untracked scratch files are never rsynced to
# the cluster, so they cannot break a job; including them let a stray editor
# buffer (dashboard/pages/01_Problem.py, rewritten to disk seconds after every
# deletion) block submission of work that was otherwise ready.
if (cd "$LROOT" && git ls-files 'src/**/*.py' 'scripts/*.py' 'dashboard/**/*.py' \
      'dashboard/*.py' 'tests/*.py' 2>/dev/null | tr '\n' '\0' \
      | xargs -0 "$PY" -m py_compile >/tmp/pf_compile.log 2>&1); then
  ok "compile (tracked sources)"
else
  bad "compile — see /tmp/pf_compile.log"; head -4 /tmp/pf_compile.log | sed 's/^/        /'
fi

echo "=== 2. reachability ==="
if ssh -T -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" true 2>/dev/null; then
  ok "ssh $REMOTE"
else
  bad "ssh $REMOTE unreachable (VPN down?) — stopping"; exit 1
fi

echo "=== 3. code actually synced (not merely committed) ==="
# Checks EVERY tracked source file, not a recent-commits window.
#
# The previous version compared `git diff HEAD~3`, which silently excluded
# anything older. src/training/trainer.py had changed 6 commits earlier, was
# committed alongside scripts/03_train.py but never rsynced, and the window
# could not see it. Three per-crop TFT jobs then trained for a full hour each
# and died on the last line:
#     TypeError: Trainer.save_metadata() got an unexpected keyword argument 'name'
# Three GPU-hours lost to a guard with a hole in it.
# macOS ships bash 3.2, which has no mapfile.
_tracked=""
while IFS= read -r _f; do _tracked="$_tracked$_f"$'\n'; done < <(cd "$LROOT" && git ls-files \
  'src/**/*.py' 'scripts/*.py' 'scripts/*.sh' 'dashboard/**/*.py' \
  'dashboard/*.py' 'configs/**/*.yaml' 'configs/*.yaml' 'tests/*.py' 2>/dev/null)
_stale=0
_count=0
_remote_sums=$(ssh -T "$REMOTE" "cd $RROOT 2>/dev/null && \
  find src scripts dashboard configs tests -type f \( -name '*.py' -o -name '*.sh' -o -name '*.yaml' \) \
  -exec sha256sum {} + 2>/dev/null")
while IFS= read -r rel; do
  [ -n "$rel" ] || continue
  [ -f "$LROOT/$rel" ] || continue
  _count=$((_count+1))
  lsum=$(shasum -a 256 "$LROOT/$rel" | cut -d' ' -f1)
  rsum=$(printf '%s\n' "$_remote_sums" | awk -v f="$rel" '$2==f || $2=="./"f {print $1; exit}')
  if [ -z "$rsum" ]; then bad "MISSING ON REMOTE: $rel"; _stale=$((_stale+1))
  elif [ "$lsum" != "$rsum" ]; then bad "STALE ON REMOTE: $rel"; _stale=$((_stale+1)); fi
done <<< "$_tracked"
[ "$_stale" -eq 0 ] && ok "all $_count tracked source files match the cluster"

echo "=== 4. sbatch directives ==="
part=$(sed -n 's/^#SBATCH --partition=\([^ ]*\).*/\1/p' "$SBATCH_FILE" | head -1)
qos=$( sed -n 's/^#SBATCH --qos=\([^ ]*\).*/\1/p'       "$SBATCH_FILE" | head -1)
gres=$(sed -n 's/^#SBATCH --gres=\([^ ]*\).*/\1/p'      "$SBATCH_FILE" | head -1)
echo "     partition=$part qos=${qos:-<none>} gres=$gres"
[ -n "$qos" ] || warn "no --qos: defaults to computervision, whose GPU-minute cap has been exhausted"

valid_qos=$(ssh -T "$REMOTE" "sacctmgr -n -P show assoc where user=\$USER format=qos" | tr ',' '\n' | tr -d ' ')
if [ -n "$qos" ]; then
  echo "$valid_qos" | grep -qx "$qos" && ok "qos '$qos' is granted" || bad "qos '$qos' NOT granted (have: $(echo $valid_qos | tr '\n' ' '))"
fi

echo "=== 5. scheduler accepts it, and at what cost ==="
res=$(ssh -T "$REMOTE" "cd $RROOT && sbatch --test-only $(basename "$SBATCH_FILE" | sed 's|^|scripts/|') 2>&1" | head -3)
echo "$res" | sed 's/^/     /'
echo "$res" | grep -qi 'error'   && bad "scheduler rejected it"      || ok "scheduler accepts"
echo "$res" | grep -qi 'preempt' && bad "would PREEMPT other users' jobs — pick another partition"

echo "=== 6. free GPUs on the target partition ==="
ssh -T "$REMOTE" "sinfo -p $part -h -o '%.10t %.4D %.14G'" 2>/dev/null | sed 's/^/     /'

echo "=== 7. inputs the job will read ==="
ssh -T "$REMOTE" "cd $RROOT && \
  echo \"     dataset.parquet: \$(test -f data/processed/dataset.parquet && echo present || echo MISSING)\" && \
  echo \"     policy sidecar : \$(test -f data/processed/dataset.policy.json && echo present || echo MISSING)\" && \
  echo \"     per-crop TFT   : \$(ls -1 results/checkpoints/tft_*.pt 2>/dev/null | grep -v abs | wc -l)/10\" && \
  echo \"     venv           : \$(test -d ~/agri-venv && echo present || echo MISSING)\""

echo "=== 8. smoke: does the training entrypoint actually RUN on the cluster? ==="
# The checks above validate the ENVIRONMENT. None of them execute the code
# path, so a signature mismatch between two synced-but-incompatible files
# survives every one of them. That is not hypothetical: five jobs trained for
# 31-60 minutes each and then died on the final line with
#     TypeError: Trainer.save_metadata() got an unexpected kwarg 'name'
# roughly 4.5 GPU-hours lost to an error a 60-second run would have surfaced.
#
# This runs the real entrypoint end to end on one crop with a 1-minute budget,
# so imports, signatures, checkpoint naming and metadata writing are all
# exercised before an hour of GPU time is committed.
_smoke=$(ssh -T "$REMOTE" "cd $RROOT && source ~/agri-venv/bin/activate 2>/dev/null && \
  timeout 600 python -u scripts/03_train.py --model dlinear --crop rye \
    --suffix _smoketest 2>&1 | tail -25" 2>&1)
if printf '%s' "$_smoke" | grep -qE "Traceback|Error:|error:"; then
  bad "training entrypoint raises"
  printf '%s\n' "$_smoke" | grep -E "Traceback|Error|File \"" | head -6 | sed 's/^/        /'
else
  ok "training entrypoint runs end to end"
  ssh -T "$REMOTE" "rm -f $RROOT/results/checkpoints/dlinear_smoketest.* \
                       $RROOT/results/metrics/dlinear_smoketest_meta.json" 2>/dev/null
fi

echo
if [ "$fails" -eq 0 ]; then
  printf '\033[32mPREFLIGHT CLEAN — safe to sbatch\033[0m\n'
else
  printf '\033[31mPREFLIGHT: %d FAILURE(S) — do not submit\033[0m\n' "$fails"; exit 1
fi
