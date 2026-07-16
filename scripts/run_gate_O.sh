#!/usr/bin/env bash
# Durable §O gate — FIRST VALID EXECUTION of the ratified §N experiment.
# (PREREG_AMENDMENT_2026-07-13_O.md, ratified 2026-07-13.)
#
# Differences vs the conformant gate (all §O, nothing else):
#   O.1 value normalization (once-per-round cadence, pinned)  O.2 Huber value loss
#   O.3 critic input = designed s_t (Choice A1)               O.4 KL prior canonicalized
#   O.5 mtilde_agreement one-frame                            O.6 ceiling on eval stream length
#   O.7 checkpoints under results/wp7/ckpt_O/                 O.8 EV telemetry in every curve
#   O.9 cost instrumentation incl. cell totals + peak memory
# Same family, same seeds 42/43/44, byte-identical streams. New --tag O so it CANNOT
# resume conformant cells. Resumes its own cells on restart.
#
# LAUNCH CONDITION (do not fire before): regression ladder L1-L5 green — see
# scripts/audit/ and PREREG_AMENDMENT_2026-07-13_O.md.
# BOX ACTION before first launch: check RAPL readability without root:
#   cat /sys/class/powercap/intel-rapl:0/energy_uj
# If permission denied, one-time admin ask (chmod a+r) — else CPU energy is a
# labeled TDP estimate (§O.9 honesty split).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs results/wp7
MARKER=runs/GATE_COLOCATION_O_DONE
[ -f "$MARKER" ] && { echo "already done"; exit 0; }
LOG="runs/gate_O_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$(basename "$LOG")" runs/gate_O_latest.log

# Ladder guard: refuse to start unless the ladder marker exists (written by
# scripts/audit/run_ladder.sh on all-green). Enforced by the program, not memory.
if [ ! -f runs/LADDER_O_GREEN ]; then
  echo "REFUSING TO START: runs/LADDER_O_GREEN missing — run the regression ladder first." | tee -a "$LOG"
  exit 1
fi

# Provenance guard (§O Δ3, STAMPED): byte-identical streams mean nothing if the
# code state has no hash. Refuse to fire on a dirty tree; the result JSON
# records the commit hash (gate_colocation_prior.py::git_state).
# --untracked-files=no: run artifacts (results/, runs/, logs) are outputs, not
# code state; the guard is about uncommitted changes to TRACKED files.
if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]; then
  echo "REFUSING TO START: tracked files have uncommitted changes (§O Δ3). Commit first." | tee -a "$LOG"
  git status --short --untracked-files=no | head -20 | tee -a "$LOG"
  exit 1
fi
# Untracked CODE is a harder failure than a dirty tree: a dirty tree still names a
# base commit to diff from, whereas an untracked runner leaves no trace at all, so
# a number it produced can never be reproduced or refuted. That is how the R family
# escaped this guard (those runners never ran through this wrapper) and how
# R.2|42 = 86.6% became unfalsifiable. Artifacts (results/, runs/, logs) stay exempt
# because the scope is limited to scripts/ and src/.
STRAY="$(git status --porcelain --untracked-files=all -- scripts src 2>/dev/null | grep '^??' || true)"
if [ -n "$STRAY" ]; then
  echo "REFUSING TO START: untracked code under scripts/ or src/ (Delta-3)." | tee -a "$LOG"
  echo "$STRAY" | head -20 | tee -a "$LOG"
  echo "Commit it (or .gitignore a genuine build artifact) and re-run." | tee -a "$LOG"
  exit 1
fi
echo "[gate-O] tree clean at commit $(git rev-parse HEAD)" | tee -a "$LOG"

n=0; MAX=12
while [ ! -f "$MARKER" ] && [ $n -lt $MAX ]; do
  # arms 2 & 3 need the LLM server; keep it alive across reruns.
  if ! curl -s -m 5 http://localhost:8000/v1/models >/dev/null 2>&1; then
    echo "[gate-O $(date +%H:%M:%S)] LLM :8000 down -> starting" | tee -a "$LOG"
    setsid ./scripts/start_llm_gpu.sh 8000 > llm_server_8000.log 2>&1 < /dev/null &
    sleep 30
  fi
  echo "[gate-O $(date +%H:%M:%S)] attempt $((n+1))/$MAX -> $LOG" | tee -a "$LOG"
  # --rounds 60, NOT the historical 15: ladder L1 measured a ~20-round critic
  # transient under §O (EV reaches ~0.99 and selection converges only by ~R50);
  # a 15-round gate would sit inside the transient and be invalid by its own
  # O.8 telemetry. Budget change flagged to the team lead in the ladder report.
  nice -n 19 ./.venv/bin/python -u scripts/gate_colocation_prior.py \
    --family C+_T-_B- --seeds 42 43 44 --rounds 60 --arrivals 60 \
    --bc-scenarios 2000 --bc-epochs 6 --port 8000 \
    --tag O --ent-c0 0.03 --ent-floor 0.01 2>&1 | tee -a "$LOG"
  [ -f "$MARKER" ] && break
  n=$((n+1)); sleep 15
done
echo "[gate-O $(date +%H:%M:%S)] exiting (done=$([ -f "$MARKER" ] && echo yes || echo no))" | tee -a "$LOG"
