#!/usr/bin/env bash
# Autonomous overnight pipeline. Runs everything EXCEPT Track D, in order, unattended
# (survives session close). Recovery is a hard gate for local-model runs; frontier R.3 runs
# regardless. Sonnet steps share the $10 cap via data/api_cost_ledger.json (order B->C->R.3 so
# R.3 -- the sacrifice-first item -- absorbs any shortfall; Track B main table is protected).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs logs
S=runs/overnight_status.log
mark() { echo "[$(date '+%F %T')] $*" | tee -a "$S"; }

: > "$S"
rm -f runs/OVERNIGHT_COMPLETE
mark "PIPELINE START"

# ---- 1. Template recovery (GATE for local-model runs) ----
mark "STEP recover_template (sweep chat formats to reproduce R.2 ~84)"
./scripts/recover_template.sh
if [ -f runs/TEMPLATE_RECOVERED ]; then
  FMT=$(cat runs/TEMPLATE_RECOVERED); mark "TEMPLATE RECOVERED: chat_format=$FMT"; REC=1
else
  mark "TEMPLATE RECOVERY FAILED -> skipping local-model runs (E.1/B-tele/C-tele); frontier-only below"; REC=0
fi

# frontier key (rotated by user) into env for the Sonnet steps
set -a; source .env.frontier 2>/dev/null || mark "WARN: .env.frontier not sourced"; set +a

# ---- 2. E.1 (local, free) — resolves R.2-vs-R45 contradiction; anchor re-confirms recovery ----
if [ "$REC" = "1" ]; then
  mark "STEP E.1 (2x2 contradiction diagnostic; ~60-70 min)"
  if .venv/bin/python scripts/track_e_runner.py > logs/track_e1_overnight.log 2>&1; then mark "E.1 DONE"; else mark "E.1 FAIL (see logs/track_e1_overnight.log)"; fi
fi

# ---- 3. Track B (Agent A eval, tele+sonnet) — the main table, protected budget ----
if [ "$REC" = "1" ]; then
  mark "STEP Track B tele+sonnet (100 intents x 2 models x K^A on/off)"
  if .venv/bin/python scripts/track_b_runner.py --arms tele,sonnet > logs/track_b_overnight.log 2>&1; then mark "Track B DONE"; else mark "Track B FAIL (see logs/track_b_overnight.log)"; fi
else
  mark "STEP Track B SONNET-ONLY (local tele contaminated; frontier arm still valid)"
  if .venv/bin/python scripts/track_b_runner.py --arms sonnet > logs/track_b_overnight.log 2>&1; then mark "Track B(sonnet) DONE"; else mark "Track B(sonnet) FAIL"; fi
fi

# ---- 4. Track C (Agent B plan probe, tele+sonnet) — tests A.2 dated prediction ----
if [ "$REC" = "1" ]; then
  mark "STEP Track C tele+sonnet (30 RC arrivals)"
  if .venv/bin/python scripts/track_c_runner.py --arms tele,sonnet > logs/track_c_overnight.log 2>&1; then mark "Track C DONE"; else mark "Track C FAIL (see logs/track_c_overnight.log)"; fi
else
  mark "STEP Track C SONNET-ONLY"
  if .venv/bin/python scripts/track_c_runner.py --arms sonnet > logs/track_c_overnight.log 2>&1; then mark "Track C(sonnet) DONE"; else mark "Track C(sonnet) FAIL"; fi
fi

# ---- 5. R.3 (frontier seed 42, $6 subcap) — valid regardless of local server ----
mark "STEP R.3 frontier (seed 42, cache-off, \$6 subcap)"
if .venv/bin/python scripts/r3_runner.py > logs/r3_overnight.log 2>&1; then mark "R.3 DONE"; else mark "R.3 FAIL (see logs/r3_overnight.log)"; fi

# ---- summary ----
LEDGER_SPENT=$( [ -f data/api_cost_ledger.json ] && grep -oE '"spent_usd": *[0-9.]+' data/api_cost_ledger.json | grep -oE '[0-9.]+' || echo 0 )
mark "TOTAL Sonnet spend so far: \$${LEDGER_SPENT}"
mark "PIPELINE DONE"
touch runs/OVERNIGHT_COMPLETE
