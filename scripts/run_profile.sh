#!/usr/bin/env bash
# Cost-profiling pass for Table "Cost per decision" in the results section.
#
# The `profiled()` wraps are already on the production path, but they are
# no-ops unless a collector is active and only --profile sets one, so no cell
# banked so far carries a single cost event. This run re-executes one cell per
# approach at L2 on the reporting instance with profiling on.
#
# Cells go to a scratch directory: these are timing cells, the acceptance
# numbers of record stay in data/parity_cells. Same checkpoint rule as the
# banked run (--final-segment), so the timing is measured on the same policy
# that produced the acceptance numbers.
set -u
cd "$HOME/ORION" || exit 1
source .venv/bin/activate 2>/dev/null
export PYTHONHASHSEED=0
export ORION_CELL_DIR=data/profile_run_cells
export ORION_PROFILE_DIR=results/profile_cells
mkdir -p logs "$ORION_CELL_DIR" "$ORION_PROFILE_DIR"

echo "=== profile run start $(date '+%F %T') pid $$"
python scripts/grid_runner.py \
    --part 1 \
    --eval-only \
    --final-segment \
    --profile \
    --scenarios conventional \
    --seeds 42 \
    --approaches Plain MDO-fullobs MDO-partial RL-alone Full \
    --levels L2 \
    --train-instances 8 \
    --eval-instances 100 \
    --arrivals 2000 \
    --port 8000 \
    --tag PROFILE
rc=$?
echo "=== profile run end $(date '+%F %T') rc=$rc"
