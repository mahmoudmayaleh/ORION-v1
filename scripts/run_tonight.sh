#!/usr/bin/env bash
# Tonight's runs, answering the reviewer's identification concern and extending
# the table to more seeds.
#
#   JOB 1  seed 42, RL-advised only. The control the headline comparison needs:
#          heuristic plan, advising ON. LLM-free, so ~21 min of CPU.
#   JOB 2  seed 43, every approach, LLM server on :8000.
#   JOB 3  seed 44, every approach, LLM server on :8002.
#
# Checkpoint rule is --final-segment throughout, matching every banked cell
# (Y13_final_segment). The §Y.14 selection probe is not registered for the new
# po_advised stack, and inventing one would make the arm's meaning depend on an
# undocumented choice.
#
# Jobs 2 and 3 both hold local-LLM backends, so the single-slot lock is disabled
# deliberately: one server per job, which is the form llm_backend.py documents.
# Memory-off is not run.
set -u
cd "$HOME/ORION" || exit 1

APPROACHES="Plain MDO-fullobs MDO-partial RL-alone RL-advised Full"
COMMON="--part 1 --final-segment --train-instances 8 --train-arrivals 500 --scenarios conventional \
        --levels L1 L2 L3 L4 --eval-instances 100 --arrivals 2000 --rounds 200"

launch() {   # launch <name> <seed> <approaches> <port> <lock-disable> [extra]
    local name=$1 seed=$2 aps=$3 port=$4 nolock=$5 extra=${6:-}
    screen -dmS "$name" bash -c "
        cd \$HOME/ORION
        source .venv/bin/activate
        export PYTHONHASHSEED=0
        export ORION_CELL_DIR=data/parity_cells
        ${nolock:+export ORION_LLM_LOCK_DISABLE=1}
        echo \"=== \$(date '+%F %T') START $name seed=$seed port=$port\"
        python scripts/grid_runner.py $COMMON --seeds $seed \
            --approaches $aps --port $port $extra --tag ${name^^} >> logs/run_$name.log 2>&1
        echo \"=== \$(date '+%F %T') END $name rc=\$?\"
    "
    echo "launched $name (seed $seed, port $port)"
}

launch seed42adv 42 "RL-advised"   8000 1 --eval-only
launch seed43    43 "$APPROACHES"  8000 1
launch seed44    44 "$APPROACHES"  8002 1
