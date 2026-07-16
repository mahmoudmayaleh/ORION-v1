#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 0
[ -f runs/MDO_REWARD_SWEEP_DONE ] && exit 0
screen -ls 2>/dev/null | grep -qE '[0-9]+\.sweep' && exit 0
screen -dmS sweep ./scripts/run_sweep.sh
