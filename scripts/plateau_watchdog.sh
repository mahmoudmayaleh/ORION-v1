#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 0
[ -f runs/MDO_PLATEAU_DONE ] && exit 0
screen -ls 2>/dev/null | grep -qE '[0-9]+\.plateau' && exit 0
screen -dmS plateau ./scripts/run_plateau.sh
