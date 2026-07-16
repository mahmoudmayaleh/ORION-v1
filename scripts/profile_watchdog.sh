#!/usr/bin/env bash
# Cron trigger: once the main batch run is COMPLETE, run the per-decision
# profiler exactly once. Retries on next tick if it dies mid-run.
cd "$(dirname "$0")/.." || exit 0
[ -f runs/COMPLETE ] || exit 0        # main run not finished yet
[ -f runs/PROFILE_DONE ] && exit 0    # already profiled
screen -ls 2>/dev/null | grep -qE '[0-9]+\.profile' && exit 0   # already running
echo "[profile-watchdog $(date '+%F %T')] main run COMPLETE -> launching profiler"
screen -dmS profile ./scripts/run_profile.sh
