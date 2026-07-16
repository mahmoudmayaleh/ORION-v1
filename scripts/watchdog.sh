#!/usr/bin/env bash
# Watchdog: ensures the durable ORION run is alive. Run from cron every 2 min.
# Recovers from ANY death of the screen session (kill, OOM, session cleanup,
# reboot) — relaunches the supervisor, which resumes from per-seed checkpoints.
cd "$(dirname "$0")/.." || exit 0

# Done? nothing to do.
[ -f runs/COMPLETE ] && exit 0

# Supervisor session already alive? leave it.
if screen -ls 2>/dev/null | grep -qE '[0-9]+\.orion'; then
  exit 0
fi

echo "[watchdog $(date '+%F %T')] no orion session and not COMPLETE -> relaunching"
screen -dmS orion ./scripts/run_durable.sh
