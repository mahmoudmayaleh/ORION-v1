#!/usr/bin/env bash
# Cron */2: relaunch WP7 run on any death until WP7_COMPLETE exists.
cd "$(dirname "$0")/.." || exit 0
[ -f runs/WP7_COMPLETE ] && exit 0
if screen -ls 2>/dev/null | grep -qE '[0-9]+\.wp7'; then exit 0; fi
screen -dmS wp7 ./scripts/run_wp7_durable.sh
