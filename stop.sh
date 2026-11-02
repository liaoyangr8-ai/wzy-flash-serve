#!/usr/bin/env bash
# stop.sh -- Graceful shutdown for wzy-flash-serve
set -u
ROOT="/home/liaoyang/wzy-flash-serve"

PIDS=$(ps -eo pid,cmd | grep -E 'wzy-flash-serve/serve/server\.py' | grep -v grep | awk '{print $1}')
if [ -n "$PIDS" ]; then
  echo "[wzy-flash-serve] Sending SIGTERM to server PID(s): $PIDS..."
  kill -TERM $PIDS 2>/dev/null || true
fi

for i in $(seq 1 20); do
  ALIVE=$(ps -eo pid,cmd | grep -E 'wzy-flash-serve/(serve/server\.py|bin/wzy_pp2_engine)' | grep -v grep | awk '{print $1}')
  if [ -z "$ALIVE" ]; then
    echo "[wzy-flash-serve] Stopped cleanly after ${i}s."
    rm -f "$ROOT/wzy-serve.pid"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    exit 0
  fi
  sleep 1
done

echo "[wzy-flash-serve] Force killing remaining processes..."
ps -eo pid,cmd | grep -E 'wzy-flash-serve/(serve/server\.py|bin/wzy_pp2_engine)' | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
rm -f "$ROOT/wzy-serve.pid"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
