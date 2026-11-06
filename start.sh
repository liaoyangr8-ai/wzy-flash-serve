#!/usr/bin/env bash
# start.sh -- Idempotent launcher for wzy-flash-serve (PP2 Dual-GPU SM75 Engine + WebUI & OpenAI Server)
# Usage: bash start.sh [--port PORT | PORT]
set -u
ROOT="/home/liaoyang/wzy-flash-serve"
cd "$ROOT" || exit 1

UV_PY="/home/liaoyang/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12"
if [ ! -x "$UV_PY" ]; then
  UV_PY="python3"
fi
export LD_LIBRARY_PATH="/home/liaoyang/ff75/third_party_env/cuda-12.8/lib64:/home/liaoyang/ff75/third_party_env/cuda-12.8/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

CLI_PORT=""
if [ "${1:-}" = "--port" ] && [ -n "${2:-}" ]; then
  CLI_PORT="$2"
elif [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  CLI_PORT="$1"
fi

PORT=$("$UV_PY" -c 'import json,sys; c=json.load(open("config.json")); print(sys.argv[1] or c.get("port",11437))' "$CLI_PORT")
LPORT=$("$UV_PY" -c 'import json; c=json.load(open("config.json")); print(c.get("launcher_port",11439))')

check_ready() {
  local code
  code=$(curl -s -o /dev/null -m 2 -w "%{http_code}" "http://127.0.0.1:${PORT}/health" 2>/dev/null || true)
  [ "$code" = "200" ]
}

if check_ready; then
  echo "[wzy-flash-serve] Already running and healthy on :${PORT} (WebUI: http://192.168.2.100:${PORT}/ | Launcher: http://192.168.2.100:${LPORT}/)"
  curl -s "http://127.0.0.1:${PORT}/health"
  echo
  exit 0
fi

for pt in "$PORT" "$LPORT"; do
  P=$(ss -ltnp 2>/dev/null | grep ":${pt} " | grep -oP 'pid=\K[0-9]+' | sort -u)
  if [ -n "$P" ]; then
    echo "[wzy-flash-serve] Stopping stale listener on :${pt} (PID $P)..."
    kill -TERM $P 2>/dev/null || true
    sleep 2
  fi
done

EXTRA_ARGS=()
if [ -n "$CLI_PORT" ]; then
  EXTRA_ARGS+=(--port "$CLI_PORT")
fi

echo "[wzy-flash-serve] Starting PP2 v2 Tensor-Core + MTP service & WebUI on :${PORT} (Launcher :${LPORT})..."
setsid nohup "$UV_PY" -u "$ROOT/serve/server.py" --config "$ROOT/config.json" "${EXTRA_ARGS[@]}" > "$ROOT/wzy-serve.log" 2>&1 < /dev/null &
SRV_PID=$!
echo "$SRV_PID" > "$ROOT/wzy-serve.pid"
sleep 1
echo "[wzy-flash-serve] WebUI & Launcher immediately live at: http://192.168.2.100:${PORT}/ (and http://192.168.2.100:${LPORT}/)"
echo "[wzy-flash-serve] Server PID $SRV_PID spawned, waiting for 63.3 GiB host pin & 13,224 VRAM slot prefill (~75s)..."

for i in $(seq 1 120); do
  if check_ready; then
    echo "[wzy-flash-serve] READY after $((i*2))s!"
    curl -s "http://127.0.0.1:${PORT}/health"
    echo
    exit 0
  fi
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[wzy-flash-serve] ERROR: server process exited unexpectedly. Tail of wzy-serve.log:"
    tail -n 25 "$ROOT/wzy-serve.log"
    exit 1
  fi
  sleep 2
done

echo "[wzy-flash-serve] TIMEOUT waiting for readiness. Tail of wzy-serve.log:"
tail -n 25 "$ROOT/wzy-serve.log"
exit 1
