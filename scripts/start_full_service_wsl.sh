#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/d/nano-vllm-qos}"
VENV_DIR="${VENV_DIR:-/home/xuhang/.venvs/nanovllm-qos}"
PORT="${PORT:-8020}"

MOONCAKE_MASTER_HOST="${MOONCAKE_MASTER_HOST:-127.0.0.1}"
MOONCAKE_MASTER_PORT="${MOONCAKE_MASTER_PORT:-50051}"
MOONCAKE_STORE_HOST="${MOONCAKE_STORE_HOST:-127.0.0.1}"
MOONCAKE_STORE_HTTP_PORT="${MOONCAKE_STORE_HTTP_PORT:-9004}"

export PROJECT_DIR
export VENV_DIR
export PORT
export MOONCAKE_MASTER="${MOONCAKE_MASTER:-$MOONCAKE_MASTER_HOST:$MOONCAKE_MASTER_PORT}"
export MOONCAKE_MASTER_PORT
export MOONCAKE_STORE_HTTP_PORT
export PATH="$VENV_DIR/bin:$PATH"

cd "$PROJECT_DIR"

port_is_open() {
  local host="$1"
  local port="$2"
  "$VENV_DIR/bin/python" - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.25)
try:
    sys.exit(0 if sock.connect_ex((host, port)) == 0 else 1)
finally:
    sock.close()
PY
}

wait_for_port() {
  local name="$1"
  local host="$2"
  local port="$3"
  local attempts="${4:-30}"

  for _ in $(seq 1 "$attempts"); do
    if port_is_open "$host" "$port"; then
      echo "[nano-vLLM QoS] $name is ready on $host:$port"
      return 0
    fi
    sleep 1
  done

  echo "[nano-vLLM QoS] WARNING: $name did not become ready on $host:$port" >&2
  return 1
}

start_mooncake_process() {
  local name="$1"
  local port="$2"
  local mode="$3"
  local stdout_log="$PROJECT_DIR/mooncake-${mode}.oneclick.stdout.log"
  local stderr_log="$PROJECT_DIR/mooncake-${mode}.oneclick.stderr.log"

  if port_is_open 127.0.0.1 "$port"; then
    echo "[nano-vLLM QoS] $name already appears to be running on port $port"
    return 0
  fi

  echo "[nano-vLLM QoS] Starting $name..."
  nohup bash scripts/run_mooncake_wsl.sh "$mode" >"$stdout_log" 2>"$stderr_log" &
}

if port_is_open 127.0.0.1 "$PORT"; then
  echo "[nano-vLLM QoS] Service already appears to be running on http://127.0.0.1:$PORT/"
  exit 0
fi

if [[ "${ENABLE_MOONCAKE:-0}" == "1" ]]; then
  start_mooncake_process "Mooncake Master" "$MOONCAKE_MASTER_PORT" master
  wait_for_port "Mooncake Master" "$MOONCAKE_MASTER_HOST" "$MOONCAKE_MASTER_PORT" 20 || true

  start_mooncake_process "Mooncake Store" "$MOONCAKE_STORE_HTTP_PORT" store
  wait_for_port "Mooncake Store" "$MOONCAKE_STORE_HOST" "$MOONCAKE_STORE_HTTP_PORT" 20 || true
fi

echo "[nano-vLLM QoS] Starting nano-vLLM on http://127.0.0.1:$PORT/"
exec bash scripts/run_nanovllm_wsl.sh
