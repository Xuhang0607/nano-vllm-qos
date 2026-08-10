#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
VENV_DIR="${VENV_DIR:-/home/xuhang/.venvs/nanovllm-qos}"
MASTER_ADDR="${MOONCAKE_MASTER:-127.0.0.1:50051}"

case "$MODE" in
  master)
    exec "$VENV_DIR/bin/mooncake_master" \
      --rpc_port="${MOONCAKE_MASTER_PORT:-50051}" \
      --metrics_port="${MOONCAKE_METRICS_PORT:-9003}"
    ;;
  store)
    export MOONCAKE_MASTER="$MASTER_ADDR"
    exec "$VENV_DIR/bin/mc_store_rest_server" \
      -Dlocal_hostname="${MOONCAKE_STORE_HOSTNAME:-localhost}" \
      -Dmetadata_server="${MOONCAKE_METADATA:-P2PHANDSHAKE}" \
      -Dglobal_segment_size="${MOONCAKE_STORE_SEGMENT_BYTES:-2147483648}" \
      -Dlocal_buffer_size="${MOONCAKE_STORE_BUFFER_BYTES:-67108864}" \
      -Dprotocol="${MOONCAKE_PROTOCOL:-tcp}" \
      -Ddevice_name="${MOONCAKE_RDMA_DEVICES:-}" \
      -Dmaster_server_address="$MASTER_ADDR" \
      --port "${MOONCAKE_STORE_HTTP_PORT:-9004}"
    ;;
  *)
    echo "Usage: $0 {master|store}" >&2
    exit 2
    ;;
esac
