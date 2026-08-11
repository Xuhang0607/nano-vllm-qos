#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/d/nano-vllm-qos}"
MODEL_DIR="${MODEL_DIR:-/mnt/d/models/Qwen3-0.6B}"
VENV_DIR="${VENV_DIR:-/home/xuhang/.venvs/nanovllm-qos}"
PORT="${PORT:-8020}"
SCHEDULING_POLICY="${SCHEDULING_POLICY:-pals}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
export PATH="$CUDA_HOME/bin:$VENV_DIR/bin:$PATH"

EXTRA_ARGS=()
if [[ "${ENABLE_MOONCAKE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(
    --kv-storage-backend mooncake
    --mooncake-master "${MOONCAKE_MASTER:-127.0.0.1:50051}"
    --mooncake-metadata "${MOONCAKE_METADATA:-P2PHANDSHAKE}"
    --mooncake-hostname "${MOONCAKE_HOSTNAME:-localhost}"
    --mooncake-protocol "${MOONCAKE_PROTOCOL:-tcp}"
    --mooncake-global-segment-mib "${MOONCAKE_GLOBAL_SEGMENT_MIB:-0}"
    --mooncake-local-buffer-mib "${MOONCAKE_LOCAL_BUFFER_MIB:-512}"
    --remote-kv-bandwidth-gbps "${REMOTE_KV_BANDWIDTH_GBPS:-100}"
    --remote-kv-fixed-latency-ms "${REMOTE_KV_FIXED_LATENCY_MS:-0.1}"
  )
fi

cd "$PROJECT_DIR"
exec "$VENV_DIR/bin/python" -m nanovllm.serve \
  --model "$MODEL_DIR" \
  --served-model-name Qwen3-0.6B \
  --scheduling-policy "$SCHEDULING_POLICY" \
  --prefix-cache-backend radix \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization 0.80 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port "$PORT" \
  "${EXTRA_ARGS[@]}"
