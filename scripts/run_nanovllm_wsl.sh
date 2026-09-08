#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/d/nano-vllm-qos}"
MODEL_DIR="${MODEL_DIR:-/mnt/d/models/Qwen3-0.6B}"
VENV_DIR="${VENV_DIR:-/home/xuhang/.venvs/nanovllm-qos}"
PORT="${PORT:-8020}"
SCHEDULING_POLICY="${SCHEDULING_POLICY:-pals}"
PREFIX_CACHE_BACKEND="${PREFIX_CACHE_BACKEND:-radix}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
export PATH="$CUDA_HOME/bin:$VENV_DIR/bin:$PATH"

EXTRA_ARGS=()
if [[ -n "${MAX_NUM_ACTIVE_SEQS:-}" ]]; then
  EXTRA_ARGS+=(--max-num-active-seqs "$MAX_NUM_ACTIVE_SEQS")
fi
if [[ -n "${SCHEDULER_TRACE_PATH:-}" ]]; then
  EXTRA_ARGS+=(--scheduler-trace-path "$SCHEDULER_TRACE_PATH")
fi
if [[ -n "${NUM_KVCACHE_BLOCKS:-}" ]]; then
  EXTRA_ARGS+=(--num-kvcache-blocks "$NUM_KVCACHE_BLOCKS")
fi
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
  if [[ "${REMOTE_KV_FORCE_RESTORE:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--remote-kv-force-restore)
  fi
fi

cd "$PROJECT_DIR"
exec "$VENV_DIR/bin/python" -m "${VALIDATION_SERVE_MODULE:-nanovllm.serve}" \
  --model "$MODEL_DIR" \
  --served-model-name Qwen3-0.6B \
  --scheduling-policy "$SCHEDULING_POLICY" \
  --metrics-summary-mode "${METRICS_SUMMARY_MODE:-cached}" \
  --kv-admission-lookahead "${KV_ADMISSION_LOOKAHEAD:-0}" \
  --prefix-cache-backend "$PREFIX_CACHE_BACKEND" \
  --kv-reclaim-policy "${KV_RECLAIM_POLICY:-slo_aware}" \
  --kv-reclaim-min-keep-ratio "${KV_RECLAIM_MIN_KEEP_RATIO:-0.0}" \
  --kv-reclaim-max-keep-ratio "${KV_RECLAIM_MAX_KEEP_RATIO:-0.75}" \
  --kv-reclaim-budget-scale-ms "${KV_RECLAIM_BUDGET_SCALE_MS:-1000}" \
  --kv-reclaim-target-free-blocks "${KV_RECLAIM_TARGET_FREE_BLOCKS:-2}" \
  --kv-compression-policy "${KV_COMPRESSION_POLICY:-none}" \
  --kv-compression-sink-blocks "${KV_COMPRESSION_SINK_BLOCKS:-1}" \
  --kv-compression-recent-blocks "${KV_COMPRESSION_RECENT_BLOCKS:-8}" \
  --kv-compression-importance-blocks "${KV_COMPRESSION_IMPORTANCE_BLOCKS:-2}" \
  --kv-compression-query-tokens "${KV_COMPRESSION_QUERY_TOKENS:-64}" \
  --kv-compression-trigger-free-ratio "${KV_COMPRESSION_TRIGGER_FREE_RATIO:-0.15}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enforce-eager \
  --host 0.0.0.0 \
  --port "$PORT" \
  "${EXTRA_ARGS[@]}"
