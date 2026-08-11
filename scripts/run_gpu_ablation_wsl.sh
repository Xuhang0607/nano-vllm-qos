#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/d/nano-vllm-qos}"
VENV_DIR="${VENV_DIR:-/home/xuhang/.venvs/nanovllm-qos}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_DIR/benchmarks/results/repeated}"
REPEATS="${REPEATS:-3}"
HARDWARE="${HARDWARE:-NVIDIA GeForce RTX 4060 Laptop GPU 8GB}"
SUITE="${1:-prefix}"
SERVER_PID=""

mkdir -p "$RESULTS_DIR"
cd "$PROJECT_DIR"

stop_worker() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
  local leftovers
  leftovers="$(pgrep -f "$VENV_DIR/bin/python -m nanovllm.serve" || true)"
  if [[ -n "$leftovers" ]]; then
    kill -TERM $leftovers 2>/dev/null || true
  fi
  sleep 2
}

wait_server() {
  local policy="$1"
  local prefix="$2"
  local backend_fragment="$3"
  "$VENV_DIR/bin/python" - "$policy" "$prefix" "$backend_fragment" <<'PY'
import json
import sys
import time
import urllib.request

policy, prefix, backend_fragment = sys.argv[1:]
deadline = time.monotonic() + 600
last_error = None
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8020/v1/metrics", timeout=2) as response:
            metrics = json.load(response)
        if (
            metrics.get("worker_alive")
            and metrics.get("policy") == policy
            and metrics.get("prefix_cache_backend") == prefix
            and backend_fragment in metrics.get("backend", "")
        ):
            print(
                f"ready policy={policy} prefix={prefix} "
                f"backend={metrics.get('backend')} capacity={metrics.get('kv_cache_capacity_tokens')}"
            )
            raise SystemExit(0)
    except Exception as exc:
        last_error = exc
    time.sleep(2)
raise SystemExit(f"server did not become ready: {last_error}")
PY
}

start_worker() {
  local name="$1"
  local policy="$2"
  local prefix="$3"
  local mooncake="$4"
  local max_num_seqs="$5"
  local force_restore="${6:-0}"
  stop_worker
  env \
    SCHEDULING_POLICY="$policy" \
    PREFIX_CACHE_BACKEND="$prefix" \
    MAX_NUM_SEQS="$max_num_seqs" \
    GPU_MEMORY_UTILIZATION=0.80 \
    ENABLE_MOONCAKE="$mooncake" \
    REMOTE_KV_FORCE_RESTORE="$force_restore" \
    bash scripts/run_nanovllm_wsl.sh \
    >"$PROJECT_DIR/ablation-$name.stdout.log" \
    2>"$PROJECT_DIR/ablation-$name.stderr.log" &
  SERVER_PID=$!
  local backend="CUDA"
  if [[ "$mooncake" == "1" ]]; then
    backend="Mooncake"
  fi
  wait_server "$policy" "$prefix" "$backend"
}

restore_full_service() {
  local exit_code=$?
  trap - EXIT INT TERM
  stop_worker
  setsid -f env ENABLE_MOONCAKE=1 bash scripts/run_nanovllm_wsl.sh \
    >"$PROJECT_DIR/nanovllm-8020-restored.stdout.log" \
    2>"$PROJECT_DIR/nanovllm-8020-restored.stderr.log"
  echo "restoring PALS + Radix + Mooncake service on port 8020"
  exit "$exit_code"
}
trap restore_full_service EXIT INT TERM

run_benchmark() {
  local output="$1"
  local workload_id="$2"
  shift 2
  "$VENV_DIR/bin/python" -m benchmarks.benchmark_serving_gpu \
    --hardware "$HARDWARE" \
    --workload-id "$workload_id" \
    --output-json "$output" \
    "$@"
}

aggregate_groups() {
  local baseline_name="$1"
  local candidate_name="$2"
  local report_name="$3"
  shift 3
  local separator=0
  local baseline=()
  local candidate=()
  for path in "$@"; do
    if [[ "$path" == "--" ]]; then
      separator=1
    elif [[ "$separator" == "0" ]]; then
      baseline+=("$path")
    else
      candidate+=("$path")
    fi
  done
  "$VENV_DIR/bin/python" -m benchmarks.aggregate_serving_runs \
    --baseline-name "$baseline_name" \
    --baseline "${baseline[@]}" \
    --candidate-name "$candidate_name" \
    --candidate "${candidate[@]}" \
    --output-json "$RESULTS_DIR/$report_name.json" \
    --output-markdown "$RESULTS_DIR/$report_name.md"
}

run_scheduler_suite() {
  local fcfs_files=()
  local pals_files=()
  for run in $(seq 1 "$REPEATS"); do
    start_worker "scheduler-fcfs-$run" fcfs radix 0 1
    local fcfs="$RESULTS_DIR/scheduler-fcfs-$run.json"
    run_benchmark "$fcfs" scheduler-repeated-v1 \
      --batch-requests 4 --interactive-requests 4 \
      --shared-prefix-repeats 32 --batch-output-tokens 64 \
      --interactive-output-tokens 4
    fcfs_files+=("$fcfs")

    start_worker "scheduler-pals-$run" pals radix 0 1
    local pals="$RESULTS_DIR/scheduler-pals-$run.json"
    run_benchmark "$pals" scheduler-repeated-v1 \
      --batch-requests 4 --interactive-requests 4 \
      --shared-prefix-repeats 32 --batch-output-tokens 64 \
      --interactive-output-tokens 4
    pals_files+=("$pals")
  done
  aggregate_groups FCFS PALS scheduler-fcfs-vs-pals \
    "${fcfs_files[@]}" -- "${pals_files[@]}"
}

run_pressure_level() {
  local label="$1"
  local delay_ms="$2"
  local fcfs_files=()
  local pals_files=()
  for run in $(seq 1 "$REPEATS"); do
    start_worker "pressure-$label-fcfs-$run" fcfs radix 0 1
    local fcfs="$RESULTS_DIR/pressure-$label-fcfs-$run.json"
    run_benchmark "$fcfs" "pressure-$label-v1" \
      --workload-label "$label" \
      --batch-requests 4 --interactive-requests 8 \
      --interactive-delay-ms "$delay_ms" \
      --shared-prefix-repeats 32 --batch-output-tokens 64 \
      --interactive-output-tokens 4
    fcfs_files+=("$fcfs")

    start_worker "pressure-$label-pals-$run" pals radix 0 1
    local pals="$RESULTS_DIR/pressure-$label-pals-$run.json"
    run_benchmark "$pals" "pressure-$label-v1" \
      --workload-label "$label" \
      --batch-requests 4 --interactive-requests 8 \
      --interactive-delay-ms "$delay_ms" \
      --shared-prefix-repeats 32 --batch-output-tokens 64 \
      --interactive-output-tokens 4
    pals_files+=("$pals")
  done
  aggregate_groups FCFS PALS "pressure-$label-fcfs-vs-pals" \
    "${fcfs_files[@]}" -- "${pals_files[@]}"
}

run_pressure_suite() {
  run_pressure_level low 500
  run_pressure_level medium 200
  run_pressure_level high 50
  "$VENV_DIR/bin/python" -m benchmarks.render_pressure_report \
    --comparisons \
      "$RESULTS_DIR/pressure-low-fcfs-vs-pals.json" \
      "$RESULTS_DIR/pressure-medium-fcfs-vs-pals.json" \
      "$RESULTS_DIR/pressure-high-fcfs-vs-pals.json" \
    --output-json "$RESULTS_DIR/pals-pressure-summary.json" \
    --output-markdown "$RESULTS_DIR/pals-pressure-summary.md" \
    --output-svg "$PROJECT_DIR/assets/pals-pressure.svg"
}

run_prefix_suite() {
  local hash_files=()
  local radix_files=()
  for run in $(seq 1 "$REPEATS"); do
    start_worker "prefix-hash-$run" pals hash 0 64
    local hash="$RESULTS_DIR/prefix-hash-$run.json"
    run_benchmark "$hash" prefix-repeated-v1 \
      --batch-requests 6 --interactive-requests 0 \
      --shared-prefix-repeats 32 --batch-output-tokens 8
    hash_files+=("$hash")

    start_worker "prefix-radix-$run" pals radix 0 64
    local radix="$RESULTS_DIR/prefix-radix-$run.json"
    run_benchmark "$radix" prefix-repeated-v1 \
      --batch-requests 6 --interactive-requests 0 \
      --shared-prefix-repeats 32 --batch-output-tokens 8
    radix_files+=("$radix")
  done
  aggregate_groups Hash Radix prefix-hash-vs-radix \
    "${hash_files[@]}" -- "${radix_files[@]}"
}

run_remote_suite() {
  local local_files=()
  local remote_files=()
  for run in $(seq 1 "$REPEATS"); do
    start_worker "remote-local-$run" pals radix 0 64
    local local_result="$RESULTS_DIR/remote-local-$run.json"
    run_benchmark "$local_result" remote-repeated-v1 \
      --batch-requests 1 --interactive-requests 0 \
      --shared-prefix-repeats 32 --batch-output-tokens 4 --no-warmup
    local_files+=("$local_result")

    start_worker "remote-seed-$run" pals radix 1 64
    run_benchmark "$RESULTS_DIR/.remote-seed-$run.json" remote-repeated-v1 \
      --batch-requests 1 --interactive-requests 0 \
      --shared-prefix-repeats 32 --batch-output-tokens 1 --no-warmup
    start_worker "remote-restore-$run" pals radix 1 64 1
    local remote_result="$RESULTS_DIR/remote-mooncake-$run.json"
    run_benchmark "$remote_result" remote-repeated-v1 \
      --batch-requests 1 --interactive-requests 0 \
      --shared-prefix-repeats 32 --batch-output-tokens 4 --no-warmup
    remote_files+=("$remote_result")
    rm -f "$RESULTS_DIR/.remote-seed-$run.json" \
      "$RESULTS_DIR/.remote-seed-$run.requests.csv"
  done
  aggregate_groups Local Mooncake remote-local-vs-mooncake \
    "${local_files[@]}" -- "${remote_files[@]}"
}

case "$SUITE" in
  scheduler) run_scheduler_suite ;;
  pressure) run_pressure_suite ;;
  prefix) run_prefix_suite ;;
  remote) run_remote_suite ;;
  *) echo "usage: $0 {scheduler|pressure|prefix|remote}" >&2; exit 2 ;;
esac
