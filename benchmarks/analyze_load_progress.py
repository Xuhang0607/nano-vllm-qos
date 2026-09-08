"""Reconstruct client outstanding work; this is not the engine's active queue."""

import argparse
import json
import math
from pathlib import Path

from benchmarks.benchmark_serving_gpu import percentile


def analyze_run(run, window_s=60):
    if not math.isfinite(window_s) or window_s <= 0:
        raise ValueError("window_s must be positive and finite")
    records = run["requests"]
    if not records:
        raise ValueError("request records are required")
    end_ms = max(record["arrival_ms"] for record in records)
    if end_ms <= 0:
        raise ValueError("an arrival interval is required")
    terminal = []
    for record in records:
        elapsed = record.get("scheduled_e2e_ms")
        terminal.append(record["arrival_ms"] + elapsed
                        if isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and elapsed >= 0
                        else None)
    bucket_ms = window_s * 1000
    count = math.ceil(end_ms / bucket_ms)
    windows = []
    for index in range(count):
        left, right = index * bucket_ms, min((index + 1) * bucket_ms, end_ms)
        cohort = [record for record in records
                  if left <= record["arrival_ms"] < right
                  or (index == count - 1 and record["arrival_ms"] == right)]
        arrived = sum(record["arrival_ms"] <= right for record in records)
        ended = sum(value is not None and value <= right for value in terminal)
        unknown_arrived = sum(value is None and record["arrival_ms"] <= right
                              for record, value in zip(records, terminal))
        queue = [(record.get("metrics") or {}).get("queue_ms") for record in cohort]
        queue = [value for value in queue if value is not None and math.isfinite(value)]
        windows.append({"end_s": right / 1000, "cohort_requests": len(cohort),
                        "cohort_failures": sum(not record["success"] for record in cohort),
                        "client_outstanding": arrived - ended if not unknown_arrived else None,
                        "unknown_terminal_times": unknown_arrived, "queue_samples": len(queue),
                        "queue_p50_ms": percentile(queue, 0.5) if queue else None,
                        "queue_p95_ms": percentile(queue, 0.95) if queue else None})
    return {"scope": __doc__, "mode": run["metadata"].get("device_context_mode"),
            "requests": len(records), "failures": sum(not record["success"] for record in records),
            "arrival_window_s": end_ms / 1000,
            "drain_s": max(0, (max(terminal) - end_ms) / 1000) if all(v is not None for v in terminal) else None,
            "windows": windows,
            "warning": "Timeout ends a client wait, not necessarily backend work; no stability guarantee"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist")
    reports = [{"input": str(path), **analyze_run(json.loads(path.read_text(encoding="utf-8")))}
               for path in args.inputs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(reports, stream, ensure_ascii=False, indent=2)
    print(json.dumps(reports, ensure_ascii=False))


if __name__ == "__main__":
    main()
