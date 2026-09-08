"""Validate Nsight SQLite evidence before drawing GPU timing conclusions."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3


def audit_capture(path):
    # Read-only URI prevents accidentally creating an empty database on a typo.
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        counts = {}
        for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_KERNEL",
                      "CUPTI_ACTIVITY_KIND_MEMCPY", "NVTX_EVENTS"):
            counts[table] = db.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] if table in tables else 0
        diagnostics = ([dict(zip(("severity", "text"), row)) for row in
                        db.execute("SELECT severity, text FROM DIAGNOSTIC_EVENT")]
                       if "DIAGNOSTIC_EVENT" in tables else [])
        runtime = []
        if counts["CUPTI_ACTIVITY_KIND_RUNTIME"] and "StringIds" in tables:
            runtime = [{"name": name, "calls": calls, "inclusive_host_ms": ns / 1e6}
                       for name, calls, ns in db.execute(
                           "SELECT s.value, COUNT(*), SUM(r.end-r.start) "
                           "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId "
                           "GROUP BY s.value ORDER BY SUM(r.end-r.start) DESC LIMIT 20")]
        kernel_present = counts["CUPTI_ACTIVITY_KIND_KERNEL"] > 0
        return {"scope": "capture integrity, not an inference benchmark",
                "sqlite": str(path), "event_counts": counts,
                "gpu_kernel_data_present": kernel_present,
                "gpu_timing_status": "requires timeline review" if kernel_present else "unavailable",
                "warning": "API durations are host-side and may overlap; never substitute for GPU time",
                "diagnostics": diagnostics, "top_cuda_apis": runtime}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-kernels", action="store_true",
                        help="Exit 2 after writing audit if kernel events are missing")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist")
    report = audit_capture(args.sqlite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key != "diagnostics"}))
    if args.require_kernels and not report["gpu_kernel_data_present"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
