"""Summarize captured incident windows, not whole-run performance statistics."""

import argparse
import json
from collections import Counter
from pathlib import Path


def analyze(rows):
    header, *events = rows
    if header.get("schema_version") != 1 or header.get("event") != "header":
        raise ValueError("unsupported scheduler trace header")
    pressures = [row for row in events if row["event"] == "append_pressure"]
    first = pressures[0] if pressures else None
    preceding = []
    if first:
        preceding = [row for row in events if row["event_id"] < first["event_id"]
                     and row["event"] in ("prefill_admitted", "prefill_blocked", "execution")][-12:]
    victims = Counter(str(row["victim_id"]) for row in pressures)
    return {
        "scope": "bounded diagnostic windows; not a throughput or whole-run preemption measurement",
        "configuration": header,
        "captured_events": len(events),
        "captured_append_pressure_events": len(pressures),
        "victim_counts": dict(victims),
        "first_pressure": first,
        "preceding_events": preceding,
        "captured_preemptions": [row for row in events if row["event"] == "preempted"],
        "note": "Header-only means no incident was captured, not proof that all scheduling was optimal.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.trace.open(encoding="utf-8") as source:
        report = analyze([json.loads(line) for line in source])
    payload = json.dumps(report, indent=2)
    if args.output:
        with args.output.open("x", encoding="utf-8") as output:
            output.write(payload + "\n")
    else:
        print(payload)


if __name__ == "__main__":
    main()
