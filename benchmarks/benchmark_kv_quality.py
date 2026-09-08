"""Live Needle-in-a-Haystack quality benchmark for KV compression policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(slots=True, frozen=True)
class NeedleCase:
    case_id: str
    needle_fraction: float
    expected: str
    prompt: str
    task: str = "literal"
    filler_lines: int = 96


def build_needle_case(case_id: str, needle_fraction: float, filler_lines: int = 96):
    if not 0 <= needle_fraction <= 1:
        raise ValueError("needle fraction must be between 0 and 1")
    if filler_lines < 8:
        raise ValueError("at least eight filler lines are required")
    expected = f"ZXQ-{case_id.upper()}-7419"
    needle = (
        f"IMPORTANT: the access code for project ORION-{case_id} is {expected}. "
        f"Remember the access code for project ORION-{case_id}."
    )
    fillers = [
        f"Record {case_id}-{index:03d}: routine telemetry is nominal and contains no access code."
        for index in range(filler_lines)
    ]
    position = round(needle_fraction * len(fillers))
    records = fillers[:position] + [needle] + fillers[position:]
    question = (
        f"/no_think\nWhat is the access code for project ORION-{case_id}? "
        "Answer with only the code."
    )
    return NeedleCase(
        case_id=case_id,
        needle_fraction=needle_fraction,
        expected=expected,
        prompt="\n".join([*records, "", question]),
    )


def normalize_answer(value: str):
    return "".join(character for character in value.upper() if character.isalnum())


def final_answer_text(answer: str):
    if "</think>" in answer:
        return answer.rsplit("</think>", 1)[1].strip()
    if "<think>" in answer:
        return ""
    return answer.strip()


def answer_matches(answer: str, expected: str):
    return normalize_answer(final_answer_text(answer)) == normalize_answer(expected)


def build_quality_cases(filler_lines: int, repeats_per_position: int = 1):
    if repeats_per_position < 1:
        raise ValueError("repeats per position must be positive")
    cases = []
    for label, fraction in (("early", 0.2), ("middle", 0.5), ("late", 0.8)):
        for repeat in range(repeats_per_position):
            case_id = label if repeats_per_position == 1 else f"{label}-{repeat}"
            cases.append(build_needle_case(case_id, fraction, filler_lines))
    return cases


def build_quality_matrix(lengths=(48, 96, 144), repeats=6, seed=42):
    """Synthetic retrieval probes with random answers and held-out case seeds."""
    if repeats < 1 or not lengths or any(length < 8 for length in lengths):
        raise ValueError("matrix needs positive repeats and lengths >= 8")
    rng = random.Random(seed)
    cases = []
    for length in lengths:
        for fraction in (0.2, 0.5, 0.8):
            for task in ("literal", "paraphrase", "distractors", "two_hop"):
                for repeat in range(repeats):
                    case_id = f"{seed}-{task}-{length}-{fraction}-{repeat}"
                    entity = f"P{rng.randrange(100000, 999999)}"
                    alias = f"A{rng.randrange(100000, 999999)}"
                    expected = f"{rng.randrange(10000000, 99999999)}"
                    records = [
                        f"Log {index}: project P{rng.randrange(100000, 999999)} "
                        f"has status {rng.choice(('pending', 'archived', 'reviewed'))}; "
                        f"reference {rng.randrange(10000000, 99999999)}."
                        for index in range(length)
                    ]
                    fact = f"The current access code for project {entity} is {expected}."
                    question = f"What is the current access code for project {entity}?"
                    if task == "paraphrase":
                        fact = f"Project {entity} uses {expected} as its entry credential."
                        question = f"Which numeric password unlocks {entity}?"
                    elif task == "distractors":
                        for index in range(0, length, 4):
                            records[index] = (
                                f"Obsolete access code for project {entity}: "
                                f"{rng.randrange(10000000, 99999999)}. It is revoked."
                            )
                    elif task == "two_hop":
                        records[int((1.0 - fraction) * (length - 1))] = (
                            f"Department {alias} is assigned to project {entity}."
                        )
                        question = f"What is the access code of the project assigned to department {alias}?"
                    records.insert(round(fraction * length), fact)
                    prompt = "\n".join(records + ["", "/no_think", question,
                                                     "Answer with only the numeric code."])
                    cases.append(NeedleCase(case_id, fraction, expected, prompt, task, length))
    rng.shuffle(cases)
    return cases


def quality_fingerprint(cases):
    payload = json.dumps([asdict(case) for case in cases], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def retained_page_budget(metrics):
    if metrics.get("kv_compression_policy") == "none":
        return None
    budget = metrics["kv_compression_sink_blocks"] + metrics["kv_compression_recent_blocks"]
    if metrics.get("kv_compression_policy") == "query_aware":
        budget += metrics["kv_compression_importance_blocks"]
    return budget


def request_json(url, payload=None, timeout_s=300.0):
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def run_quality_benchmark(
    base_url: str,
    model: str,
    cases: list[NeedleCase],
    max_tokens: int = 128,
    timeout_s: float = 300.0,
):
    base_url = base_url.rstrip("/")
    server_metrics = request_json(f"{base_url}/v1/metrics", timeout_s=timeout_s)
    records = []
    for index, case in enumerate(cases):
        started = time.perf_counter()
        response = request_json(
            f"{base_url}/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": case.prompt}],
                "temperature": 0.1,
                "max_tokens": max_tokens,
                "stream": False,
                "priority": 10,
                "request_class": "kv_quality",
                "ttft_slo_ms": 10000,
                "tpot_slo_ms": 1000,
                "e2e_slo_ms": 30000,
            },
            timeout_s,
        )
        answer = response["choices"][0]["message"]["content"]
        records.append(
            {
                "case_id": case.case_id,
                "task": case.task,
                "filler_lines": case.filler_lines,
                "needle_fraction": case.needle_fraction,
                "expected": case.expected,
                "answer": answer,
                "correct": answer_matches(answer, case.expected),
                "client_e2e_ms": (time.perf_counter() - started) * 1000.0,
                "usage": response.get("usage", {}),
                "metrics": response.get("x_nanovllm_metrics", {}),
                "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
            }
        )
        if (index + 1) % 12 == 0:
            print(f"quality progress {index + 1}/{len(cases)}", flush=True)
    correct = sum(record["correct"] for record in records)
    dropped_tokens = sum(
        record["metrics"].get("kv_compression_dropped_tokens", 0)
        for record in records
    )
    prompt_tokens = sum(record["usage"].get("prompt_tokens", 0) for record in records)
    return {
        "metadata": {
            "benchmark": "live KV compression Needle-in-a-Haystack",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "thinking_mode": "disabled",
            "answer_match": "strict_final_answer",
            "kv_compression_policy": server_metrics.get("kv_compression_policy"),
            "kv_compression_sink_blocks": server_metrics.get(
                "kv_compression_sink_blocks"
            ),
            "kv_compression_recent_blocks": server_metrics.get(
                "kv_compression_recent_blocks"
            ),
            "kv_compression_importance_blocks": server_metrics.get(
                "kv_compression_importance_blocks"
            ),
            "kv_compression_query_tokens": server_metrics.get(
                "kv_compression_query_tokens"
            ),
            "kvcache_block_size": server_metrics.get("kvcache_block_size"),
            "cases": len(cases),
            "case_fingerprint": quality_fingerprint(cases),
            "retained_page_budget": retained_page_budget(server_metrics),
            "kv_compression_trigger_free_ratio": server_metrics.get("kv_compression_trigger_free_ratio"),
            "num_kvcache_blocks": server_metrics.get("num_kvcache_blocks"),
            "temperature": 0.1,
            "max_tokens": max_tokens,
        },
        "summary": {
            "correct": correct,
            "cases": len(records),
            "accuracy": correct / len(records) if records else 0.0,
            "prompt_tokens": prompt_tokens,
            "compression_dropped_tokens": dropped_tokens,
            "compression_drop_ratio": (
                dropped_tokens / prompt_tokens if prompt_tokens else 0.0
            ),
        },
        "cases": records,
        "by_task": {
            task: {
                "cases": sum(record["task"] == task for record in records),
                "correct": sum(record["correct"] for record in records if record["task"] == task),
            }
            for task in sorted({record["task"] for record in records})
        },
    }


def print_report(results):
    metadata = results["metadata"]
    summary = results["summary"]
    print(
        f"policy={metadata['kv_compression_policy']} "
        f"accuracy={summary['correct']}/{summary['cases']} "
        f"({summary['accuracy']:.1%}) "
        f"drop_ratio={summary['compression_drop_ratio']:.1%}"
    )
    for record in results["cases"]:
        print(
            f"{record['case_id']:<8} position={record['needle_fraction']:.0%} "
            f"correct={record['correct']} answer={record['answer']!r}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8020")
    parser.add_argument("--model", default="Qwen3-0.6B")
    parser.add_argument("--filler-lines", type=int, default=96)
    parser.add_argument("--repeats-per-position", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--lengths", type=int, nargs="+", default=[48, 96, 144])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cases = (build_quality_matrix(args.lengths, args.repeats_per_position, args.seed)
             if args.matrix else build_quality_cases(args.filler_lines, args.repeats_per_position))
    results = run_quality_benchmark(
        args.base_url,
        args.model,
        cases,
        args.max_tokens,
        args.timeout_s,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print_report(results)
    print(f"JSON: {args.output_json}")


if __name__ == "__main__":
    main()
