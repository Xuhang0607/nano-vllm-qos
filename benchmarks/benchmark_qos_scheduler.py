"""Deterministic control-plane benchmark for nano-vLLM scheduling policies.

This simulator exercises the real Scheduler and BlockManager without loading a
model. Its timing model is intentionally simple, so results must not be reported
as GPU throughput measurements.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from nanovllm.engine.qos import RequestMetrics, RequestQoS, summarize_metrics
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class SimulatedClock:
    def __init__(self):
        self.now_ms = 0.0

    def __call__(self):
        return self.now_ms / 1000.0

    def advance_to(self, timestamp_ms: float):
        self.now_ms = max(self.now_ms, timestamp_ms)

    def advance(self, elapsed_ms: float):
        self.now_ms += elapsed_ms


@dataclass(slots=True, frozen=True)
class RequestSpec:
    name: str
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int
    qos: RequestQoS


def make_config(policy: str, prefix_cache_backend: str):
    return SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=128,
        eos=-1,
        kvcache_block_size=16,
        num_kvcache_blocks=4096,
        prefix_cache_backend=prefix_cache_backend,
        scheduling_policy=policy,
        qos_best_effort_slo_ms=60000.0,
        qos_priority_boost_ms=30.0,
        qos_aging_ms_per_step=1.0,
        qos_prefill_ms_per_token=0.08,
        qos_decode_ms_per_token=2.0,
        qos_ewma_alpha=0.2,
    )


def build_workload(batch_requests: int, interactive_requests: int):
    workload = []
    for index in range(batch_requests):
        workload.append(
            RequestSpec(
                name=f"batch-{index}",
                arrival_ms=0.0,
                prompt_tokens=256 + 16 * (index % 3),
                output_tokens=24,
                qos=RequestQoS(
                    priority=0,
                    e2e_slo_ms=1800.0,
                    request_class="batch",
                ),
            )
        )
    for index in range(interactive_requests):
        workload.append(
            RequestSpec(
                name=f"interactive-{index}",
                arrival_ms=10.0 + 20.0 * index,
                prompt_tokens=24 + 8 * (index % 3),
                output_tokens=8,
                qos=RequestQoS(
                    priority=4,
                    ttft_slo_ms=80.0,
                    tpot_slo_ms=10.0,
                    e2e_slo_ms=250.0,
                    request_class="interactive",
                ),
            )
        )
    return sorted(workload, key=lambda item: (item.arrival_ms, item.name))


def simulated_step_ms(seqs, is_prefill: bool):
    if is_prefill:
        tokens = sum(seq.num_scheduled_tokens for seq in seqs)
        return 0.3 + 0.08 * tokens
    return 1.5 + 0.15 * len(seqs)


def service_gain(metrics: list[RequestMetrics]):
    achieved = 0
    possible = 0
    for item in metrics:
        gain = item.priority + 1
        checks = [
            check
            for check in (
                item.ttft_slo_met,
                item.tpot_slo_met,
                item.e2e_slo_met,
            )
            if check is not None
        ]
        possible += gain
        if all(checks):
            achieved += gain
    return {
        "achieved": achieved,
        "possible": possible,
        "ratio": achieved / possible if possible else 0.0,
    }


def prompt_token_ids(spec: RequestSpec, request_index: int):
    shared_tokens = 16 if spec.qos.request_class == "interactive" else 32
    shared_prefix = list(range(100, 100 + shared_tokens))
    unique_tokens = spec.prompt_tokens - shared_tokens
    unique_start = 10000 + request_index * 1000
    return shared_prefix + list(range(unique_start, unique_start + unique_tokens))


def run_policy(
    policy: str,
    workload: list[RequestSpec],
    include_trace: bool = False,
    prefix_cache_backend: str = "radix",
):
    Sequence.block_size = 16
    clock = SimulatedClock()
    scheduler = Scheduler(make_config(policy, prefix_cache_backend), clock=clock)
    pending = list(workload)
    request_names = {}
    decisions = []

    while pending or not scheduler.is_finished():
        if scheduler.is_finished() and pending:
            clock.advance_to(pending[0].arrival_ms)

        while pending and pending[0].arrival_ms <= clock.now_ms:
            spec = pending.pop(0)
            request_index = len(request_names)
            sequence = Sequence(
                prompt_token_ids(spec, request_index),
                SamplingParams(max_tokens=spec.output_tokens, ignore_eos=True),
                qos=spec.qos,
                arrival_time=spec.arrival_ms / 1000.0,
            )
            request_names[sequence.seq_id] = spec.name
            scheduler.add(sequence)

        seqs, is_prefill = scheduler.schedule()
        if include_trace:
            decisions.append(
                {
                    "time_ms": round(clock.now_ms, 3),
                    "phase": "prefill" if is_prefill else "decode",
                    "requests": [request_names[seq.seq_id] for seq in seqs],
                    "budgets_ms": [
                        round(seq.last_budget_ms, 3)
                        if seq.last_budget_ms is not None
                        else None
                        for seq in seqs
                    ],
                }
            )
        elapsed_ms = simulated_step_ms(seqs, is_prefill)
        observed_tokens = (
            sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else len(seqs)
        )
        clock.advance(elapsed_ms)
        scheduler.observe_execution(is_prefill, observed_tokens, elapsed_ms)
        scheduler.postprocess(seqs, [200000 + seq.seq_id for seq in seqs], is_prefill)

    metrics = scheduler.completed_metrics
    classes = sorted({item.request_class for item in metrics})
    result = {
        "policy": policy,
        "simulated_makespan_ms": round(clock.now_ms, 3),
        "overall": summarize_metrics(metrics),
        "by_class": {
            name: summarize_metrics(
                [item for item in metrics if item.request_class == name]
            )
            for name in classes
        },
        "service_gain": service_gain(metrics),
        "estimator": {
            "prefill_ms_per_token": scheduler.estimator.prefill_ms_per_token,
            "decode_ms_per_token": scheduler.estimator.decode_ms_per_token,
        },
        "prefix_cache": scheduler.block_manager.cache_metrics(),
        "requests": [item.to_dict() for item in metrics],
    }
    if include_trace:
        result["decision_trace"] = decisions
    return result


def percent_change(before: float, after: float):
    if before == 0:
        return 0.0
    return (after - before) / before * 100.0


def print_report(results):
    fcfs = results["fcfs"]
    pals = results["pals"]
    print("nano-vLLM QoS scheduler control-plane simulation")
    print("WARNING: simulated timing; this is not a GPU throughput benchmark.\n")
    print(
        f"{'Policy':<8} {'Class':<12} {'TTFT p95':>10} {'TPOT p95':>10} "
        f"{'E2E p95':>10} {'TTFT SLO':>10} {'TPOT SLO':>10} {'E2E SLO':>10}"
    )
    for policy_result in (fcfs, pals):
        for class_name, summary in policy_result["by_class"].items():
            ttft = summary["ttft_slo_attainment"]
            tpot = summary["tpot_slo_attainment"]
            e2e = summary["e2e_slo_attainment"]
            print(
                f"{policy_result['policy']:<8} {class_name:<12} "
                f"{summary['ttft_ms_p95']:>10.2f} {summary['tpot_ms_p95']:>10.2f} "
                f"{summary['e2e_ms_p95']:>10.2f} "
                f"{('-' if ttft is None else f'{ttft:.1%}'):>10} "
                f"{('-' if tpot is None else f'{tpot:.1%}'):>10} "
                f"{('-' if e2e is None else f'{e2e:.1%}'):>10}"
            )

    fcfs_interactive = fcfs["by_class"]["interactive"]
    pals_interactive = pals["by_class"]["interactive"]
    ttft_delta = percent_change(
        fcfs_interactive["ttft_ms_p95"], pals_interactive["ttft_ms_p95"]
    )
    gain_delta = percent_change(
        fcfs["service_gain"]["ratio"], pals["service_gain"]["ratio"]
    )
    print("\nComparison")
    print(f"  interactive TTFT p95 change: {ttft_delta:+.1f}%")
    print(f"  weighted service-gain ratio change: {gain_delta:+.1f}%")
    print(f"  FCFS makespan: {fcfs['simulated_makespan_ms']:.2f} ms")
    print(f"  PALS makespan: {pals['simulated_makespan_ms']:.2f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-requests", type=int, default=12)
    parser.add_argument("--interactive-requests", type=int, default=12)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--include-trace", action="store_true")
    parser.add_argument(
        "--prefix-cache-backend", choices=("hash", "radix"), default="radix"
    )
    args = parser.parse_args()
    if args.batch_requests <= 0 or args.interactive_requests <= 0:
        parser.error("request counts must be positive")
    workload = build_workload(args.batch_requests, args.interactive_requests)
    results = {
        "metadata": {
            "benchmark": "deterministic control-plane simulation",
            "gpu_measurement": False,
            "batch_requests": args.batch_requests,
            "interactive_requests": args.interactive_requests,
            "prefix_cache_backend": args.prefix_cache_backend,
            "timing_model": {
                "prefill_ms": "0.3 + 0.08 * scheduled_tokens",
                "decode_ms": "1.5 + 0.15 * batch_size",
            },
        }
    }
    results.update(
        {
            policy: run_policy(
                policy,
                workload,
                args.include_trace,
                args.prefix_cache_backend,
            )
            for policy in ("fcfs", "pals")
        }
    )
    print_report(results)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nDetailed JSON: {args.output_json}")


if __name__ == "__main__":
    main()
