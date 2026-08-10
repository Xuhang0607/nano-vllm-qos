"""Control-plane simulation for transfer-vs-recompute KV cache decisions."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from nanovllm.engine.hierarchical_cache import (
    CacheTier,
    KVAccessPlan,
    KVCacheGeometry,
    KVTransferPlanner,
    StorageTierProfile,
)


@dataclass(slots=True, frozen=True)
class CacheRequest:
    total_tokens: int
    local_cached_tokens: int
    remote_cached_tokens: int
    prefill_ms_per_token: float
    congestion_multiplier: float


def build_workload(num_requests: int):
    prompt_lengths = (512, 1024, 2048, 4096)
    local_ratios = (0.0, 0.125, 0.25)
    remote_ratios = (0.4, 0.7, 0.9)
    prefill_costs = (0.04, 0.06, 0.1)
    congestion = (1.0, 2.0, 8.0, 20.0)
    workload = []
    for index in range(num_requests):
        total = prompt_lengths[index % len(prompt_lengths)]
        local = int(total * local_ratios[index % len(local_ratios)])
        remote = max(
            local,
            int(total * remote_ratios[(index // len(local_ratios)) % len(remote_ratios)]),
        )
        workload.append(
            CacheRequest(
                total_tokens=total,
                local_cached_tokens=local,
                remote_cached_tokens=remote,
                prefill_ms_per_token=prefill_costs[index % len(prefill_costs)],
                congestion_multiplier=congestion[index % len(congestion)],
            )
        )
    return workload


def make_planner(congestion_multiplier: float):
    return KVTransferPlanner(
        KVCacheGeometry(28, 8, 128, 2),
        {
            CacheTier.MOONCAKE: StorageTierProfile(
                bandwidth_gbps=12.5,
                fixed_latency_ms=0.3,
                congestion_multiplier=congestion_multiplier,
            )
        },
    )


def recompute_plan(request: CacheRequest):
    return make_planner(request.congestion_multiplier).plan(
        total_tokens=request.total_tokens,
        local_cached_tokens=request.local_cached_tokens,
        cached_tokens_by_tier={},
        prefill_ms_per_token=request.prefill_ms_per_token,
    )


def always_restore_plan(request: CacheRequest):
    if request.remote_cached_tokens <= request.local_cached_tokens:
        return recompute_plan(request)
    planner = make_planner(request.congestion_multiplier)
    restored_tokens = request.remote_cached_tokens - request.local_cached_tokens
    transfer_bytes = planner.geometry.size_bytes(restored_tokens)
    transfer_ms = planner.tier_profiles[CacheTier.MOONCAKE].transfer_ms(transfer_bytes)
    recomputed_tokens = request.total_tokens - request.remote_cached_tokens
    recompute_ms = recomputed_tokens * request.prefill_ms_per_token
    return KVAccessPlan(
        action="restore",
        source_tier=CacheTier.MOONCAKE,
        local_cached_tokens=request.local_cached_tokens,
        source_cached_tokens=request.remote_cached_tokens,
        restored_tokens=restored_tokens,
        recomputed_tokens=recomputed_tokens,
        transfer_bytes=transfer_bytes,
        transfer_ms=transfer_ms,
        recompute_ms=recompute_ms,
        total_ms=transfer_ms + recompute_ms,
    )


def adaptive_plan(request: CacheRequest):
    return make_planner(request.congestion_multiplier).plan(
        total_tokens=request.total_tokens,
        local_cached_tokens=request.local_cached_tokens,
        cached_tokens_by_tier={CacheTier.MOONCAKE: request.remote_cached_tokens},
        prefill_ms_per_token=request.prefill_ms_per_token,
    )


def summarize(plans):
    return {
        "requests": len(plans),
        "estimated_total_ms": sum(plan.total_ms for plan in plans),
        "estimated_average_ms": sum(plan.total_ms for plan in plans) / len(plans),
        "restore_requests": sum(plan.action == "restore" for plan in plans),
        "recompute_requests": sum(plan.action == "recompute" for plan in plans),
        "local_hit_requests": sum(plan.action == "local" for plan in plans),
        "transfer_gib": sum(plan.transfer_bytes for plan in plans) / 1024**3,
        "recomputed_tokens": sum(plan.recomputed_tokens for plan in plans),
    }


def run_benchmark(num_requests: int):
    workload = build_workload(num_requests)
    strategies = {
        "local_recompute": [recompute_plan(request) for request in workload],
        "always_restore": [always_restore_plan(request) for request in workload],
        "cost_aware": [adaptive_plan(request) for request in workload],
    }
    return {
        "metadata": {
            "benchmark": "deterministic transfer-vs-recompute simulation",
            "gpu_measurement": False,
            "requests": num_requests,
            "geometry": {
                "layers": 28,
                "kv_heads": 8,
                "head_dim": 128,
                "dtype_bytes": 2,
            },
            "mooncake_bandwidth_gbps": 12.5,
        },
        "strategies": {
            name: summarize(plans) for name, plans in strategies.items()
        },
    }


def print_report(results):
    print("Hierarchical KV cache control-plane simulation")
    print("WARNING: modeled transfer/prefill time; this is not a hardware benchmark.\n")
    print(
        f"{'Strategy':<18} {'Avg ms':>10} {'Restore':>10} "
        f"{'Transfer GiB':>14} {'Recompute tok':>15}"
    )
    for name, summary in results["strategies"].items():
        print(
            f"{name:<18} {summary['estimated_average_ms']:>10.2f} "
            f"{summary['restore_requests']:>10} {summary['transfer_gib']:>14.2f} "
            f"{summary['recomputed_tokens']:>15}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=120)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.requests <= 0:
        parser.error("requests must be positive")
    results = run_benchmark(args.requests)
    print_report(results)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nDetailed JSON: {args.output_json}")


if __name__ == "__main__":
    main()
