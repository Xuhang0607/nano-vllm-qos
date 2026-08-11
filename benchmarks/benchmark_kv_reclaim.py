"""Deterministic KV-pressure benchmark for preemption reclaim policies.

This benchmark exercises the real Scheduler and BlockManager without loading a
model. Its timing model is synthetic; results describe control-plane behavior,
not GPU latency or throughput.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.kv_reclaim import create_kv_reclaim_policy
from nanovllm.engine.qos import RequestQoS
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class SimulatedClock:
    def __init__(self):
        self.now_ms = 0.0

    def __call__(self):
        return self.now_ms / 1000.0

    def advance(self, elapsed_ms: float):
        self.now_ms += elapsed_ms


def make_config(reclaim_policy: str):
    return SimpleNamespace(
        max_num_seqs=3,
        max_num_batched_tokens=12,
        max_model_len=64,
        requested_max_model_len=64,
        eos=-1,
        kvcache_block_size=4,
        num_kvcache_blocks=11,
        prefix_cache_backend="hash",
        scheduling_policy="pals",
        qos_best_effort_slo_ms=1000.0,
        qos_priority_boost_ms=20.0,
        qos_aging_ms_per_step=1.0,
        qos_prefill_ms_per_token=0.1,
        qos_decode_ms_per_token=1.0,
        qos_ewma_alpha=0.2,
        kv_reclaim_policy=reclaim_policy,
        kv_reclaim_min_keep_ratio=0.0,
        kv_reclaim_max_keep_ratio=0.75,
        kv_reclaim_budget_scale_ms=1000.0,
        kv_reclaim_target_free_blocks=2,
    )


def step_ms(seqs, is_prefill: bool):
    if is_prefill:
        return 0.2 + 0.1 * sum(seq.num_scheduled_tokens for seq in seqs)
    return 1.0 + 0.1 * len(seqs)


def run_policy(reclaim_policy: str):
    Sequence.block_size = 4
    clock = SimulatedClock()
    scheduler = Scheduler(make_config(reclaim_policy), clock=clock)
    for index in range(3):
        scheduler.add(
            Sequence(
                list(range(index * 100, index * 100 + 12)),
                SamplingParams(max_tokens=8, ignore_eos=True),
                RequestQoS(
                    priority=3 - index,
                    ttft_slo_ms=10.0,
                    e2e_slo_ms=80.0,
                    request_class="interactive",
                ),
            )
        )

    steps = 0
    while not scheduler.is_finished():
        steps += 1
        if steps > 500:
            raise RuntimeError("KV reclaim benchmark failed to make progress")
        seqs, is_prefill = scheduler.schedule()
        elapsed_ms = step_ms(seqs, is_prefill)
        tokens = (
            sum(seq.num_scheduled_tokens for seq in seqs)
            if is_prefill
            else len(seqs)
        )
        clock.advance(elapsed_ms)
        scheduler.observe_execution(is_prefill, tokens, elapsed_ms)
        scheduler.postprocess(
            seqs,
            [100000 + steps * 10 + index for index, _ in enumerate(seqs)],
            is_prefill,
        )

    return {
        "policy": reclaim_policy,
        "simulated_makespan_ms": round(clock.now_ms, 3),
        "steps": steps,
        "metrics": scheduler.metrics(),
    }


def allocate_and_fill(manager: BlockManager, sequence: Sequence):
    cached_blocks = manager.can_allocate(sequence)
    if cached_blocks == -1:
        raise RuntimeError("eviction probe cannot allocate request")
    manager.allocate(sequence, cached_blocks)
    sequence.num_scheduled_tokens = sequence.num_tokens - sequence.num_cached_tokens
    manager.hash_blocks(sequence)
    sequence.num_cached_tokens += sequence.num_scheduled_tokens
    sequence.num_scheduled_tokens = 0


def run_eviction_probe(reclaim_policy: str):
    """Measure whether a retained prefix survives intervening block reuse."""
    Sequence.block_size = 4
    config = make_config(reclaim_policy)
    manager = BlockManager(num_blocks=6, block_size=4)
    victim = Sequence(
        list(range(12)),
        SamplingParams(max_tokens=1, ignore_eos=True),
    )
    allocate_and_fill(manager, victim)
    policy = create_kv_reclaim_policy(reclaim_policy, config)
    decision = policy.plan(victim, budget_ms=-1.0, min_reclaim_blocks=2)
    manager.reclaim_suffix(victim, decision.keep_blocks)

    pressure = Sequence(
        list(range(100, 120)),
        SamplingParams(max_tokens=1, ignore_eos=True),
    )
    allocate_and_fill(manager, pressure)
    manager.deallocate(pressure)
    overwrite = Sequence(
        list(range(200, 204)),
        SamplingParams(max_tokens=1, ignore_eos=True),
    )
    allocate_and_fill(manager, overwrite)
    manager.deallocate(overwrite)

    surviving_blocks = manager.cached_prefix_blocks(victim)
    return {
        "policy": reclaim_policy,
        "initial_cached_tokens": 12,
        "explicitly_retained_blocks": decision.keep_blocks,
        "surviving_cached_blocks": surviving_blocks,
        "surviving_cached_tokens": surviving_blocks * victim.block_size,
        "tokens_to_recompute": victim.num_tokens
        - surviving_blocks * victim.block_size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    results = {
        "metadata": {
            "benchmark": "deterministic KV-pressure control-plane simulation",
            "gpu_measurement": False,
            "num_requests": 3,
            "num_kvcache_blocks": 11,
            "block_size": 4,
        },
        "recompute": run_policy("recompute"),
        "slo_aware": run_policy("slo_aware"),
        "eviction_probe": {
            "recompute": run_eviction_probe("recompute"),
            "slo_aware": run_eviction_probe("slo_aware"),
        },
    }
    print(json.dumps(results, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
