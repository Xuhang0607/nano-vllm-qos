from types import SimpleNamespace

import pytest

from benchmarks.benchmark_kv_reclaim import run_eviction_probe
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.kv_reclaim import (
    KVReclaimPolicy,
    SLOAwareKVReclaimPolicy,
)
from nanovllm.engine.qos import RequestQoS
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def make_sequence(qos=None):
    return Sequence(
        list(range(12)),
        SamplingParams(max_tokens=4, ignore_eos=True),
        qos=qos,
    )


def make_scheduler_config():
    return SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=16,
        max_model_len=64,
        requested_max_model_len=64,
        eos=-1,
        kvcache_block_size=4,
        num_kvcache_blocks=8,
        prefix_cache_backend="hash",
        scheduling_policy="pals",
        qos_best_effort_slo_ms=1000.0,
        qos_priority_boost_ms=20.0,
        qos_aging_ms_per_step=5.0,
        qos_prefill_ms_per_token=1.0,
        qos_decode_ms_per_token=10.0,
        qos_ewma_alpha=0.2,
        kv_reclaim_policy="slo_aware",
        kv_reclaim_min_keep_ratio=0.0,
        kv_reclaim_max_keep_ratio=0.75,
        kv_reclaim_budget_scale_ms=1000.0,
        kv_reclaim_target_free_blocks=1,
    )


def allocate_complete_prefix(manager, sequence):
    assert manager.can_allocate(sequence) == 0
    manager.allocate(sequence, 0)
    sequence.num_cached_tokens = sequence.num_tokens


def test_slo_aware_reclaim_retains_more_kv_for_urgent_request():
    Sequence.block_size = 4
    sequence = make_sequence()
    sequence.block_table = [0, 1, 2]
    sequence.num_cached_tokens = 12
    policy = SLOAwareKVReclaimPolicy(0.0, 0.75, 1000.0)

    urgent = policy.plan(sequence, budget_ms=-1.0)
    relaxed = policy.plan(sequence, budget_ms=9000.0)

    assert urgent.urgency == 1.0
    assert urgent.keep_blocks == 2
    assert urgent.reclaim_blocks == 1
    assert urgent.invalidated_tokens == 4
    assert relaxed.urgency == pytest.approx(0.1)
    assert relaxed.keep_blocks == 0
    assert relaxed.reclaim_blocks == 3


def test_recompute_policy_reclaims_every_owned_block():
    Sequence.block_size = 4
    sequence = make_sequence()
    sequence.block_table = [0, 1, 2]
    sequence.num_cached_tokens = 12

    decision = KVReclaimPolicy().plan(sequence, budget_ms=-100.0)

    assert decision.keep_blocks == 0
    assert decision.reclaim_blocks == 3
    assert decision.invalidated_tokens == 12


def test_block_manager_reclaims_suffix_and_resumes_request():
    Sequence.block_size = 4
    manager = BlockManager(num_blocks=8, block_size=4)
    sequence = make_sequence()
    allocate_complete_prefix(manager, sequence)
    original_prefix = sequence.block_table[:2]

    freed = manager.reclaim_suffix(sequence, keep_blocks=2)

    assert freed == 1
    assert sequence.block_table == original_prefix
    assert sequence.num_cached_tokens == 8
    assert manager.can_resume(sequence)

    manager.resume(sequence)
    assert sequence.block_table[:2] == original_prefix
    assert len(sequence.block_table) == sequence.num_blocks
    assert sequence.num_cached_tokens == 8


def test_radix_resume_reattaches_cached_suffix_before_allocating_new_blocks():
    Sequence.block_size = 4
    manager = BlockManager(num_blocks=8, block_size=4, prefix_cache_backend="radix")
    sequence = make_sequence()
    assert manager.can_allocate(sequence) == 0
    manager.allocate(sequence, 0)
    sequence.num_scheduled_tokens = sequence.num_tokens
    manager.hash_blocks(sequence)
    sequence.num_cached_tokens = sequence.num_tokens
    sequence.num_scheduled_tokens = 0
    original_blocks = sequence.block_table.copy()

    manager.reclaim_suffix(sequence, keep_blocks=1)

    assert manager.can_resume(sequence)
    reused_tokens = manager.resume(sequence)
    assert reused_tokens == 4
    assert sequence.block_table[:2] == original_blocks[:2]
    assert sequence.num_cached_tokens == 8
    assert len(sequence.block_table) == sequence.num_blocks


def test_scheduler_preemption_uses_slo_aware_reclaim_and_records_metrics():
    Sequence.block_size = 4
    scheduler = Scheduler(make_scheduler_config(), clock=lambda: 0.0)
    sequence = make_sequence(RequestQoS(ttft_slo_ms=1.0))
    scheduler.add(sequence)
    allocate_complete_prefix(scheduler.block_manager, sequence)
    sequence.status = SequenceStatus.RUNNING

    scheduler.preempt(sequence)

    assert sequence.status == SequenceStatus.WAITING
    assert sequence.preemption_count == 1
    assert len(sequence.block_table) == 2
    assert sequence.num_cached_tokens == 8
    assert sequence.kv_invalidated_tokens == 4
    metrics = scheduler.metrics()
    assert metrics["kv_reclaim_policy"] == "slo_aware"
    assert metrics["kv_reclaim_events"] == 1
    assert metrics["kv_reclaim_freed_blocks"] == 1
    assert metrics["kv_reclaim_retained_blocks"] == 2
    assert metrics["kv_reclaim_invalidated_tokens"] == 4


def test_slo_aware_retained_prefix_survives_intervening_block_reuse():
    recompute = run_eviction_probe("recompute")
    slo_aware = run_eviction_probe("slo_aware")

    assert recompute["surviving_cached_tokens"] == 0
    assert slo_aware["surviving_cached_tokens"] == 4
    assert slo_aware["tokens_to_recompute"] < recompute["tokens_to_recompute"]


def test_scheduler_forces_full_reclaim_when_retained_prefixes_block_resume():
    Sequence.block_size = 4
    config = make_scheduler_config()
    config.num_kvcache_blocks = 4
    scheduler = Scheduler(config, clock=lambda: 0.0)
    selected = make_sequence(RequestQoS(ttft_slo_ms=1.0))
    selected.token_ids = list(range(12))
    selected.num_tokens = 12
    selected.num_prompt_tokens = 12
    scheduler.add(selected)
    allocate_complete_prefix(scheduler.block_manager, selected)
    scheduler.block_manager.reclaim_suffix(selected, keep_blocks=1)

    other = Sequence(
        list(range(100, 108)),
        SamplingParams(max_tokens=1, ignore_eos=True),
        RequestQoS(ttft_slo_ms=1000.0),
    )
    scheduler.add(other)
    allocate_complete_prefix(scheduler.block_manager, other)

    assert not scheduler.block_manager.can_resume(selected)
    scheduler._ensure_resume_capacity(selected, now=0.0)

    assert scheduler.block_manager.can_resume(selected)
    assert not other.block_table
    assert scheduler.metrics()["kv_reclaim_forced_fallbacks"] == 1


def test_resume_waits_for_running_work_before_discarding_selected_prefix():
    Sequence.block_size = 4
    config = make_scheduler_config()
    config.num_kvcache_blocks = 5
    config.kv_reclaim_min_keep_ratio = 0.5
    config.kv_reclaim_max_keep_ratio = 0.5
    scheduler = Scheduler(config, clock=lambda: 0.0)

    selected = make_sequence(RequestQoS(ttft_slo_ms=1.0))
    scheduler.add(selected)
    allocate_complete_prefix(scheduler.block_manager, selected)
    scheduler.block_manager.reclaim_suffix(selected, keep_blocks=1)

    victim = make_sequence(RequestQoS(ttft_slo_ms=1000.0))
    victim.mark_arrived(0, 0.0)
    allocate_complete_prefix(scheduler.block_manager, victim)
    victim.status = SequenceStatus.RUNNING
    scheduler.running.append(victim)

    assert not scheduler.block_manager.can_resume(selected)
    ready = scheduler._ensure_resume_capacity(selected, now=0.0)

    assert ready is False
    assert len(selected.block_table) == 1
    assert victim.preemption_count == 0
    assert victim in scheduler.running
    assert scheduler.metrics()["kv_reclaim_forced_fallbacks"] == 0


def test_resume_never_preempts_sequence_already_scheduled_in_current_step():
    Sequence.block_size = 4
    config = make_scheduler_config()
    config.num_kvcache_blocks = 5
    scheduler = Scheduler(config, clock=lambda: 0.0)

    selected = make_sequence(RequestQoS(ttft_slo_ms=1.0))
    scheduler.add(selected)
    allocate_complete_prefix(scheduler.block_manager, selected)
    scheduler.block_manager.reclaim_suffix(selected, keep_blocks=1)

    protected = make_sequence(RequestQoS(ttft_slo_ms=1000.0))
    protected.mark_arrived(0, 0.0)
    allocate_complete_prefix(scheduler.block_manager, protected)
    protected.status = SequenceStatus.RUNNING
    scheduler.running.append(protected)

    ready = scheduler._ensure_resume_capacity(selected, now=0.0)

    assert ready is False
    assert protected.preemption_count == 0
    assert protected in scheduler.running
    assert len(selected.block_table) == 1
    assert scheduler.metrics()["kv_reclaim_forced_fallbacks"] == 0
