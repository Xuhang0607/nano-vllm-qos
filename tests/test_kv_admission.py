from itertools import count
import random
from types import SimpleNamespace

import pytest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.kv_admission import growth_pages, priority_reserve_pages
from nanovllm.engine.qos import RequestQoS
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def settings(horizon):
    return SimpleNamespace(
        max_num_seqs=4, max_num_batched_tokens=16, eos=-1,
        kvcache_block_size=4, num_kvcache_blocks=4,
        scheduling_policy="fcfs", qos_prefill_ms_per_token=1.0,
        qos_decode_ms_per_token=1.0, qos_ewma_alpha=0.2,
        kv_reclaim_policy="recompute", kv_admission_lookahead=horizon,
    )


@pytest.mark.parametrize("cached,owned,future,expected", [
    (4, 1, 1, 1), (3, 1, 1, 0), (4, 1, 0, 0), (8, 3, 1, 0), (4, 1, 9, 3),
])
def test_growth_counts_only_new_physical_pages(cached, owned, future, expected):
    assert growth_pages(cached, owned, future, 4) == expected


def test_priority_and_pending_prefill_reservation(monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", 4)
    resident = Sequence([1] * 4, SamplingParams(max_tokens=3))
    resident.num_scheduled_tokens = 4
    resident.block_table = [0]
    candidate = Sequence([2] * 3, SamplingParams(max_tokens=1))
    assert priority_reserve_pages(candidate, [resident], 2, 4) == 1
    candidate.qos = RequestQoS(priority=10)
    assert priority_reserve_pages(candidate, [resident], 2, 4) == 0
    candidate.qos = RequestQoS()
    resident.max_tokens = 1
    assert priority_reserve_pages(candidate, [resident], 2, 4) == 0


def test_compressed_resident_uses_physical_not_logical_length(monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", 4)
    resident = Sequence([1] * 100, SamplingParams(max_tokens=3))
    resident.num_physical_cached_tokens = 4
    resident.num_cached_tokens = 100
    resident.block_table = [0]
    candidate = Sequence([2] * 3, SamplingParams(max_tokens=1))
    assert priority_reserve_pages(candidate, [resident], 2, 4) == 1


@pytest.mark.parametrize("backend", ["hash", "radix"])
def test_planned_page_cost_distinguishes_active_and_free_cached_pages(monkeypatch, backend):
    monkeypatch.setattr(Sequence, "block_size", 4)
    manager = BlockManager(8, 4, backend)
    first, second = Sequence(list(range(8))), Sequence(list(range(8)))
    manager.allocate(first, manager.can_allocate(first))
    first.num_scheduled_tokens = 8
    manager.hash_blocks(first)
    assert manager.can_allocate(second) == 1
    lookups = manager.cache_lookups
    assert manager.planned_allocation_pages(second) == 1
    assert manager.cache_lookups == lookups
    manager.deallocate(first)
    assert manager.can_allocate(second) == 1
    assert manager.planned_allocation_pages(second) == 2


def test_page_reserve_reduces_small_reproduction_without_changing_outputs(monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", 4)
    results = []
    for horizon in (0, 2):
        monkeypatch.setattr(Sequence, "counter", count())
        scheduler = Scheduler(settings(horizon), clock=lambda: 1.0)
        for start in (100, 200):
            scheduler.add(Sequence(list(range(start, start + 8)),
                                   SamplingParams(max_tokens=3, ignore_eos=True)))
        for _ in range(30):
            if scheduler.is_finished():
                break
            seqs, prefill = scheduler.schedule()
            scheduler.postprocess(seqs, [999] * len(seqs), prefill)
        assert scheduler.is_finished()
        assert [item.completion_tokens for item in scheduler.completed_metrics] == [3, 3]
        assert not scheduler.block_manager.pending_matches
        results.append(scheduler)
    assert results[0].kv_reclaim_events > 0
    assert results[1].kv_reclaim_events == 0
    assert results[1].kv_admission_deferred_checks > 0


@pytest.mark.parametrize("resident_length", [8, 12])
def test_large_deferred_request_does_not_block_small_urgent_request(monkeypatch, resident_length):
    monkeypatch.setattr(Sequence, "block_size", 4)
    scheduler = Scheduler(settings(2), clock=lambda: 1.0)
    resident = Sequence([1] * resident_length, SamplingParams(max_tokens=3, ignore_eos=True))
    scheduler.add(resident)
    seqs, prefill = scheduler.schedule()
    scheduler.postprocess(seqs, [999], prefill)
    large = Sequence([2] * 8, SamplingParams(max_tokens=3, ignore_eos=True))
    urgent = Sequence([3] * 3, SamplingParams(max_tokens=1), RequestQoS(priority=10))
    scheduler.add(large)
    scheduler.add(urgent)
    seqs, prefill = scheduler.schedule()
    assert prefill and seqs == [urgent]
    assert not large.block_table
    assert large.seq_id not in scheduler.block_manager.pending_matches


def test_invalid_horizon():
    with pytest.raises(ValueError, match="kv_admission_lookahead"):
        Scheduler(settings(-1))


def test_pals_progresses_partial_prefill_when_new_request_cannot_fit(monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", 4)
    config = settings(0)
    config.max_num_batched_tokens = 4
    config.scheduling_policy = "pals"
    config.qos_best_effort_slo_ms = 1000.0
    config.qos_priority_boost_ms = 20.0
    config.qos_aging_ms_per_step = 1.0
    scheduler = Scheduler(config, clock=lambda: 1.0)
    resident = Sequence([1] * 8, SamplingParams(max_tokens=4), RequestQoS(priority=10))
    newcomer = Sequence([2] * 12, SamplingParams(max_tokens=2), RequestQoS(priority=10))
    scheduler.add(resident)
    seqs, prefill = scheduler.schedule()
    scheduler.postprocess(seqs, [999], prefill)
    scheduler.add(newcomer)
    assert scheduler.policy.select_waiting(scheduler.waiting, scheduler.step_id, 1.0) is newcomer
    seqs, prefill = scheduler.schedule()
    assert prefill and seqs == [resident]
    assert not newcomer.block_table


def test_last_decode_self_preemption_retries_prefill(monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", 4)
    config = settings(0)
    config.scheduling_policy = "pals"
    config.qos_best_effort_slo_ms = 1000.0
    config.qos_priority_boost_ms = 20.0
    config.qos_aging_ms_per_step = 1.0
    scheduler = Scheduler(config, clock=lambda: 1.0)
    live = Sequence([1] * 8, SamplingParams(max_tokens=5), RequestQoS(priority=10))
    scheduler.add(live)
    seqs, prefill = scheduler.schedule()
    scheduler.postprocess(seqs, [999], prefill)
    partial = Sequence([2] * 8, SamplingParams(max_tokens=2))
    scheduler.add(partial)
    scheduler.block_manager.allocate(partial, scheduler.block_manager.can_allocate(partial))
    partial.num_cached_tokens = partial.num_physical_cached_tokens = 4
    assert not scheduler.block_manager.free_block_ids
    seqs, prefill = scheduler.schedule()
    assert prefill and seqs == [partial]
    assert live.preemption_count == 1
    assert live in scheduler.waiting


@pytest.mark.parametrize("policy", ["fcfs", "pals"])
@pytest.mark.parametrize("seed", [7, 11, 19])
def test_mixed_lengths_priorities_and_chunked_prefill_eventually_complete(monkeypatch, policy, seed):
    monkeypatch.setattr(Sequence, "block_size", 4)
    rng = random.Random(seed)
    requests = [(rng.randint(3, 20), rng.randint(1, 8), rng.choice([0, 10]))
                for _ in range(20)]
    for horizon in (0, 8):
        config = settings(horizon)
        config.num_kvcache_blocks = 8
        config.max_num_batched_tokens = 8
        config.scheduling_policy = policy
        config.qos_best_effort_slo_ms = 1000.0
        config.qos_priority_boost_ms = 20.0
        config.qos_aging_ms_per_step = 1.0
        scheduler = Scheduler(config, clock=lambda: 1.0)
        sequences = []
        for index, (length, output, priority) in enumerate(requests):
            seq = Sequence(list(range(index * 100, index * 100 + length)),
                           SamplingParams(max_tokens=output, ignore_eos=True),
                           RequestQoS(priority=priority))
            scheduler.add(seq)
            sequences.append(seq)
        for _ in range(3000):
            if scheduler.is_finished():
                break
            try:
                seqs, prefill = scheduler.schedule()
            except RuntimeError as error:
                state = [(seq.seq_id, seq.num_tokens, seq.num_cached_tokens,
                          len(seq.block_table), seq.qos.priority, seq.preemption_count)
                         for seq in scheduler.waiting]
                pytest.fail(f"horizon={horizon}, step={scheduler.step_id}, "
                            f"free={len(scheduler.block_manager.free_block_ids)}, "
                            f"running={len(scheduler.running)}, waiting={state}: {error}")
            scheduler.postprocess(seqs, [9999] * len(seqs), prefill)
        assert scheduler.is_finished()
        assert len(scheduler.completed_metrics) == len(requests)
        assert len(scheduler.block_manager.free_block_ids) == 8
        for seq, (_, output, _) in zip(sequences, requests):
            assert seq.completion_token_ids == [9999] * output
