from collections import deque
from types import SimpleNamespace

import pytest

from nanovllm.engine.qos import (
    ExecutionTimeEstimator,
    PriorityAwareLatencyBudgetPolicy,
    RequestQoS,
)
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance_ms(self, value):
        self.value += value / 1000.0


def make_config(
    policy="fcfs",
    max_num_seqs=1,
    block_size=4,
    num_blocks=16,
    prefix_cache_backend="hash",
):
    return SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=64,
        eos=-1,
        kvcache_block_size=block_size,
        num_kvcache_blocks=num_blocks,
        prefix_cache_backend=prefix_cache_backend,
        scheduling_policy=policy,
        qos_best_effort_slo_ms=1000.0,
        qos_priority_boost_ms=20.0,
        qos_aging_ms_per_step=5.0,
        qos_prefill_ms_per_token=1.0,
        qos_decode_ms_per_token=10.0,
        qos_ewma_alpha=0.2,
    )


def make_sequence(name_token, priority=0, ttft_slo_ms=None, max_tokens=1):
    return Sequence(
        [name_token, name_token + 1],
        SamplingParams(max_tokens=max_tokens, ignore_eos=True),
        RequestQoS(priority=priority, ttft_slo_ms=ttft_slo_ms),
    )


def test_fcfs_preserves_arrival_order():
    Sequence.block_size = 4
    scheduler = Scheduler(make_config("fcfs"), clock=FakeClock())
    first = make_sequence(1, priority=0)
    second = make_sequence(10, priority=10)
    scheduler.add(first)
    scheduler.add(second)

    scheduled, is_prefill = scheduler.schedule()
    assert is_prefill
    assert scheduled == [first]


def test_pals_prioritizes_client_value():
    Sequence.block_size = 4
    clock = FakeClock()
    scheduler = Scheduler(make_config("pals"), clock=clock)
    low = make_sequence(1, priority=0)
    high = make_sequence(10, priority=5)
    scheduler.add(low)
    scheduler.add(high)

    assert scheduler.policy.budget_ms(high, 1, clock()) < scheduler.policy.budget_ms(
        low, 1, clock()
    )
    scheduled, _ = scheduler.schedule()
    assert scheduled == [high]


def test_pals_compares_prefill_and_decode_budgets():
    Sequence.block_size = 4
    clock = FakeClock()
    scheduler = Scheduler(make_config("pals"), clock=clock)
    running = Sequence(
        [1, 2],
        SamplingParams(max_tokens=3, ignore_eos=True),
        RequestQoS(e2e_slo_ms=30.0),
    )
    scheduler.add(running)
    scheduled, is_prefill = scheduler.schedule()
    scheduler.postprocess(scheduled, [99], is_prefill)

    waiting = make_sequence(10, ttft_slo_ms=100.0)
    scheduler.add(waiting)
    scheduled, is_prefill = scheduler.schedule()

    assert not is_prefill
    assert scheduled == [running]


def test_pals_switches_to_urgent_prefill():
    Sequence.block_size = 4
    clock = FakeClock()
    scheduler = Scheduler(make_config("pals"), clock=clock)
    running = Sequence(
        [1, 2],
        SamplingParams(max_tokens=3, ignore_eos=True),
        RequestQoS(e2e_slo_ms=100.0),
    )
    scheduler.add(running)
    scheduled, is_prefill = scheduler.schedule()
    scheduler.postprocess(scheduled, [99], is_prefill)

    waiting = make_sequence(10, ttft_slo_ms=10.0)
    scheduler.add(waiting)
    scheduled, is_prefill = scheduler.schedule()

    assert is_prefill
    assert scheduled == [waiting]


def test_pals_orders_decode_requests_by_budget():
    Sequence.block_size = 4
    clock = FakeClock()
    scheduler = Scheduler(make_config("pals", max_num_seqs=2), clock=clock)
    low = make_sequence(1, priority=0, max_tokens=2)
    high = make_sequence(10, priority=5, max_tokens=2)
    scheduler.add(low)
    scheduler.add(high)
    scheduled, is_prefill = scheduler.schedule()
    scheduler.postprocess(scheduled, [91, 92], is_prefill)

    scheduler.max_num_seqs = 1
    scheduled, is_prefill = scheduler.schedule()

    assert not is_prefill
    assert scheduled == [high]


def test_latency_budget_accounts_for_remaining_work_and_cache():
    estimator = ExecutionTimeEstimator(1.0, 10.0, 0.2)
    policy = PriorityAwareLatencyBudgetPolicy(estimator, 1000.0, 0.0, 0.0)
    short = make_sequence(1, ttft_slo_ms=100.0)
    long = Sequence(
        list(range(20)),
        SamplingParams(max_tokens=1, ignore_eos=True),
        RequestQoS(ttft_slo_ms=100.0),
    )
    short.mark_arrived(0, 0.0)
    long.mark_arrived(0, 0.0)

    assert policy.budget_ms(long, 1, 0.0) < policy.budget_ms(short, 1, 0.0)
    long.num_cached_tokens = 19
    assert policy.budget_ms(long, 1, 0.0) > policy.budget_ms(short, 1, 0.0)


def test_aging_prevents_best_effort_starvation():
    estimator = ExecutionTimeEstimator(1.0, 10.0, 0.2)
    policy = PriorityAwareLatencyBudgetPolicy(estimator, 1000.0, 20.0, 10.0)
    old = make_sequence(1, priority=0)
    new = make_sequence(10, priority=2)
    old.mark_arrived(0, 0.0)
    new.mark_arrived(5, 0.0)

    selected = policy.select_waiting(deque([old, new]), step=5, now=0.0)
    assert selected is old


def test_preemption_victim_has_most_budget():
    estimator = ExecutionTimeEstimator(1.0, 10.0, 0.2)
    policy = PriorityAwareLatencyBudgetPolicy(estimator, 1000.0, 20.0, 0.0)
    low = make_sequence(1, priority=0)
    high = make_sequence(10, priority=5)
    low.mark_arrived(0, 0.0)
    high.mark_arrived(0, 0.0)

    victim = policy.select_preemption_victim(deque([high, low]), step=1, now=0.0)
    assert victim is low


def test_request_metrics_and_slo_attainment():
    Sequence.block_size = 4
    clock = FakeClock()
    scheduler = Scheduler(make_config("fcfs"), clock=clock)
    sequence = Sequence(
        [1, 2],
        SamplingParams(max_tokens=1, ignore_eos=True),
        RequestQoS(priority=1, ttft_slo_ms=25.0, e2e_slo_ms=40.0),
    )
    scheduler.add(sequence)
    clock.advance_ms(10)
    scheduled, is_prefill = scheduler.schedule()
    clock.advance_ms(20)
    scheduler.postprocess(scheduled, [99], is_prefill)

    metrics = scheduler.request_metrics(sequence.seq_id)
    assert metrics.queue_ms == pytest.approx(10.0)
    assert metrics.ttft_ms == pytest.approx(30.0)
    assert metrics.e2e_ms == pytest.approx(30.0)
    assert metrics.ttft_slo_met is False
    assert metrics.e2e_slo_met is True


def test_request_metrics_include_tpot():
    sequence = Sequence(
        [1],
        SamplingParams(max_tokens=3, ignore_eos=True),
        RequestQoS(tpot_slo_ms=20.0),
    )
    sequence.mark_arrived(0, 0.0)
    sequence.mark_scheduled(0.0)
    sequence.append_token(2, 0.010)
    sequence.append_token(3, 0.020)
    sequence.append_token(4, 0.040)
    sequence.mark_finished(0.040)

    metrics = sequence.metrics()
    assert metrics.tpot_ms == pytest.approx(15.0)
    assert metrics.tpot_slo_met is True


def test_tpot_budget_prioritizes_urgent_decode():
    estimator = ExecutionTimeEstimator(1.0, 10.0, 0.2)
    policy = PriorityAwareLatencyBudgetPolicy(estimator, 1000.0, 0.0, 0.0)
    tight = Sequence(
        [1],
        SamplingParams(max_tokens=2, ignore_eos=True),
        RequestQoS(tpot_slo_ms=5.0),
    )
    loose = Sequence(
        [2],
        SamplingParams(max_tokens=2, ignore_eos=True),
        RequestQoS(tpot_slo_ms=50.0),
    )
    for sequence in (tight, loose):
        sequence.mark_arrived(0, 0.0)
        sequence.append_token(3, 0.0)
        sequence.is_prefill = False

    assert policy.select_running(deque([loose, tight]), 1, 0.0) is tight


def test_execution_time_estimator_uses_ewma():
    estimator = ExecutionTimeEstimator(0.1, 2.0, 0.2)
    estimator.observe(is_prefill=True, num_tokens=100, elapsed_ms=50.0)
    estimator.observe(is_prefill=False, num_tokens=4, elapsed_ms=20.0)
    assert estimator.prefill_ms_per_token == pytest.approx(0.18)
    assert estimator.decode_ms_per_token == pytest.approx(2.6)


def test_scheduler_uses_radix_prefix_cache_end_to_end():
    Sequence.block_size = 2
    clock = FakeClock()
    scheduler = Scheduler(
        make_config(
            "fcfs",
            block_size=2,
            prefix_cache_backend="radix",
        ),
        clock=clock,
    )
    first = Sequence(
        [1, 2, 3, 4, 5],
        SamplingParams(max_tokens=1, ignore_eos=True),
    )
    scheduler.add(first)
    scheduled, is_prefill = scheduler.schedule()
    scheduler.postprocess(scheduled, [90], is_prefill)

    second = Sequence(
        [1, 2, 3, 4, 9],
        SamplingParams(max_tokens=1, ignore_eos=True),
    )
    scheduler.add(second)
    scheduled, is_prefill = scheduler.schedule()

    assert is_prefill
    assert scheduled == [second]
    assert second.num_cached_tokens == 4
    assert second.num_scheduled_tokens == 1
    metrics = scheduler.metrics()
    assert metrics["prefix_cache_hit_blocks"] == 2
    assert metrics["max_num_seqs"] == 1
    assert metrics["kvcache_block_size"] == 2
    assert metrics["num_kvcache_blocks"] == 16
    assert metrics["kv_cache_capacity_tokens"] == 32
    assert metrics["remote_kv_cost_aware"] is True


def test_cancel_waiting_request_removes_it_from_scheduler():
    Sequence.block_size = 4
    scheduler = Scheduler(make_config("fcfs"), clock=FakeClock())
    sequence = make_sequence(1)
    scheduler.add(sequence)

    assert scheduler.cancel(sequence.seq_id)
    assert sequence.status == SequenceStatus.CANCELLED
    assert sequence.is_terminal
    assert not sequence.is_finished
    assert scheduler.is_finished()
    assert scheduler.metrics()["cancelled_requests"] == 1
    assert not scheduler.cancel(sequence.seq_id)


def test_cancel_running_request_releases_kv_blocks():
    Sequence.block_size = 4
    scheduler = Scheduler(make_config("fcfs"), clock=FakeClock())
    sequence = make_sequence(1, max_tokens=2)
    scheduler.add(sequence)
    scheduled, is_prefill = scheduler.schedule()
    scheduler.postprocess(scheduled, [99], is_prefill)

    assert sequence.status == SequenceStatus.RUNNING
    assert sequence.block_table
    assert scheduler.cancel(sequence.seq_id)
    assert not sequence.block_table
    assert not scheduler.block_manager.used_block_ids
    assert scheduler.is_finished()
