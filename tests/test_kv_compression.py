from types import SimpleNamespace

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.qos import RequestQoS
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def make_config():
    return SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=64,
        max_model_len=128,
        requested_max_model_len=128,
        eos=-1,
        kvcache_block_size=4,
        num_kvcache_blocks=8,
        prefix_cache_backend="radix",
        scheduling_policy="pals",
        qos_best_effort_slo_ms=1000.0,
        qos_priority_boost_ms=20.0,
        qos_aging_ms_per_step=5.0,
        qos_prefill_ms_per_token=1.0,
        qos_decode_ms_per_token=10.0,
        qos_ewma_alpha=0.2,
        kv_reclaim_policy="recompute",
        kv_reclaim_min_keep_ratio=0.0,
        kv_reclaim_max_keep_ratio=0.75,
        kv_reclaim_budget_scale_ms=1000.0,
        kv_reclaim_target_free_blocks=1,
        kv_compression_policy="sink_recent",
        kv_compression_sink_blocks=1,
        kv_compression_recent_blocks=2,
        kv_compression_trigger_free_ratio=0.25,
    )


def make_long_sequence():
    return Sequence(
        list(range(24)),
        SamplingParams(max_tokens=4, ignore_eos=True),
        RequestQoS(e2e_slo_ms=1000.0),
    )


def allocate_cached(manager, sequence):
    assert manager.can_allocate(sequence) == 0
    manager.allocate(sequence, 0)
    sequence.num_cached_tokens = sequence.num_tokens
    sequence.num_physical_cached_tokens = sequence.num_tokens


def test_sink_recent_compression_keeps_first_and_latest_pages():
    Sequence.block_size = 4
    manager = BlockManager(8, 4, "radix")
    sequence = make_long_sequence()
    allocate_cached(manager, sequence)
    original = sequence.block_table.copy()

    dropped, freed, dropped_tokens = manager.compress_sink_recent(sequence, 1, 2)

    assert dropped == 3
    assert freed == 3
    assert dropped_tokens == 12
    assert sequence.block_table == [original[0], *original[-2:]]
    assert sequence.num_cached_tokens == 24
    assert sequence.num_physical_cached_tokens == 12
    assert sequence.attention_context_tokens == 13


def test_query_aware_compression_retains_history_block_matching_tail_query():
    Sequence.block_size = 4
    manager = BlockManager(8, 4, "radix")
    token_ids = [
        1, 2, 3, 4,
        10, 11, 12, 13,
        42, 20, 21, 22,
        30, 31, 32, 33,
        40, 41, 43, 44,
        50, 51, 52, 53,
        42, 90, 91, 92,
    ]
    sequence = Sequence(token_ids, SamplingParams(max_tokens=1, ignore_eos=True))
    allocate_cached(manager, sequence)
    original = sequence.block_table.copy()

    dropped, freed, _dropped_tokens = manager.compress_query_aware(
        sequence,
        sink_blocks=1,
        recent_blocks=2,
        importance_blocks=1,
        query_tokens=4,
    )

    assert dropped == 3
    assert freed == 3
    assert sequence.block_table == [
        original[0],
        original[2],
        original[5],
        original[6],
    ]
    assert sequence.kv_block_logical_indices == [0, 2, 5, 6]


def test_scheduler_compresses_under_pressure_and_preemption_resets_layout():
    Sequence.block_size = 4
    scheduler = Scheduler(make_config(), clock=lambda: 0.0)
    sequence = make_long_sequence()
    scheduler.add(sequence)
    allocate_cached(scheduler.block_manager, sequence)
    sequence.status = SequenceStatus.RUNNING
    scheduler.waiting.remove(sequence)
    scheduler.running.append(sequence)

    freed = scheduler._maybe_compress(sequence)

    assert freed == 3
    assert sequence.kv_compressed is True
    assert sequence.kv_compression_dropped_tokens == 12
    assert scheduler.metrics()["kv_compression_events"] == 1

    scheduler.running.remove(sequence)
    scheduler.preempt(sequence)

    assert sequence.kv_compressed is False
    assert sequence.num_cached_tokens == 0
    assert sequence.num_physical_cached_tokens == 0
    assert not sequence.block_table
    assert sequence.kv_invalidated_tokens == 24


def test_scheduler_dispatches_query_aware_compression_policy():
    Sequence.block_size = 4
    config = make_config()
    config.kv_compression_policy = "query_aware"
    config.kv_compression_importance_blocks = 1
    config.kv_compression_query_tokens = 4
    scheduler = Scheduler(config, clock=lambda: 0.0)
    sequence = make_long_sequence()
    scheduler.add(sequence)
    allocate_cached(scheduler.block_manager, sequence)

    freed = scheduler._maybe_compress(sequence)

    assert freed >= 2
    assert sequence.kv_compressed is True
    assert scheduler.metrics()["kv_compression_events"] == 1


def test_query_aware_fills_budget_with_recent_history_when_scores_are_zero():
    Sequence.block_size = 4
    manager = BlockManager(8, 4, "radix")
    sequence = make_long_sequence()
    allocate_cached(manager, sequence)
    manager.compress_query_aware(sequence, 1, 2, 1, 4)
    assert len(sequence.block_table) == 4
    assert sequence.kv_block_logical_indices == [0, 3, 4, 5]
