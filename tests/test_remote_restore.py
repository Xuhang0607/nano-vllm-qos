import threading
from types import SimpleNamespace

import pytest

from nanovllm.engine.hierarchical_cache import (
    CacheTier,
    KVCacheGeometry,
    KVTransferPlanner,
    StorageTierProfile,
)
from nanovllm.engine.remote_restore import (
    RemoteKVPageMissing,
    RemoteKVRestoreService,
    RemotePageDescriptor,
    RemotePrefixCatalog,
)
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.storage_backend import InMemoryKVStore
from nanovllm.sampling_params import SamplingParams


class BlockingGetStore(InMemoryKVStore):
    def __init__(self):
        super().__init__()
        self.get_started = threading.Event()
        self.allow_get = threading.Event()
        self.get_count = 0
        self._count_lock = threading.Lock()

    def get(self, key):
        with self._count_lock:
            self.get_count += 1
            self.get_started.set()
        if not self.allow_get.wait(timeout=2):
            raise TimeoutError("test did not release the remote read")
        return super().get(key)


class OneMissingOneBlockingStore(InMemoryKVStore):
    def __init__(self):
        super().__init__()
        self.objects["page-1"] = b"envelope-1"
        self.slow_get_started = threading.Event()
        self.allow_slow_get = threading.Event()

    def get(self, key):
        if key == "page-0":
            return None
        self.slow_get_started.set()
        if not self.allow_slow_get.wait(timeout=2):
            raise TimeoutError("test did not release the slow remote read")
        return super().get(key)


def make_config(num_blocks=16, max_num_seqs=4):
    return SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=64,
        eos=-1,
        kvcache_block_size=2,
        num_kvcache_blocks=num_blocks,
        prefix_cache_backend="radix",
        scheduling_policy="fcfs",
        qos_best_effort_slo_ms=1000.0,
        qos_priority_boost_ms=20.0,
        qos_aging_ms_per_step=5.0,
        qos_prefill_ms_per_token=1.0,
        qos_decode_ms_per_token=10.0,
        qos_ewma_alpha=0.2,
    )


def make_sequence(suffix=9):
    return Sequence(
        [1, 2, 3, 4, suffix],
        SamplingParams(max_tokens=1, ignore_eos=True),
    )


def make_prefix():
    keys = ((1, 2), (3, 4))
    descriptors = (
        RemotePageDescriptor("page-0", "model-a", 0, 0),
        RemotePageDescriptor("page-1", "model-a", 0, 1),
    )
    envelopes = (b"envelope-0", b"envelope-1")
    return keys, descriptors, envelopes


def wait_for_restore(service, scheduler):
    service.wait_for_any(
        state.transfer for state in scheduler.pending_restores.values()
    )
    for state in scheduler.pending_restores.values():
        for future in state.transfer.futures:
            future.result(timeout=2)


def test_catalog_keeps_a_canonical_descriptor_path_for_duplicate_prefixes():
    keys, descriptors, _ = make_prefix()
    catalog = RemotePrefixCatalog()
    catalog.register(keys, descriptors)
    duplicates = (
        RemotePageDescriptor("duplicate-0", "model-a", 0, 0),
        RemotePageDescriptor("duplicate-1", "model-a", 0, 1),
    )
    catalog.register(keys, duplicates)

    assert catalog.match(keys) == descriptors
    assert len(catalog.descriptors) == 2
    catalog.validate()

    other_model = (
        RemotePageDescriptor("other-0", "model-b", 0, 0),
        RemotePageDescriptor("other-1", "model-b", 0, 1),
    )
    with pytest.raises(ValueError, match="identity"):
        catalog.register(((7, 8), (9, 10)), other_model)


def test_scheduler_restores_remote_prefix_then_registers_it_atomically():
    Sequence.block_size = 2
    keys, descriptors, envelopes = make_prefix()
    backend = InMemoryKVStore()
    backend.put_many(dict(zip((page.object_key for page in descriptors), envelopes)))
    service = RemoteKVRestoreService(backend)
    try:
        service.register_existing_prefix(keys, descriptors)
        scheduler = Scheduler(
            make_config(),
            remote_restore_service=service,
        )
        sequence = make_sequence()
        scheduler.add(sequence)

        scheduled, is_prefill = scheduler.schedule()
        assert scheduled == []
        assert is_prefill
        assert sequence.status == SequenceStatus.WAITING_FOR_KV
        assert sequence.seq_id in scheduler.pending_restores
        assert len(sequence.block_table) == sequence.num_blocks

        probe = make_sequence(suffix=8)
        assert scheduler.block_manager._lookup_cached_blocks(probe) == []

        wait_for_restore(service, scheduler)
        state = scheduler.ready_remote_restores()[0]
        assert state.transfer.result() == envelopes
        scheduler.complete_remote_restore(sequence.seq_id)

        assert sequence.status == SequenceStatus.WAITING
        assert sequence.num_cached_tokens == 4
        assert sequence.remote_restored_tokens == 4
        assert scheduler.block_manager._lookup_cached_blocks(probe) == sequence.block_table[:2]

        scheduled, is_prefill = scheduler.schedule()
        assert scheduled == [sequence]
        assert is_prefill
        assert sequence.num_scheduled_tokens == 1
    finally:
        service.close()


def test_missing_remote_page_releases_blocks_and_falls_back_to_prefill():
    Sequence.block_size = 2
    keys, descriptors, _ = make_prefix()
    service = RemoteKVRestoreService(InMemoryKVStore())
    try:
        service.register_existing_prefix(keys, descriptors)
        scheduler = Scheduler(
            make_config(),
            remote_restore_service=service,
        )
        sequence = make_sequence()
        scheduler.add(sequence)

        scheduled, _ = scheduler.schedule()
        assert scheduled == []
        wait_for_restore(service, scheduler)
        state = scheduler.ready_remote_restores()[0]
        with pytest.raises(RemoteKVPageMissing) as exc_info:
            state.transfer.result()
        scheduler.fail_remote_restore(sequence.seq_id, exc_info.value)

        assert sequence.status == SequenceStatus.WAITING
        assert sequence.block_table == []
        assert sequence.remote_restore_failures == 1
        assert sequence.remote_restore_attempted

        scheduled, is_prefill = scheduler.schedule()
        assert scheduled == [sequence]
        assert is_prefill
        assert sequence.num_scheduled_tokens == sequence.num_tokens
        assert scheduler.remote_restore_started == 1
        assert scheduler.remote_restore_failed == 1
    finally:
        service.close()


def test_cancel_pending_remote_restore_releases_reserved_blocks():
    Sequence.block_size = 2
    keys, descriptors, envelopes = make_prefix()
    backend = BlockingGetStore()
    backend.put_many(dict(zip((page.object_key for page in descriptors), envelopes)))
    service = RemoteKVRestoreService(backend)
    try:
        service.register_existing_prefix(keys, descriptors)
        scheduler = Scheduler(
            make_config(),
            remote_restore_service=service,
        )
        sequence = make_sequence()
        scheduler.add(sequence)

        scheduled, _ = scheduler.schedule()
        assert scheduled == []
        assert backend.get_started.wait(timeout=2)
        assert sequence.status == SequenceStatus.WAITING_FOR_KV
        assert sequence.block_table

        assert scheduler.cancel(sequence.seq_id)
        assert sequence.status == SequenceStatus.CANCELLED
        assert sequence.block_table == []
        assert not scheduler.pending_restores
        assert not scheduler.block_manager.used_block_ids
        assert scheduler.is_finished()
    finally:
        backend.allow_get.set()
        service.close()


def test_concurrent_requests_coalesce_remote_page_reads():
    Sequence.block_size = 2
    keys, descriptors, envelopes = make_prefix()
    backend = BlockingGetStore()
    backend.put_many(dict(zip((page.object_key for page in descriptors), envelopes)))
    service = RemoteKVRestoreService(backend)
    try:
        service.register_existing_prefix(keys, descriptors)
        scheduler = Scheduler(
            make_config(max_num_seqs=4),
            remote_restore_service=service,
        )
        first = make_sequence(suffix=9)
        second = make_sequence(suffix=10)
        scheduler.add(first)
        scheduler.add(second)

        scheduled, _ = scheduler.schedule()
        assert scheduled == []
        assert len(scheduler.pending_restores) == 2
        assert backend.get_started.wait(timeout=1)
        backend.allow_get.set()
        wait_for_restore(service, scheduler)

        assert backend.get_count == 2
        assert service.metrics()["backend_gets"] == 2
        assert service.metrics()["coalesced_gets"] == 2
        for state in tuple(scheduler.ready_remote_restores()):
            assert state.transfer.result() == envelopes
            scheduler.complete_remote_restore(state.sequence.seq_id)
        assert scheduler.remote_restore_completed == 2
    finally:
        backend.allow_get.set()
        service.close()


def test_cost_model_can_reject_a_slow_remote_hit():
    Sequence.block_size = 2
    keys, descriptors, envelopes = make_prefix()
    planner = KVTransferPlanner(
        KVCacheGeometry(2, 2, 8, 2),
        {
            CacheTier.MOONCAKE: StorageTierProfile(
                bandwidth_gbps=0.000001,
                fixed_latency_ms=1000.0,
            )
        },
    )
    backend = InMemoryKVStore()
    backend.put_many(dict(zip((page.object_key for page in descriptors), envelopes)))
    service = RemoteKVRestoreService(backend, planner=planner)
    try:
        service.register_existing_prefix(keys, descriptors)
        scheduler = Scheduler(
            make_config(),
            remote_restore_service=service,
        )
        sequence = make_sequence()
        scheduler.add(sequence)

        scheduled, is_prefill = scheduler.schedule()
        assert scheduled == [sequence]
        assert is_prefill
        assert sequence.status == SequenceStatus.RUNNING
        assert sequence.num_scheduled_tokens == sequence.num_tokens
        assert not scheduler.has_pending_restores
    finally:
        service.close()


def test_restore_fails_fast_when_one_page_is_missing():
    _, descriptors, _ = make_prefix()
    backend = OneMissingOneBlockingStore()
    service = RemoteKVRestoreService(backend)
    try:
        pending = service.submit_restore(1, descriptors, (4, 5))
        assert backend.slow_get_started.wait(timeout=1)
        service.wait_for_any([pending], timeout=1)

        assert pending.done
        with pytest.raises(RemoteKVPageMissing, match="page-0"):
            pending.result()
        pending.cancel_waiters()
    finally:
        backend.allow_slow_get.set()
        service.close()


def test_cancelled_request_does_not_cancel_a_shared_remote_fetch():
    descriptor = RemotePageDescriptor("page", "model-a", 0, 0)
    backend = BlockingGetStore()
    backend.put("page", b"envelope")
    service = RemoteKVRestoreService(backend)
    try:
        cancelled = service.submit_restore(1, (descriptor,), (4,))
        assert backend.get_started.wait(timeout=1)
        cancelled.cancel_waiters()
        survivor = service.submit_restore(2, (descriptor,), (5,))
        backend.allow_get.set()

        assert survivor.result(timeout=2) == (b"envelope",)
        assert backend.get_count == 1
        assert service.metrics()["cancelled_waiters"] == 1
    finally:
        backend.allow_get.set()
        service.close()


def test_insufficient_capacity_does_not_reserve_partial_restore_blocks():
    Sequence.block_size = 2
    keys, descriptors, envelopes = make_prefix()
    backend = InMemoryKVStore()
    backend.put_many(dict(zip((page.object_key for page in descriptors), envelopes)))
    service = RemoteKVRestoreService(backend)
    try:
        service.register_existing_prefix(keys, descriptors)
        scheduler = Scheduler(
            make_config(num_blocks=2),
            remote_restore_service=service,
        )
        sequence = make_sequence()
        scheduler.add(sequence)

        with pytest.raises(RuntimeError, match="available KV cache"):
            scheduler.schedule()
        assert sequence.status == SequenceStatus.WAITING
        assert sequence.block_table == []
        assert not scheduler.has_pending_restores
        assert len(scheduler.block_manager.free_block_ids) == 2
    finally:
        service.close()


def test_commit_failure_keeps_pending_state_available_for_atomic_rollback():
    Sequence.block_size = 2
    keys, descriptors, envelopes = make_prefix()
    backend = InMemoryKVStore()
    backend.put_many(dict(zip((page.object_key for page in descriptors), envelopes)))
    service = RemoteKVRestoreService(backend)
    try:
        service.register_existing_prefix(keys, descriptors)
        scheduler = Scheduler(
            make_config(),
            remote_restore_service=service,
        )
        sequence = make_sequence()
        scheduler.add(sequence)
        scheduler.schedule()
        wait_for_restore(service, scheduler)

        def fail_commit(seq, num_cached_blocks):
            raise RuntimeError("radix commit failed")

        scheduler.block_manager.commit_restored_prefix = fail_commit
        with pytest.raises(RuntimeError, match="radix commit failed"):
            scheduler.complete_remote_restore(sequence.seq_id)
        assert sequence.seq_id in scheduler.pending_restores

        scheduler.fail_remote_restore(
            sequence.seq_id,
            RuntimeError("radix commit failed"),
        )
        assert sequence.block_table == []
        assert sequence.status == SequenceStatus.WAITING
    finally:
        service.close()
