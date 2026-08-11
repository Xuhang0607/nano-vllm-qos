from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.kv_reclaim import (
    KVReclaimDecision,
    KVReclaimPolicy,
    create_kv_reclaim_policy,
)
from nanovllm.engine.qos import ExecutionTimeEstimator, create_policy, summarize_metrics
from nanovllm.engine.sequence import Sequence, SequenceStatus

if TYPE_CHECKING:
    from nanovllm.config import Config


@dataclass(slots=True)
class PendingRestoreState:
    sequence: Sequence
    transfer: object
    total_cached_blocks: int
    restored_blocks: int


class Scheduler:

    def __init__(
        self,
        config: "Config",
        clock=perf_counter,
        remote_restore_service=None,
    ):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_model_len = getattr(
            config,
            "max_model_len",
            config.max_num_batched_tokens,
        )
        self.requested_max_model_len = getattr(
            config, "requested_max_model_len", self.max_model_len
        )
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.num_kvcache_blocks_override = getattr(
            config, "num_kvcache_blocks_override", None
        )
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            getattr(config, "prefix_cache_backend", "hash"),
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.clock = clock
        self.step_id = 0
        self.estimator = ExecutionTimeEstimator(
            prefill_ms_per_token=config.qos_prefill_ms_per_token,
            decode_ms_per_token=config.qos_decode_ms_per_token,
            alpha=config.qos_ewma_alpha,
        )
        self.policy_name = config.scheduling_policy
        self.remote_kv_cost_aware = getattr(config, "remote_kv_cost_aware", True)
        self.policy = create_policy(self.policy_name, self.estimator, config)
        self.kv_reclaim_policy_name = getattr(
            config, "kv_reclaim_policy", "slo_aware"
        )
        self.kv_reclaim_policy = create_kv_reclaim_policy(
            self.kv_reclaim_policy_name, config
        )
        self.kv_reclaim_target_free_blocks = getattr(
            config, "kv_reclaim_target_free_blocks", 2
        )
        self.kv_reclaim_events = 0
        self.kv_reclaim_requested_blocks = 0
        self.kv_reclaim_freed_blocks = 0
        self.kv_reclaim_retained_blocks = 0
        self.kv_reclaim_invalidated_tokens = 0
        self.kv_reclaim_forced_fallbacks = 0
        self.kv_compression_policy = getattr(
            config, "kv_compression_policy", "none"
        )
        self.kv_compression_sink_blocks = getattr(
            config, "kv_compression_sink_blocks", 1
        )
        self.kv_compression_recent_blocks = getattr(
            config, "kv_compression_recent_blocks", 8
        )
        self.kv_compression_importance_blocks = getattr(
            config, "kv_compression_importance_blocks", 2
        )
        self.kv_compression_query_tokens = getattr(
            config, "kv_compression_query_tokens", 64
        )
        self.kv_compression_trigger_free_ratio = getattr(
            config, "kv_compression_trigger_free_ratio", 0.15
        )
        self.kv_compression_events = 0
        self.kv_compression_dropped_blocks = 0
        self.kv_compression_freed_blocks = 0
        self.kv_compression_dropped_tokens = 0
        self.completed_metrics = []
        self.metrics_by_seq_id = {}
        self.remote_restore_service = remote_restore_service
        self.pending_restores: dict[int, PendingRestoreState] = {}
        self.remote_restore_started = 0
        self.remote_restore_completed = 0
        self.remote_restore_failed = 0
        self.cancelled_requests = 0

    def is_finished(self):
        return not self.waiting and not self.running and not self.pending_restores

    def add(self, seq: Sequence):
        seq.mark_arrived(self.step_id, self.clock())
        self.waiting.append(seq)

    def cancel(self, seq_id: int) -> bool:
        """Cancel a queued request and release every KV block it owns."""
        seq = next((item for item in self.waiting if item.seq_id == seq_id), None)
        if seq is not None:
            self.waiting.remove(seq)
        else:
            seq = next((item for item in self.running if item.seq_id == seq_id), None)
            if seq is not None:
                self.running.remove(seq)
            else:
                state = self.pending_restores.pop(seq_id, None)
                if state is None:
                    return False
                seq = state.sequence
                state.transfer.cancel_waiters()

        self.block_manager.pending_matches.pop(seq_id, None)
        self.block_manager.pending_resume_matches.pop(seq_id, None)
        if seq.block_table:
            self.block_manager.deallocate(seq)
        seq.num_scheduled_tokens = 0
        seq.cancel(self.clock())
        self.cancelled_requests += 1
        return True

    def _start_remote_restore(
        self,
        seq: Sequence,
        local_cached_blocks: int,
        now: float,
    ) -> bool:
        if self.remote_restore_service is None or seq.remote_restore_attempted:
            return False
        candidate = self.remote_restore_service.plan_restore(
            seq,
            local_cached_blocks,
            self.estimator.prefill_ms_per_token,
        )
        if candidate is None:
            return False

        total_cached_blocks = candidate.num_blocks
        block_ids = self.block_manager.reserve_restore(
            seq,
            local_cached_blocks,
            total_cached_blocks,
        )
        pages = candidate.pages[local_cached_blocks:total_cached_blocks]
        seq.begin_remote_restore(now)
        try:
            transfer = self.remote_restore_service.submit_restore(
                seq.seq_id,
                pages,
                block_ids,
            )
        except Exception as exc:  # noqa: BLE001 - restore setup falls back to prefill
            self.block_manager.abort_restore(seq)
            seq.fail_remote_restore(now, exc)
            self.remote_restore_failed += 1
            return False

        self.waiting.remove(seq)
        self.pending_restores[seq.seq_id] = PendingRestoreState(
            sequence=seq,
            transfer=transfer,
            total_cached_blocks=total_cached_blocks,
            restored_blocks=len(pages),
        )
        self.remote_restore_started += 1
        return True

    def ready_remote_restores(self):
        return tuple(
            state
            for state in self.pending_restores.values()
            if state.transfer.done
        )

    def complete_remote_restore(self, seq_id: int):
        state = self.pending_restores[seq_id]
        self.block_manager.commit_restored_prefix(
            state.sequence,
            state.total_cached_blocks,
        )
        state.sequence.finish_remote_restore(
            self.clock(),
            state.restored_blocks * self.block_size,
        )
        del self.pending_restores[seq_id]
        self.waiting.append(state.sequence)
        self.remote_restore_completed += 1

    def fail_remote_restore(self, seq_id: int, error: Exception):
        state = self.pending_restores.pop(seq_id)
        state.transfer.cancel_waiters()
        self.block_manager.abort_restore(state.sequence)
        state.sequence.fail_remote_restore(self.clock(), error)
        self.waiting.appendleft(state.sequence)
        self.remote_restore_failed += 1

    @property
    def has_pending_restores(self):
        return bool(self.pending_restores)

    def _apply_kv_reclaim(
        self, seq: Sequence, decision: KVReclaimDecision
    ) -> int:
        freed_blocks = self.block_manager.reclaim_suffix(
            seq, decision.keep_blocks
        )
        seq.record_kv_reclaim(
            reclaimed_blocks=freed_blocks,
            retained_blocks=decision.keep_blocks,
            invalidated_tokens=decision.invalidated_tokens,
        )
        self.kv_reclaim_events += 1
        self.kv_reclaim_requested_blocks += decision.reclaim_blocks
        self.kv_reclaim_freed_blocks += freed_blocks
        self.kv_reclaim_retained_blocks += decision.keep_blocks
        self.kv_reclaim_invalidated_tokens += decision.invalidated_tokens
        return freed_blocks

    def _maybe_compress(self, seq: Sequence) -> int:
        if self.kv_compression_policy not in ("sink_recent", "query_aware"):
            return 0
        total_blocks = len(self.block_manager.blocks)
        free_ratio = len(self.block_manager.free_block_ids) / total_blocks
        if free_ratio > self.kv_compression_trigger_free_ratio:
            return 0
        if self.kv_compression_policy == "query_aware":
            dropped, freed, dropped_tokens = (
                self.block_manager.compress_query_aware(
                    seq,
                    self.kv_compression_sink_blocks,
                    self.kv_compression_recent_blocks,
                    self.kv_compression_importance_blocks,
                    self.kv_compression_query_tokens,
                )
            )
        else:
            dropped, freed, dropped_tokens = (
                self.block_manager.compress_sink_recent(
                    seq,
                    self.kv_compression_sink_blocks,
                    self.kv_compression_recent_blocks,
                )
            )
        if dropped == 0:
            return 0
        seq.record_kv_compression(dropped, dropped_tokens)
        self.kv_compression_events += 1
        self.kv_compression_dropped_blocks += dropped
        self.kv_compression_freed_blocks += freed
        self.kv_compression_dropped_tokens += dropped_tokens
        return freed

    def _ensure_resume_capacity(
        self,
        seq: Sequence,
        now: float,
    ):
        """Evict retained waiting prefixes until the selected request can resume."""
        def reclaim_order(item: Sequence):
            budget_ms = self.policy.budget_ms(item, self.step_id, now)
            return (
                float("inf") if budget_ms is None else budget_ms,
                item.arrival_step,
                item.seq_id,
            )

        while seq.block_table and not self.block_manager.can_resume(seq):
            candidates = [
                item for item in self.waiting if item is not seq and item.block_table
            ]
            if candidates:
                victim = max(candidates, key=reclaim_order)
                budget_ms = self.policy.budget_ms(victim, self.step_id, now)
                decision = KVReclaimPolicy().plan(victim, budget_ms)
                self._apply_kv_reclaim(victim, decision)
                self.kv_reclaim_forced_fallbacks += 1
                continue

            if self.running:
                return False

            budget_ms = self.policy.budget_ms(seq, self.step_id, now)
            decision = KVReclaimPolicy().plan(seq, budget_ms)
            self._apply_kv_reclaim(seq, decision)
            self.kv_reclaim_forced_fallbacks += 1
        return self.block_manager.can_resume(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        self.step_id += 1
        now = self.clock()
        scheduled_seqs = []
        num_batched_tokens = 0

        # Compare the most urgent prefill and decode requests before choosing a phase.
        schedule_prefill = self.policy.should_schedule_prefill(
            self.waiting, self.running, self.step_id, now
        )
        while schedule_prefill and self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.policy.select_waiting(self.waiting, self.step_id, now)
            seq.last_budget_ms = self.policy.budget_ms(seq, self.step_id, now)
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if seq.block_table and not self._ensure_resume_capacity(seq, now):
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                if self._start_remote_restore(seq, num_cached_blocks, now):
                    continue
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                reused_tokens = self.block_manager.resume(seq)
                seq.record_reused_tokens(reused_tokens)
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                previous_cached_tokens = seq.num_cached_tokens
                self.block_manager.allocate(seq, num_cached_blocks)
                seq.record_reused_tokens(
                    max(0, seq.num_cached_tokens - previous_cached_tokens)
                )
            seq.mark_scheduled(now)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.remove(seq)
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.policy.select_running(self.running, self.step_id, now)
            seq.last_budget_ms = self.policy.budget_ms(seq, self.step_id, now)
            self.running.remove(seq)
            self._maybe_compress(seq)
            while not self.block_manager.can_append(seq):
                if self.running:
                    victim = self.policy.select_preemption_victim(self.running, self.step_id, now)
                    self.running.remove(victim)
                    self.preempt(victim)
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        if self.pending_restores and not self.running:
            return [], True
        if not scheduled_seqs:
            raise RuntimeError("no runnable request fits in the available KV cache")
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        if seq.kv_compressed:
            owned_blocks = len(seq.block_table)
            invalidated_tokens = seq.num_cached_tokens
            free_before = len(self.block_manager.free_block_ids)
            self.block_manager.deallocate(seq)
            freed_blocks = len(self.block_manager.free_block_ids) - free_before
            seq.record_kv_reclaim(
                reclaimed_blocks=freed_blocks,
                retained_blocks=0,
                invalidated_tokens=invalidated_tokens,
            )
            self.kv_reclaim_events += 1
            self.kv_reclaim_requested_blocks += owned_blocks
            self.kv_reclaim_freed_blocks += freed_blocks
            self.kv_reclaim_invalidated_tokens += invalidated_tokens
            seq.status = SequenceStatus.WAITING
            seq.is_prefill = True
            seq.preemption_count += 1
            self.waiting.appendleft(seq)
            return
        budget_ms = self.policy.budget_ms(seq, self.step_id, self.clock())
        min_reclaim_blocks = max(
            1,
            self.kv_reclaim_target_free_blocks
            - len(self.block_manager.free_block_ids),
        )
        decision = self.kv_reclaim_policy.plan(
            seq,
            budget_ms,
            min_reclaim_blocks=min_reclaim_blocks,
        )
        self._apply_kv_reclaim(seq, decision)
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.preemption_count += 1
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        now = self.clock()
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            if is_prefill:
                seq.record_recomputed_tokens(seq.num_scheduled_tokens)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_physical_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id, now)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                seq.mark_finished(now)
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                metrics = seq.metrics()
                self.completed_metrics.append(metrics)
                self.metrics_by_seq_id[seq.seq_id] = metrics

    def observe_execution(self, is_prefill: bool, num_tokens: int, elapsed_ms: float):
        self.estimator.observe(is_prefill, num_tokens, elapsed_ms)

    def request_metrics(self, seq_id: int):
        return self.metrics_by_seq_id.get(seq_id)

    def metrics(self):
        result = summarize_metrics(self.completed_metrics)
        result.update({
            "policy": self.policy_name,
            "max_model_len": self.max_model_len,
            "requested_max_model_len": self.requested_max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "kvcache_block_size": self.block_size,
            "num_kvcache_blocks": len(self.block_manager.blocks),
            "num_kvcache_blocks_override": getattr(
                self, "num_kvcache_blocks_override", None
            ),
            "kv_cache_capacity_tokens": len(self.block_manager.blocks)
            * self.block_size,
            "prefill_ms_per_token_ewma": self.estimator.prefill_ms_per_token,
            "decode_ms_per_token_ewma": self.estimator.decode_ms_per_token,
            "remote_kv_cost_aware": self.remote_kv_cost_aware,
            "kv_reclaim_policy": self.kv_reclaim_policy_name,
            "kv_reclaim_target_free_blocks": self.kv_reclaim_target_free_blocks,
            "kv_reclaim_events": self.kv_reclaim_events,
            "kv_reclaim_requested_blocks": self.kv_reclaim_requested_blocks,
            "kv_reclaim_freed_blocks": self.kv_reclaim_freed_blocks,
            "kv_reclaim_retained_blocks": self.kv_reclaim_retained_blocks,
            "kv_reclaim_invalidated_tokens": self.kv_reclaim_invalidated_tokens,
            "kv_reclaim_forced_fallbacks": self.kv_reclaim_forced_fallbacks,
            "kv_compression_policy": self.kv_compression_policy,
            "kv_compression_sink_blocks": self.kv_compression_sink_blocks,
            "kv_compression_recent_blocks": self.kv_compression_recent_blocks,
            "kv_compression_importance_blocks": (
                self.kv_compression_importance_blocks
            ),
            "kv_compression_query_tokens": self.kv_compression_query_tokens,
            "kv_compression_trigger_free_ratio": (
                self.kv_compression_trigger_free_ratio
            ),
            "kv_compression_events": self.kv_compression_events,
            "kv_compression_dropped_blocks": self.kv_compression_dropped_blocks,
            "kv_compression_freed_blocks": self.kv_compression_freed_blocks,
            "kv_compression_dropped_tokens": self.kv_compression_dropped_tokens,
        })
        result.update(self.block_manager.cache_metrics())
        result.update({
            "remote_restore_started": self.remote_restore_started,
            "remote_restore_completed": self.remote_restore_completed,
            "remote_restore_failed": self.remote_restore_failed,
            "remote_restore_pending": len(self.pending_restores),
            "cancelled_requests": self.cancelled_requests,
            "waiting_requests": len(self.waiting),
            "running_requests": len(self.running),
        })
        if self.remote_restore_service is not None:
            result.update({
                f"remote_io_{key}": value
                for key, value in self.remote_restore_service.metrics().items()
            })
        return result
