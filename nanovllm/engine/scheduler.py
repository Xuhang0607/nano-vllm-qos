from collections import deque
from dataclasses import dataclass
from itertools import islice
from time import perf_counter
from typing import TYPE_CHECKING

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.kv_admission import priority_reserve_pages
from nanovllm.engine.kv_reclaim import (
    KVReclaimDecision,
    KVReclaimPolicy,
    create_kv_reclaim_policy,
)
from nanovllm.engine.qos import ExecutionTimeEstimator, create_policy, summarize_metrics
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler_trace import SchedulerTrace

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
        self.max_num_active_seqs = getattr(config, "max_num_active_seqs", None)
        self.kv_admission_lookahead = getattr(config, "kv_admission_lookahead", 0)
        if self.kv_admission_lookahead < 0:
            raise ValueError("kv_admission_lookahead must be non-negative")
        self.kv_admission_deferred_checks = 0
        if self.max_num_active_seqs is not None and self.max_num_active_seqs < 1:
            raise ValueError("max_num_active_seqs must be positive")
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
        self.metrics_summary_mode = getattr(config, "metrics_summary_mode", "cached")
        if self.metrics_summary_mode not in ("cached", "full"):
            raise ValueError("metrics_summary_mode must be cached or full")
        self._completed_summary = {}
        self._completed_summary_count = -1
        self._summary_calls = 0
        self._summary_refreshes = 0
        self._summary_elapsed_ms = 0.0
        self.metrics_by_seq_id = {}
        self.remote_restore_service = remote_restore_service
        self.pending_restores: dict[int, PendingRestoreState] = {}
        self.remote_restore_started = 0
        self.remote_restore_completed = 0
        self.remote_restore_failed = 0
        self.cancelled_requests = 0
        trace_path = getattr(config, "scheduler_trace_path", None)
        self.trace = SchedulerTrace(trace_path, {
            "policy": self.policy_name,
            "kv_reclaim_policy": self.kv_reclaim_policy_name,
            "kv_compression_policy": self.kv_compression_policy,
            "prefix_cache_backend": self.block_manager.prefix_cache_backend,
            "block_size": self.block_size,
            "num_blocks": len(self.block_manager.blocks),
            "max_num_seqs": self.max_num_seqs,
            "max_num_active_seqs": self.max_num_active_seqs,
            "kv_admission_lookahead": self.kv_admission_lookahead,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "sequence_snapshot_limit": 64,
        }) if trace_path else None

    def _trace(self, event, seqs=(), *, now=None, trigger=False, **details):
        if self.trace is None or not self.trace.enabled:
            return
        now = self.clock() if now is None else now

        def snapshot(seq):
            return {
                "seq_id": seq.seq_id, "status": seq.status.name,
                "priority": seq.qos.priority, "request_class": seq.qos.request_class,
                "age_ms": max(0.0, (now - seq.arrival_time) * 1000),
                "budget_ms": self.policy.budget_ms(seq, self.step_id, now),
                "tokens": seq.num_tokens, "completion_tokens": seq.num_completion_tokens,
                "max_tokens": seq.max_tokens, "cached_tokens": seq.num_cached_tokens,
                "physical_cached_tokens": seq.num_physical_cached_tokens,
                "owned_blocks": len(seq.block_table),
                "scheduled_tokens": seq.num_scheduled_tokens,
                "next_decode_needs_page": len(seq) % self.block_size == 1,
                "preemptions": seq.preemption_count,
                "recomputed_tokens": seq.kv_recomputed_tokens,
                "compressed": seq.kv_compressed,
            }

        self.trace.record({
            "event": event, "step": self.step_id, "time_s": now,
            "free_blocks": len(self.block_manager.free_block_ids),
            "waiting_count": len(self.waiting), "running_count": len(self.running),
            "pending_restore_count": len(self.pending_restores),
            "prefill_ms_per_token": self.estimator.prefill_ms_per_token,
            "decode_ms_per_token": self.estimator.decode_ms_per_token,
            "waiting": [snapshot(seq) for seq in islice(self.waiting, 64)],
            "running": [snapshot(seq) for seq in islice(self.running, 64)],
            "selected": [snapshot(seq) for seq in islice(seqs, 64)],
            **details,
        }, trigger=trigger)

    def is_finished(self):
        return not self.waiting and not self.running and not self.pending_restores

    def _resident_count(self):
        # Partial prefill, retained waiting prefixes and restores also own KV.
        return (len(self.running) + len(self.pending_restores)
                + sum(bool(seq.block_table) for seq in self.waiting))

    def add(self, seq: Sequence):
        seq.mark_arrived(self.step_id, self.clock())
        self.waiting.append(seq)

        self._trace("arrival", (seq,))

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
        self._trace("cancel", (seq,))
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
        self._trace("reclaim_plan", (seq,), trigger=True,
                    keep_blocks=decision.keep_blocks,
                    reclaim_blocks=decision.reclaim_blocks,
                    invalidated_tokens=decision.invalidated_tokens)
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
        self._trace("reclaim_applied", (seq,), freed_blocks=freed_blocks)
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
        result = self._schedule_once()
        if result is None:
            # The last decode request released its KV; retry the prefill phase.
            result = self._schedule_once()
        if result is None:
            raise RuntimeError("scheduler retry failed to make progress")
        return result

    def _schedule_once(self) -> tuple[list[Sequence], bool] | None:
        self.step_id += 1
        now = self.clock()
        scheduled_seqs = []
        num_batched_tokens = 0
        budget_deferred = set()
        resident_progress_only = False
        retry_prefill = False

        # Compare the most urgent prefill and decode requests before choosing a phase.
        schedule_prefill = self.policy.should_schedule_prefill(
            self.waiting, self.running, self.step_id, now
        )
        self._trace("schedule_start", now=now, prefer_prefill=schedule_prefill)
        while schedule_prefill and self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            eligible = self.waiting
            if resident_progress_only:
                eligible = [item for item in eligible if item.block_table]
                if not eligible:
                    break
            if budget_deferred:
                eligible = [item for item in eligible if item.seq_id not in budget_deferred]
                if not eligible or not self.policy.should_schedule_prefill(
                    eligible, self.running, self.step_id, now
                ):
                    break
            if (self.max_num_active_seqs is not None
                    and self._resident_count() >= self.max_num_active_seqs):
                eligible = [item for item in eligible if item.block_table]
                if not self.policy.should_schedule_prefill(
                    eligible, self.running, self.step_id, now
                ):
                    self._trace("admission_deferred", now=now,
                                reason="resident_limit", resident_count=self._resident_count())
                    break
            seq = self.policy.select_waiting(eligible, self.step_id, now)
            seq.last_budget_ms = self.policy.budget_ms(seq, self.step_id, now)
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if seq.block_table and not self._ensure_resume_capacity(seq, now):
                self._trace("prefill_blocked", (seq,), now=now, reason="resume_capacity")
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    self._trace("prefill_blocked", (seq,), now=now, reason="allocation_capacity")
                    # With no decode work, an allocated partial prefill must be
                    # allowed to progress instead of failing behind a new request.
                    if not self.running and any(item.block_table for item in self.waiting):
                        resident_progress_only = True
                        self._trace("prefill_progress_fallback", (seq,), now=now)
                        continue
                    if self.kv_admission_lookahead and self.running:
                        budget_deferred.add(seq.seq_id)
                        self.kv_admission_deferred_checks += 1
                        if len(budget_deferred) < self.max_num_seqs:
                            continue
                    break
                if self.kv_admission_lookahead and self.running:
                    required = self.block_manager.planned_allocation_pages(seq)
                    reserve = priority_reserve_pages(
                        seq, self.running, self.kv_admission_lookahead, self.block_size
                    )
                    if len(self.block_manager.free_block_ids) - required < reserve:
                        self.block_manager.pending_matches.pop(seq.seq_id, None)
                        self.kv_admission_deferred_checks += 1
                        budget_deferred.add(seq.seq_id)
                        self._trace("admission_deferred", (seq,), now=now,
                                    reason="priority_page_reserve", required_pages=required,
                                    reserved_pages=reserve)
                        if len(budget_deferred) >= self.max_num_seqs:
                            break
                        continue
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
            self._trace("prefill_admitted", (seq,), now=now,
                        batched_tokens=num_batched_tokens, remaining_token_budget=remaining)

        if scheduled_seqs:
            self._trace("schedule_end", scheduled_seqs, now=now, phase="prefill")
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
                    self._trace("append_pressure", (seq, victim), now=now, trigger=True,
                                reason="decode_page_exhausted", victim_id=victim.seq_id,
                                requester_id=seq.seq_id)
                    self.running.remove(victim)
                    self.preempt(victim)
                else:
                    self._trace("append_pressure", (seq,), now=now, trigger=True,
                                reason="decode_self_preemption", victim_id=seq.seq_id,
                                requester_id=seq.seq_id)
                    self.preempt(seq)
                    retry_prefill = True
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        if self.pending_restores and not scheduled_seqs:
            return [], True
        if not scheduled_seqs:
            if retry_prefill:
                return None
            self._trace("no_runnable", now=now, trigger=True)
            raise RuntimeError("no runnable request fits in the available KV cache")
        self.running.extendleft(reversed(scheduled_seqs))
        self._trace("schedule_end", scheduled_seqs, now=now, phase="decode")
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
            self._trace("preempted", (seq,), freed_blocks=freed_blocks)
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
        self._trace("preempted", (seq,), keep_blocks=decision.keep_blocks,
                    invalidated_tokens=decision.invalidated_tokens)

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
                self._trace("finished", (seq,), now=now)

    def observe_execution(self, is_prefill: bool, num_tokens: int, elapsed_ms: float):
        self.estimator.observe(is_prefill, num_tokens, elapsed_ms)
        self._trace("execution", phase="prefill" if is_prefill else "decode",
                    num_tokens=num_tokens, elapsed_ms=elapsed_ms)

    def request_metrics(self, seq_id: int):
        return self.metrics_by_seq_id.get(seq_id)

    def metrics(self):
        self._summary_calls += 1
        count = len(self.completed_metrics)
        # Completed RequestMetrics are immutable and this history is append-only.
        # Live queue/cache counters below must never be served from this cache.
        if self.metrics_summary_mode == "full" or count != self._completed_summary_count:
            started = perf_counter()
            self._completed_summary = summarize_metrics(self.completed_metrics)
            self._summary_elapsed_ms += (perf_counter() - started) * 1000.0
            self._completed_summary_count = count
            self._summary_refreshes += 1
        result = self._completed_summary.copy()
        result.update({
            "metrics_summary_mode": self.metrics_summary_mode,
            "scheduler_trace_enabled": self.trace is not None,
            "metrics_summary_calls": self._summary_calls,
            "metrics_summary_refreshes": self._summary_refreshes,
            "metrics_summary_compute_ms": self._summary_elapsed_ms,
            "policy": self.policy_name,
            "max_model_len": self.max_model_len,
            "requested_max_model_len": self.requested_max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "max_num_active_seqs": self.max_num_active_seqs,
            "kv_admission_lookahead": self.kv_admission_lookahead,
            "kv_admission_deferred_checks": self.kv_admission_deferred_checks,
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
