from collections import deque
from time import perf_counter
from typing import TYPE_CHECKING

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.qos import ExecutionTimeEstimator, create_policy, summarize_metrics

if TYPE_CHECKING:
    from nanovllm.config import Config


class Scheduler:

    def __init__(self, config: "Config", clock=perf_counter):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
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
        self.policy = create_policy(self.policy_name, self.estimator, config)
        self.completed_metrics = []
        self.metrics_by_seq_id = {}

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        seq.mark_arrived(self.step_id, self.clock())
        self.waiting.append(seq)

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
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
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
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        seq.preemption_count += 1
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        now = self.clock()
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
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
            "prefill_ms_per_token_ewma": self.estimator.prefill_ms_per_token,
            "decode_ms_per_token_ewma": self.estimator.decode_ms_per_token,
        })
        result.update(self.block_manager.cache_metrics())
        return result
