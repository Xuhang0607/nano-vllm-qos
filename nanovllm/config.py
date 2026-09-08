from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import AutoConfig


def effective_context_window(max_model_len: int, num_blocks: int, block_size: int):
    if max_model_len <= 0 or num_blocks <= 0 or block_size <= 0:
        raise ValueError("context and KV cache dimensions must be positive")
    return min(max_model_len, num_blocks * block_size)


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_num_active_seqs: int | None = None
    kv_admission_lookahead: int = 0
    max_model_len: int = 40960
    requested_max_model_len: int = field(init=False)
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    num_kvcache_blocks_override: int | None = None
    prefix_cache_backend: str = "hash"
    scheduling_policy: str = "fcfs"
    metrics_summary_mode: str = "cached"
    scheduler_trace_path: str | None = None
    qos_best_effort_slo_ms: float = 60000.0
    qos_priority_boost_ms: float = 50.0
    qos_aging_ms_per_step: float = 1.0
    qos_prefill_ms_per_token: float = 0.05
    qos_decode_ms_per_token: float = 5.0
    qos_ewma_alpha: float = 0.2
    kv_reclaim_policy: str = "slo_aware"
    kv_reclaim_min_keep_ratio: float = 0.0
    kv_reclaim_max_keep_ratio: float = 0.75
    kv_reclaim_budget_scale_ms: float = 1000.0
    kv_reclaim_target_free_blocks: int = 2
    kv_compression_policy: str = "none"
    kv_compression_sink_blocks: int = 1
    kv_compression_recent_blocks: int = 8
    kv_compression_importance_blocks: int = 2
    kv_compression_query_tokens: int = 64
    kv_compression_trigger_free_ratio: float = 0.15
    kv_cache_model_id: str | None = None
    kv_cache_model_revision: str = "local"
    remote_kv_cost_aware: bool = True
    remote_kv_bandwidth_gbps: float = 12.5
    remote_kv_fixed_latency_ms: float = 0.3
    remote_kv_congestion_multiplier: float = 1.0
    remote_kv_writeback: bool = True
    remote_kv_min_prefix_blocks: int = 1
    remote_kv_persist_catalog: bool = True
    remote_kv_catalog_timeout_s: float = 10.0

    def __post_init__(self):
        from transformers import AutoConfig

        assert os.path.isdir(self.model)
        assert self.max_num_active_seqs is None or self.max_num_active_seqs > 0
        assert self.kv_admission_lookahead >= 0
        assert self.kvcache_block_size % 256 == 0
        assert (
            self.num_kvcache_blocks_override is None
            or self.num_kvcache_blocks_override > 0
        )
        assert self.prefix_cache_backend in ("hash", "radix")
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.scheduling_policy in ("fcfs", "pals")
        assert self.metrics_summary_mode in ("cached", "full")
        assert self.qos_best_effort_slo_ms > 0
        assert self.qos_priority_boost_ms >= 0
        assert self.qos_aging_ms_per_step >= 0
        assert self.qos_prefill_ms_per_token > 0
        assert self.qos_decode_ms_per_token > 0
        assert 0 < self.qos_ewma_alpha <= 1
        assert self.kv_reclaim_policy in ("recompute", "slo_aware")
        assert (
            0
            <= self.kv_reclaim_min_keep_ratio
            <= self.kv_reclaim_max_keep_ratio
            <= 1
        )
        assert self.kv_reclaim_budget_scale_ms > 0
        assert self.kv_reclaim_target_free_blocks >= 1
        assert self.kv_compression_policy in ("none", "sink_recent", "query_aware")
        assert self.kv_compression_sink_blocks >= 1
        assert self.kv_compression_recent_blocks >= 1
        assert self.kv_compression_importance_blocks >= 1
        assert self.kv_compression_query_tokens >= 1
        assert 0 <= self.kv_compression_trigger_free_ratio <= 1
        assert self.kv_cache_model_revision
        assert self.remote_kv_bandwidth_gbps > 0
        assert self.remote_kv_fixed_latency_ms >= 0
        assert self.remote_kv_congestion_multiplier >= 1
        assert self.remote_kv_min_prefix_blocks >= 1
        assert self.remote_kv_catalog_timeout_s > 0
        self.requested_max_model_len = self.max_model_len
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if self.kv_cache_model_id is None:
            self.kv_cache_model_id = getattr(
                self.hf_config,
                "_name_or_path",
                self.model,
            )
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )
