import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    prefix_cache_backend: str = "hash"
    scheduling_policy: str = "fcfs"
    qos_best_effort_slo_ms: float = 60000.0
    qos_priority_boost_ms: float = 50.0
    qos_aging_ms_per_step: float = 1.0
    qos_prefill_ms_per_token: float = 0.05
    qos_decode_ms_per_token: float = 5.0
    qos_ewma_alpha: float = 0.2

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert self.prefix_cache_backend in ("hash", "radix")
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.scheduling_policy in ("fcfs", "pals")
        assert self.qos_best_effort_slo_ms > 0
        assert self.qos_priority_boost_ms >= 0
        assert self.qos_aging_ms_per_step >= 0
        assert self.qos_prefill_ms_per_token > 0
        assert self.qos_decode_ms_per_token > 0
        assert 0 < self.qos_ewma_alpha <= 1
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
