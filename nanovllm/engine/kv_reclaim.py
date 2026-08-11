from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanovllm.engine.sequence import Sequence


@dataclass(slots=True, frozen=True)
class KVReclaimDecision:
    policy: str
    budget_ms: float | None
    urgency: float
    owned_blocks: int
    cached_blocks: int
    keep_blocks: int
    reclaim_blocks: int
    invalidated_tokens: int


class KVReclaimPolicy:
    name = "recompute"

    def plan(
        self,
        seq: Sequence,
        budget_ms: float | None,
        min_reclaim_blocks: int = 1,
    ) -> KVReclaimDecision:
        cached_blocks = seq.num_cached_tokens // seq.block_size
        return self._decision(seq, budget_ms, 0.0, cached_blocks, 0)

    def _decision(
        self,
        seq: Sequence,
        budget_ms: float | None,
        urgency: float,
        cached_blocks: int,
        keep_blocks: int,
    ) -> KVReclaimDecision:
        owned_blocks = len(seq.block_table)
        keep_blocks = max(0, min(keep_blocks, cached_blocks, owned_blocks))
        retained_tokens = keep_blocks * seq.block_size
        return KVReclaimDecision(
            policy=self.name,
            budget_ms=budget_ms,
            urgency=urgency,
            owned_blocks=owned_blocks,
            cached_blocks=cached_blocks,
            keep_blocks=keep_blocks,
            reclaim_blocks=owned_blocks - keep_blocks,
            invalidated_tokens=max(0, seq.num_cached_tokens - retained_tokens),
        )


class SLOAwareKVReclaimPolicy(KVReclaimPolicy):
    name = "slo_aware"

    def __init__(
        self,
        min_keep_ratio: float,
        max_keep_ratio: float,
        budget_scale_ms: float,
    ):
        if not 0.0 <= min_keep_ratio <= max_keep_ratio <= 1.0:
            raise ValueError("KV keep ratios must satisfy 0 <= min <= max <= 1")
        if budget_scale_ms <= 0:
            raise ValueError("KV reclaim budget scale must be positive")
        self.min_keep_ratio = min_keep_ratio
        self.max_keep_ratio = max_keep_ratio
        self.budget_scale_ms = budget_scale_ms

    def plan(
        self,
        seq: Sequence,
        budget_ms: float | None,
        min_reclaim_blocks: int = 1,
    ) -> KVReclaimDecision:
        owned_blocks = len(seq.block_table)
        cached_blocks = seq.num_cached_tokens // seq.block_size
        if budget_ms is None:
            urgency = 0.0
        elif budget_ms <= 0:
            urgency = 1.0
        else:
            urgency = 1.0 / (1.0 + budget_ms / self.budget_scale_ms)

        keep_ratio = self.min_keep_ratio + (
            self.max_keep_ratio - self.min_keep_ratio
        ) * urgency
        desired_keep = floor(cached_blocks * keep_ratio)
        max_keep = max(0, owned_blocks - min_reclaim_blocks)
        keep_blocks = min(desired_keep, max_keep)
        return self._decision(
            seq,
            budget_ms,
            urgency,
            cached_blocks,
            keep_blocks,
        )


def create_kv_reclaim_policy(name: str, config) -> KVReclaimPolicy:
    if name == "recompute":
        return KVReclaimPolicy()
    if name == "slo_aware":
        return SLOAwareKVReclaimPolicy(
            min_keep_ratio=getattr(config, "kv_reclaim_min_keep_ratio", 0.0),
            max_keep_ratio=getattr(config, "kv_reclaim_max_keep_ratio", 0.75),
            budget_scale_ms=getattr(config, "kv_reclaim_budget_scale_ms", 1000.0),
        )
    raise ValueError(f"unsupported KV reclaim policy: {name}")
