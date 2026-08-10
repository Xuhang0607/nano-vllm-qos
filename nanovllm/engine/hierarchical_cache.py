from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable, Mapping, Optional

from nanovllm.engine.radix_cache import BlockKey, RadixPrefixCache


class CacheTier(str, Enum):
    GPU = "gpu"
    CPU = "cpu"
    MOONCAKE = "mooncake"


@dataclass(slots=True, frozen=True)
class KVCacheGeometry:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype_bytes: int

    def __post_init__(self):
        if min(
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
            self.dtype_bytes,
        ) <= 0:
            raise ValueError("KV cache geometry values must be positive")

    @property
    def bytes_per_token(self):
        return (
            2
            * self.num_layers
            * self.num_kv_heads
            * self.head_dim
            * self.dtype_bytes
        )

    def size_bytes(self, num_tokens: int):
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        return self.bytes_per_token * num_tokens


@dataclass(slots=True, frozen=True)
class StorageTierProfile:
    bandwidth_gbps: float
    fixed_latency_ms: float
    congestion_multiplier: float = 1.0

    def __post_init__(self):
        if self.bandwidth_gbps <= 0:
            raise ValueError("bandwidth_gbps must be positive")
        if self.fixed_latency_ms < 0:
            raise ValueError("fixed_latency_ms must be non-negative")
        if self.congestion_multiplier < 1:
            raise ValueError("congestion_multiplier must be at least 1")

    def transfer_ms(self, size_bytes: int):
        wire_ms = size_bytes / (self.bandwidth_gbps * 1_000_000_000) * 1000.0
        return self.fixed_latency_ms + wire_ms * self.congestion_multiplier


@dataclass(slots=True, frozen=True)
class KVAccessPlan:
    action: str
    source_tier: CacheTier
    local_cached_tokens: int
    source_cached_tokens: int
    restored_tokens: int
    recomputed_tokens: int
    transfer_bytes: int
    transfer_ms: float
    recompute_ms: float
    total_ms: float

    def to_dict(self):
        result = asdict(self)
        result["source_tier"] = self.source_tier.value
        return result


class KVTransferPlanner:
    def __init__(
        self,
        geometry: KVCacheGeometry,
        tier_profiles: Mapping[CacheTier, StorageTierProfile],
    ):
        self.geometry = geometry
        self.tier_profiles = dict(tier_profiles)

    def _recompute_plan(
        self,
        total_tokens: int,
        local_cached_tokens: int,
        prefill_ms_per_token: float,
    ):
        recomputed_tokens = total_tokens - local_cached_tokens
        recompute_ms = recomputed_tokens * prefill_ms_per_token
        return KVAccessPlan(
            action="local" if recomputed_tokens == 0 else "recompute",
            source_tier=CacheTier.GPU,
            local_cached_tokens=local_cached_tokens,
            source_cached_tokens=local_cached_tokens,
            restored_tokens=0,
            recomputed_tokens=recomputed_tokens,
            transfer_bytes=0,
            transfer_ms=0.0,
            recompute_ms=recompute_ms,
            total_ms=recompute_ms,
        )

    def plan(
        self,
        total_tokens: int,
        local_cached_tokens: int,
        cached_tokens_by_tier: Mapping[CacheTier, int],
        prefill_ms_per_token: float,
    ) -> KVAccessPlan:
        if total_tokens < 0:
            raise ValueError("total_tokens must be non-negative")
        if not 0 <= local_cached_tokens <= total_tokens:
            raise ValueError("local_cached_tokens must be within the request length")
        if prefill_ms_per_token <= 0:
            raise ValueError("prefill_ms_per_token must be positive")

        candidates = [
            self._recompute_plan(
                total_tokens, local_cached_tokens, prefill_ms_per_token
            )
        ]
        for tier, source_cached_tokens in cached_tokens_by_tier.items():
            if source_cached_tokens < 0:
                raise ValueError("cached prefix lengths must be non-negative")
            if tier == CacheTier.GPU or source_cached_tokens <= local_cached_tokens:
                continue
            if source_cached_tokens > total_tokens:
                raise ValueError("cached prefix cannot exceed the request length")
            profile = self.tier_profiles.get(tier)
            if profile is None:
                continue
            restored_tokens = source_cached_tokens - local_cached_tokens
            transfer_bytes = self.geometry.size_bytes(restored_tokens)
            transfer_ms = profile.transfer_ms(transfer_bytes)
            recomputed_tokens = total_tokens - source_cached_tokens
            recompute_ms = recomputed_tokens * prefill_ms_per_token
            candidates.append(
                KVAccessPlan(
                    action="restore",
                    source_tier=tier,
                    local_cached_tokens=local_cached_tokens,
                    source_cached_tokens=source_cached_tokens,
                    restored_tokens=restored_tokens,
                    recomputed_tokens=recomputed_tokens,
                    transfer_bytes=transfer_bytes,
                    transfer_ms=transfer_ms,
                    recompute_ms=recompute_ms,
                    total_ms=transfer_ms + recompute_ms,
                )
            )

        return min(
            candidates,
            key=lambda item: (
                item.total_ms,
                item.transfer_bytes,
                -item.source_cached_tokens,
            ),
        )


@dataclass(slots=True, frozen=True)
class TierMatch:
    tier: CacheTier
    handles: tuple[int, ...]

    @property
    def num_blocks(self):
        return len(self.handles)


class HierarchicalRadixCache:
    """Independent radix indexes for local and remote KV cache replicas."""

    def __init__(self):
        self.indexes = {tier: RadixPrefixCache() for tier in CacheTier}

    def insert(
        self,
        tier: CacheTier,
        keys: Iterable[BlockKey],
        handles: Iterable[int],
    ):
        self.indexes[tier].insert(keys, handles)

    def match(self, keys: Iterable[BlockKey]):
        keys = tuple(keys)
        return {
            tier: TierMatch(tier, tuple(index.match(keys)))
            for tier, index in self.indexes.items()
        }

    def remove(self, tier: CacheTier, handle: int):
        return self.indexes[tier].remove_block(handle)

    def longest_match(self, keys: Iterable[BlockKey]) -> Optional[TierMatch]:
        matches = self.match(keys).values()
        tier_order = {
            CacheTier.GPU: 0,
            CacheTier.CPU: 1,
            CacheTier.MOONCAKE: 2,
        }
        best = max(
            matches,
            key=lambda item: (item.num_blocks, -tier_order[item.tier]),
        )
        return best if best.num_blocks else None

    def stats(self):
        return {
            tier.value: index.stats()
            for tier, index in self.indexes.items()
        }

    def validate(self):
        for index in self.indexes.values():
            index.validate()
