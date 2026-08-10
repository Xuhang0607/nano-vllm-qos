import pytest

from nanovllm.engine.hierarchical_cache import (
    CacheTier,
    HierarchicalRadixCache,
    KVCacheGeometry,
    KVTransferPlanner,
    StorageTierProfile,
)


A = (1, 2)
B = (3, 4)
C = (5, 6)
D = (7, 8)


def make_planner(cpu_bandwidth=25.0, mooncake_bandwidth=12.5):
    geometry = KVCacheGeometry(
        num_layers=28,
        num_kv_heads=8,
        head_dim=128,
        dtype_bytes=2,
    )
    return KVTransferPlanner(
        geometry,
        {
            CacheTier.CPU: StorageTierProfile(cpu_bandwidth, 0.05),
            CacheTier.MOONCAKE: StorageTierProfile(mooncake_bandwidth, 0.3),
        },
    )


def test_kv_geometry_accounts_for_keys_values_layers_and_dtype():
    geometry = KVCacheGeometry(28, 8, 128, 2)
    assert geometry.bytes_per_token == 114688
    assert geometry.size_bytes(1000) == 114688000


def test_planner_restores_remote_prefix_when_transfer_beats_recompute():
    planner = make_planner()
    plan = planner.plan(
        total_tokens=1000,
        local_cached_tokens=100,
        cached_tokens_by_tier={CacheTier.MOONCAKE: 900},
        prefill_ms_per_token=0.08,
    )

    assert plan.action == "restore"
    assert plan.source_tier == CacheTier.MOONCAKE
    assert plan.restored_tokens == 800
    assert plan.recomputed_tokens == 100
    assert plan.total_ms < 900 * 0.08


def test_planner_recomputes_when_remote_tier_is_slow():
    planner = make_planner(mooncake_bandwidth=0.5)
    plan = planner.plan(
        total_tokens=1000,
        local_cached_tokens=100,
        cached_tokens_by_tier={CacheTier.MOONCAKE: 900},
        prefill_ms_per_token=0.08,
    )

    assert plan.action == "recompute"
    assert plan.recomputed_tokens == 900
    assert plan.transfer_bytes == 0


def test_planner_selects_best_of_cpu_mooncake_and_recompute():
    planner = make_planner(cpu_bandwidth=30.0, mooncake_bandwidth=12.5)
    plan = planner.plan(
        total_tokens=1200,
        local_cached_tokens=200,
        cached_tokens_by_tier={
            CacheTier.CPU: 800,
            CacheTier.MOONCAKE: 1100,
        },
        prefill_ms_per_token=0.1,
    )

    assert plan.source_tier == CacheTier.MOONCAKE
    assert plan.source_cached_tokens == 1100


def test_congestion_can_flip_restore_decision_to_recompute():
    geometry = KVCacheGeometry(28, 8, 128, 2)
    planner = KVTransferPlanner(
        geometry,
        {
            CacheTier.MOONCAKE: StorageTierProfile(
                bandwidth_gbps=12.5,
                fixed_latency_ms=0.3,
                congestion_multiplier=20.0,
            )
        },
    )
    plan = planner.plan(
        total_tokens=1000,
        local_cached_tokens=100,
        cached_tokens_by_tier={CacheTier.MOONCAKE: 900},
        prefill_ms_per_token=0.08,
    )
    assert plan.action == "recompute"


def test_hierarchical_radix_cache_keeps_remote_prefix_after_local_eviction():
    cache = HierarchicalRadixCache()
    cache.insert(CacheTier.GPU, [A, B], [1, 2])
    cache.insert(CacheTier.MOONCAKE, [A, B, C, D], [101, 102, 103, 104])

    matches = cache.match([A, B, C, D])
    assert matches[CacheTier.GPU].num_blocks == 2
    assert matches[CacheTier.MOONCAKE].num_blocks == 4
    assert cache.longest_match([A, B, C, D]).tier == CacheTier.MOONCAKE

    assert cache.remove(CacheTier.GPU, 1)
    matches = cache.match([A, B, C, D])
    assert matches[CacheTier.GPU].num_blocks == 0
    assert matches[CacheTier.MOONCAKE].num_blocks == 4
    cache.validate()


def test_invalid_geometry_and_profiles_are_rejected():
    with pytest.raises(ValueError):
        KVCacheGeometry(0, 8, 128, 2)
    with pytest.raises(ValueError):
        StorageTierProfile(0, 0.1)
