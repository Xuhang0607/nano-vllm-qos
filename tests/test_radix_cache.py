from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.radix_cache import RadixPrefixCache
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


A = (1, 2)
B = (3, 4)
C = (5, 6)
D = (7, 8)
E = (9, 10)


def test_radix_cache_splits_branches_and_matches_longest_prefix():
    cache = RadixPrefixCache()
    cache.insert([A, B, C], [1, 2, 3])
    cache.insert([A, B, D], [1, 2, 4])

    assert cache.match([A, B, C]) == [1, 2, 3]
    assert cache.match([A, B, D]) == [1, 2, 4]
    assert cache.match([A, B, E]) == [1, 2]
    assert cache.match([A]) == [1]
    cache.validate()


def test_radix_cache_does_not_split_for_an_existing_shorter_prefix():
    cache = RadixPrefixCache()
    cache.insert([A, B, C], [1, 2, 3])
    cache.insert([A, B], [1, 2])

    assert cache.match([A, B, C]) == [1, 2, 3]
    assert cache.stats()["radix_nodes"] == 1
    cache.validate()


def test_concurrent_duplicate_prefix_uses_canonical_blocks():
    cache = RadixPrefixCache()
    cache.insert([A, B, C], [1, 2, 3])
    cache.insert([A, B, D], [11, 12, 4])

    assert cache.match([A, B, C]) == [1, 2, 3]
    assert cache.match([A, B, D]) == [1, 2, 4]
    assert 11 not in cache.block_locations
    assert 12 not in cache.block_locations
    cache.validate()


def test_removing_shared_block_invalidates_descendants_only():
    cache = RadixPrefixCache()
    cache.insert([A, B, C], [1, 2, 3])
    cache.insert([A, B, D], [1, 2, 4])

    assert cache.remove_block(2)
    assert cache.match([A, B, C]) == [1]
    assert cache.match([A, B, D]) == [1]
    assert not cache.remove_block(999)
    cache.validate()


def test_removing_one_branch_recompresses_the_remaining_path():
    cache = RadixPrefixCache()
    cache.insert([A, B, C], [1, 2, 3])
    cache.insert([A, B, D], [1, 2, 4])

    assert cache.remove_block(3)
    assert cache.match([A, B, C]) == [1, 2]
    assert cache.match([A, B, D]) == [1, 2, 4]
    assert cache.stats()["radix_nodes"] == 1
    cache.validate()


def test_lru_eviction_selects_the_oldest_leaf_tail():
    cache = RadixPrefixCache()
    cache.insert([A, B, C], [1, 2, 3])
    cache.insert([A, B, D], [1, 2, 4])
    cache.match([A, B, C])

    assert cache.lru_leaf_block({1, 2, 3, 4}) == 4
    assert cache.lru_leaf_block({1, 2, 3}) == 3


def make_sequence(tokens):
    return Sequence(
        tokens,
        SamplingParams(max_tokens=1, ignore_eos=True),
    )


def populate_cache(manager: BlockManager, sequence: Sequence):
    assert manager.can_allocate(sequence) == 0
    manager.allocate(sequence, 0)
    sequence.num_scheduled_tokens = sequence.num_tokens
    manager.hash_blocks(sequence)
    cached_block_ids = list(sequence.block_table)
    sequence.num_cached_tokens = sequence.num_tokens
    sequence.num_scheduled_tokens = 0
    manager.deallocate(sequence)
    return cached_block_ids


def test_block_manager_reuses_radix_prefix_blocks():
    Sequence.block_size = 2
    manager = BlockManager(8, 2, prefix_cache_backend="radix")
    first = make_sequence([1, 2, 3, 4, 5])
    first_block_ids = populate_cache(manager, first)

    second = make_sequence([1, 2, 3, 4, 9])
    assert manager.can_allocate(second) == 2
    manager.allocate(second, 2)

    assert second.num_cached_tokens == 4
    assert second.block_table[:2] == first_block_ids[:2]
    metrics = manager.cache_metrics()
    assert metrics["prefix_cache_hit_blocks"] == 2
    assert metrics["prefix_cache_block_hit_rate"] == 0.5
    manager.radix_cache.validate()


def test_block_manager_handles_concurrent_duplicate_prefix_misses():
    Sequence.block_size = 2
    manager = BlockManager(12, 2, prefix_cache_backend="radix")
    first = make_sequence([1, 2, 3, 4, 5])
    second = make_sequence([1, 2, 7, 8, 9])

    assert manager.can_allocate(first) == 0
    manager.allocate(first, 0)
    assert manager.can_allocate(second) == 0
    manager.allocate(second, 0)
    for sequence in (first, second):
        sequence.num_scheduled_tokens = sequence.num_tokens
        manager.hash_blocks(sequence)

    probe = make_sequence([1, 2, 7, 8, 99])
    assert manager.can_allocate(probe) == 2
    manager.radix_cache.validate()


def test_reallocated_physical_block_is_removed_from_radix_index():
    Sequence.block_size = 2
    manager = BlockManager(3, 2, prefix_cache_backend="radix")
    first = make_sequence([1, 2, 3, 4, 5])
    populate_cache(manager, first)

    replacement = make_sequence([11, 12, 13, 14, 15])
    assert manager.can_allocate(replacement) == 0
    manager.allocate(replacement, 0)
    manager.deallocate(replacement)

    probe = make_sequence([1, 2, 3, 4, 99])
    assert manager.can_allocate(probe) == 0
    manager.radix_cache.validate()


def test_block_manager_prefers_uncached_then_lru_leaf_blocks():
    Sequence.block_size = 2
    manager = BlockManager(4, 2, prefix_cache_backend="radix")
    first = make_sequence([1, 2, 3, 4, 5])
    populate_cache(manager, first)

    first_allocated = manager._allocate_block()
    second_allocated = manager._allocate_block()
    third_allocated = manager._allocate_block()

    assert {first_allocated, second_allocated} == {2, 3}
    assert third_allocated == 1
    assert manager.cache_evictions == 1
