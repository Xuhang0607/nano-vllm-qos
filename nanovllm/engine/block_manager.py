from collections import deque

import numpy as np
import xxhash

from nanovllm.engine.radix_cache import RadixPrefixCache
from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        prefix_cache_backend: str = "hash",
    ):
        if prefix_cache_backend not in ("hash", "radix"):
            raise ValueError(f"unsupported prefix cache backend: {prefix_cache_backend}")
        self.block_size = block_size
        self.prefix_cache_backend = prefix_cache_backend
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.radix_cache = RadixPrefixCache() if prefix_cache_backend == "radix" else None
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.pending_matches: dict[int, list[int]] = {}
        self.pending_resume_matches: dict[int, list[int]] = {}
        self.cache_lookups = 0
        self.cache_queried_blocks = 0
        self.cache_hit_blocks = 0
        self.cache_evictions = 0

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        if self.radix_cache is None:
            block_id = self.free_block_ids.popleft()
        else:
            block_id = next(
                (
                    candidate
                    for candidate in self.free_block_ids
                    if candidate not in self.radix_cache.block_locations
                ),
                None,
            )
            if block_id is None:
                block_id = self.radix_cache.lru_leaf_block(set(self.free_block_ids))
            if block_id is None:
                block_id = self.free_block_ids[0]
            self.free_block_ids.remove(block_id)
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if self.prefix_cache_backend == "hash":
            if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
                del self.hash_to_block_id[block.hash]
        else:
            if self.radix_cache.remove_block(block_id):
                self.cache_evictions += 1
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def _lookup_cached_blocks(self, seq: Sequence) -> list[int]:
        num_cacheable_blocks = max(0, seq.num_blocks - 1)
        self.cache_lookups += 1
        self.cache_queried_blocks += num_cacheable_blocks

        if self.prefix_cache_backend == "radix":
            keys = [tuple(seq.block(i)) for i in range(num_cacheable_blocks)]
            matched = self.radix_cache.match(keys)
        else:
            h = -1
            matched = []
            for i in range(num_cacheable_blocks):
                token_ids = seq.block(i)
                h = self.compute_hash(token_ids, h)
                block_id = self.hash_to_block_id.get(h, -1)
                if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                    break
                matched.append(block_id)

        self.cache_hit_blocks += len(matched)
        return matched

    def can_allocate(self, seq: Sequence) -> int:
        cached_block_ids = self._lookup_cached_blocks(seq)
        num_new_blocks = seq.num_blocks
        for block_id in cached_block_ids:
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            self.pending_matches.pop(seq.seq_id, None)
            return -1
        self.pending_matches[seq.seq_id] = cached_block_ids
        return len(cached_block_ids)

    def cached_prefix_blocks(self, seq: Sequence) -> int:
        return len(self._lookup_cached_blocks(seq))

    def planned_allocation_pages(self, seq: Sequence) -> int:
        """Free-pool cost of a successful can_allocate plan, without another lookup."""
        matched = self.pending_matches[seq.seq_id]
        return seq.num_blocks - sum(block_id in self.used_block_ids for block_id in matched)

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        cached_block_ids = self.pending_matches.pop(seq.seq_id, None)
        if cached_block_ids is None or len(cached_block_ids) != num_cached_blocks:
            cached_block_ids = self._lookup_cached_blocks(seq)
        if len(cached_block_ids) != num_cached_blocks:
            raise RuntimeError("prefix cache changed between capacity check and allocation")
        for block_id in cached_block_ids:
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.kv_block_logical_indices = list(range(seq.num_blocks))
        seq.num_cached_tokens = num_cached_blocks * self.block_size
        seq.num_physical_cached_tokens = seq.num_cached_tokens

    def reserve_restore(
        self,
        seq: Sequence,
        num_local_cached_blocks: int,
        num_remote_cached_blocks: int,
    ) -> tuple[int, ...]:
        if not num_local_cached_blocks < num_remote_cached_blocks < seq.num_blocks:
            raise ValueError(
                "restored prefix must extend the local cache and leave a writable tail"
            )
        self.allocate(seq, num_local_cached_blocks)
        return tuple(
            seq.block_table[num_local_cached_blocks:num_remote_cached_blocks]
        )

    def commit_restored_prefix(self, seq: Sequence, num_cached_blocks: int):
        current_blocks = seq.num_cached_tokens // self.block_size
        if not current_blocks < num_cached_blocks < seq.num_blocks:
            raise ValueError("restored prefix length is invalid")
        if len(seq.block_table) != seq.num_blocks:
            raise RuntimeError("restored request does not own a complete block table")

        if self.prefix_cache_backend == "radix":
            for index in range(current_blocks, num_cached_blocks):
                block = self.blocks[seq.block_table[index]]
                block.update(-1, seq.block(index))
            keys = [tuple(seq.block(index)) for index in range(num_cached_blocks)]
            self.radix_cache.insert(keys, seq.block_table[:num_cached_blocks])
        else:
            previous_hash = -1
            for index in range(num_cached_blocks):
                block = self.blocks[seq.block_table[index]]
                token_ids = seq.block(index)
                previous_hash = self.compute_hash(token_ids, previous_hash)
                block.update(previous_hash, token_ids)
                self.hash_to_block_id[previous_hash] = block.block_id
        seq.num_cached_tokens = num_cached_blocks * self.block_size
        seq.num_physical_cached_tokens = seq.num_cached_tokens

    def abort_restore(self, seq: Sequence):
        if seq.block_table:
            self.deallocate(seq)

    def deallocate(self, seq: Sequence):
        self.pending_resume_matches.pop(seq.seq_id, None)
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.num_physical_cached_tokens = 0
        seq.kv_compressed = False
        seq.block_table.clear()
        seq.kv_block_logical_indices.clear()

    def reclaim_suffix(self, seq: Sequence, keep_blocks: int) -> int:
        """Release a suffix while preserving a complete, contiguous KV prefix."""
        if not 0 <= keep_blocks <= len(seq.block_table):
            raise ValueError("keep_blocks must be within the owned block table")
        if keep_blocks * self.block_size > seq.num_cached_tokens:
            raise ValueError("cannot retain blocks containing uncomputed KV tokens")

        free_before = len(self.free_block_ids)
        self.pending_resume_matches.pop(seq.seq_id, None)
        released = seq.block_table[keep_blocks:]
        for block_id in reversed(released):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        del seq.block_table[keep_blocks:]
        del seq.kv_block_logical_indices[keep_blocks:]
        seq.num_cached_tokens = keep_blocks * self.block_size
        seq.num_physical_cached_tokens = seq.num_cached_tokens
        return len(self.free_block_ids) - free_before

    def can_resume(self, seq: Sequence) -> bool:
        missing_blocks = seq.num_blocks - len(seq.block_table)
        if missing_blocks < 0:
            raise RuntimeError("request owns more KV blocks than its sequence needs")
        cached_block_ids = self._lookup_cached_blocks(seq)
        retained_blocks = len(seq.block_table)
        if cached_block_ids[:retained_blocks] != seq.block_table:
            cached_suffix = []
        else:
            cached_suffix = cached_block_ids[
                retained_blocks : retained_blocks + missing_blocks
            ]
        already_used = sum(
            block_id in self.used_block_ids for block_id in cached_suffix
        )
        if len(self.free_block_ids) < missing_blocks - already_used:
            self.pending_resume_matches.pop(seq.seq_id, None)
            return False
        self.pending_resume_matches[seq.seq_id] = cached_suffix
        return True

    def resume(self, seq: Sequence) -> int:
        cached_suffix = self.pending_resume_matches.pop(seq.seq_id, None)
        if cached_suffix is None:
            if not self.can_resume(seq):
                raise RuntimeError("insufficient KV blocks to resume request")
            cached_suffix = self.pending_resume_matches.pop(seq.seq_id)
        for block_id in cached_suffix:
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
                block.ref_count = 1
            seq.block_table.append(block_id)
            seq.kv_block_logical_indices.append(
                len(seq.kv_block_logical_indices)
            )
        reused_tokens = len(cached_suffix) * self.block_size
        seq.num_cached_tokens += reused_tokens
        seq.num_physical_cached_tokens += reused_tokens
        while len(seq.block_table) < seq.num_blocks:
            seq.block_table.append(self._allocate_block())
            seq.kv_block_logical_indices.append(
                len(seq.kv_block_logical_indices)
            )
        return reused_tokens

    def _compress_keep_positions(
        self,
        seq: Sequence,
        keep_positions: set[int],
    ) -> tuple[int, int, int]:
        dropped_positions = [
            index
            for index in range(len(seq.block_table))
            if index not in keep_positions
        ]
        if not dropped_positions:
            return 0, 0, 0
        free_before = len(self.free_block_ids)
        for index in reversed(dropped_positions):
            block_id = seq.block_table[index]
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.block_table = [
            block_id
            for index, block_id in enumerate(seq.block_table)
            if index in keep_positions
        ]
        seq.kv_block_logical_indices = [
            logical_index
            for index, logical_index in enumerate(seq.kv_block_logical_indices)
            if index in keep_positions
        ]
        dropped_tokens = min(
            len(dropped_positions) * self.block_size,
            seq.num_physical_cached_tokens,
        )
        seq.num_physical_cached_tokens -= dropped_tokens
        seq.kv_compressed = True
        return (
            len(dropped_positions),
            len(self.free_block_ids) - free_before,
            dropped_tokens,
        )

    def compress_sink_recent(
        self,
        seq: Sequence,
        sink_blocks: int,
        recent_blocks: int,
    ) -> tuple[int, int, int]:
        """Drop full middle KV pages while retaining sink and recent pages."""
        if sink_blocks < 1 or recent_blocks < 1:
            raise ValueError("sink and recent windows must each keep at least one block")
        keep_limit = sink_blocks + recent_blocks
        if len(seq.block_table) <= keep_limit:
            return 0, 0, 0

        recent_start = max(sink_blocks, len(seq.block_table) - recent_blocks)
        dropped_positions = set(range(sink_blocks, recent_start))
        if not dropped_positions:
            return 0, 0, 0
        keep_positions = set(range(len(seq.block_table))) - dropped_positions
        return self._compress_keep_positions(seq, keep_positions)

    def compress_query_aware(
        self,
        seq: Sequence,
        sink_blocks: int,
        recent_blocks: int,
        importance_blocks: int,
        query_tokens: int,
    ) -> tuple[int, int, int]:
        if importance_blocks < 1 or query_tokens < 1:
            raise ValueError("importance blocks and query token window must be positive")
        base_keep = sink_blocks + recent_blocks
        if len(seq.block_table) <= base_keep + importance_blocks:
            return 0, 0, 0

        sink_positions = set(range(min(sink_blocks, len(seq.block_table))))
        recent_start = max(len(sink_positions), len(seq.block_table) - recent_blocks)
        recent_positions = set(range(recent_start, len(seq.block_table)))
        candidate_positions = [
            index
            for index in range(len(seq.block_table))
            if index not in sink_positions and index not in recent_positions
        ]
        query_ids = set(seq.prompt_token_ids[-query_tokens:])
        document_frequency = {}
        candidate_tokens = {}
        for position in candidate_positions:
            logical_index = seq.kv_block_logical_indices[position]
            tokens = set(seq.block(logical_index))
            candidate_tokens[position] = tokens
            for token_id in tokens:
                document_frequency[token_id] = document_frequency.get(token_id, 0) + 1

        scored = []
        for position, tokens in candidate_tokens.items():
            score = sum(
                1.0 / document_frequency[token_id]
                for token_id in tokens & query_ids
            )
            scored.append((score, position))
        # Fill the configured budget even without matches; ties prefer recent history.
        important_positions = {
            position
            for score, position in sorted(
                scored, reverse=True
            )[:importance_blocks]
        }
        keep_positions = sink_positions | recent_positions | important_positions
        return self._compress_keep_positions(seq, keep_positions)

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())
            seq.kv_block_logical_indices.append(seq.num_blocks - 1)

    def hash_blocks(self, seq: Sequence):
        if seq.kv_compressed:
            return
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        if self.prefix_cache_backend == "radix":
            for i in range(start, end):
                block = self.blocks[seq.block_table[i]]
                block.update(-1, seq.block(i))
            keys = [tuple(seq.block(i)) for i in range(end)]
            self.radix_cache.insert(keys, seq.block_table[:end])
        else:
            h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
            for i in range(start, end):
                block = self.blocks[seq.block_table[i]]
                token_ids = seq.block(i)
                h = self.compute_hash(token_ids, h)
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block.block_id

    def cache_metrics(self):
        metrics = {
            "prefix_cache_backend": self.prefix_cache_backend,
            "prefix_cache_lookups": self.cache_lookups,
            "prefix_cache_queried_blocks": self.cache_queried_blocks,
            "prefix_cache_hit_blocks": self.cache_hit_blocks,
            "prefix_cache_evictions": self.cache_evictions,
            "prefix_cache_block_hit_rate": (
                self.cache_hit_blocks / self.cache_queried_blocks
                if self.cache_queried_blocks
                else 0.0
            ),
        }
        if self.radix_cache is not None:
            metrics.update(self.radix_cache.stats())
        return metrics
