"""Short-horizon soft page reservation, not a guarantee against future eviction."""


def growth_pages(cached_tokens, owned_pages, future_tokens, block_size):
    return max(0, (cached_tokens + future_tokens + block_size - 1) // block_size - owned_pages)


def priority_reserve_pages(candidate, running, horizon, block_size):
    reserve = 0
    for seq in running:
        if seq.qos.priority < candidate.qos.priority:
            continue
        # A just-selected prefill has not executed yet or emitted its first token.
        pending_prefill = seq.num_scheduled_tokens > 0
        remaining = max(0, seq.max_tokens - seq.num_completion_tokens - int(pending_prefill))
        reserve += growth_pages(
            seq.num_physical_cached_tokens + seq.num_scheduled_tokens,
            len(seq.block_table), min(horizon, remaining), block_size,
        )
    remaining = max(0, candidate.max_tokens - candidate.num_completion_tokens - 1)
    reserve += growth_pages(candidate.num_tokens, candidate.num_blocks,
                            min(horizon, remaining), block_size)
    return reserve
