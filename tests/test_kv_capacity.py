import pytest

pytest.importorskip("torch", reason="ModelRunner import requires the GPU dependency stack")
pytest.importorskip("flash_attn")
pytest.importorskip("triton")

from nanovllm.engine.model_runner import resolve_kv_cache_blocks


def test_kv_cache_capacity_uses_all_available_blocks_by_default():
    assert resolve_kv_cache_blocks(128, None) == 128


def test_kv_cache_capacity_honors_reproducible_pressure_limit():
    assert resolve_kv_cache_blocks(128, 24) == 24


def test_kv_cache_capacity_rejects_invalid_limits():
    with pytest.raises(ValueError, match="must be positive"):
        resolve_kv_cache_blocks(128, 0)
    with pytest.raises(ValueError, match="exceeds available"):
        resolve_kv_cache_blocks(16, 24)
    with pytest.raises(RuntimeError, match="no GPU memory"):
        resolve_kv_cache_blocks(0, None)
