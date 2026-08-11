import pytest

from nanovllm.config import effective_context_window


def test_effective_context_window_is_bounded_by_model_and_kv_capacity():
    assert effective_context_window(40960, 160, 256) == 40960
    assert effective_context_window(40960, 133, 256) == 34048
    assert effective_context_window(8192, 160, 256) == 8192


@pytest.mark.parametrize("values", [(0, 1, 1), (1, 0, 1), (1, 1, 0)])
def test_effective_context_window_rejects_invalid_dimensions(values):
    with pytest.raises(ValueError, match="must be positive"):
        effective_context_window(*values)
