from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="ModelRunner import requires the GPU dependency stack")
pytest.importorskip("flash_attn")
pytest.importorskip("triton")

from torch.overrides import _get_current_function_mode_stack

from nanovllm.engine import model_runner


def prepare_runner(monkeypatch, fail=False):
    config = SimpleNamespace(kvcache_block_size=256, enforce_eager=True,
                             tensor_parallel_size=1, model="unused",
                             hf_config=SimpleNamespace(dtype=torch.float64))
    monkeypatch.setattr(model_runner.dist, "init_process_group", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a: None)

    def model_factory(config):
        assert torch.get_default_dtype() == torch.float64
        assert _get_current_function_mode_stack()[-1].device.type == "cuda"
        if fail:
            raise RuntimeError("initialization failure")
        return object()

    monkeypatch.setattr(model_runner, "Qwen3ForCausalLM", model_factory)
    monkeypatch.setattr(model_runner, "load_model", lambda *a: None)
    monkeypatch.setattr(model_runner.ModelRunner, "warmup_model", lambda *a: None)
    monkeypatch.setattr(model_runner.ModelRunner, "allocate_kv_cache", lambda *a: None)
    return config


@pytest.mark.parametrize("fail", [False, True])
def test_initialization_restores_dtype_and_device_stack(monkeypatch, fail):
    config = prepare_runner(monkeypatch, fail)
    dtype = torch.get_default_dtype()
    # An existing caller context must survive both success and failure.
    with torch.device("meta"):
        before = _get_current_function_mode_stack()
        try:
            if fail:
                with pytest.raises(RuntimeError, match="initialization failure"):
                    model_runner.ModelRunner(config, 0, [])
            else:
                model_runner.ModelRunner(config, 0, [])
            assert torch.get_default_dtype() == dtype
            assert _get_current_function_mode_stack() == before
        finally:
            torch.set_default_dtype(dtype)


def test_initialization_does_not_leave_cpu_dispatch_mode(monkeypatch):
    config = prepare_runner(monkeypatch)
    before = _get_current_function_mode_stack()
    model_runner.ModelRunner(config, 0, [])
    assert _get_current_function_mode_stack() == before


def test_graph_capture_failure_restores_defaults(monkeypatch):
    config = prepare_runner(monkeypatch)
    config.enforce_eager = False
    before = _get_current_function_mode_stack()
    dtype = torch.get_default_dtype()

    def fail_capture(self):
        assert _get_current_function_mode_stack()[-1].device.type == "cuda"
        raise RuntimeError("graph capture failure")

    monkeypatch.setattr(model_runner.ModelRunner, "capture_cudagraph", fail_capture)
    with pytest.raises(RuntimeError, match="graph capture failure"):
        model_runner.ModelRunner(config, 0, [])
    assert _get_current_function_mode_stack() == before
    assert torch.get_default_dtype() == dtype


def test_input_preparation_explicitly_uses_pinned_cpu_memory(monkeypatch):
    from nanovllm.engine.sequence import Sequence
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.utils.context import reset_context

    allocations = []

    class FakeTensor:
        def cuda(self, non_blocking):
            assert non_blocking
            return self

    def factory(data, **kwargs):
        allocations.append(kwargs)
        assert kwargs["device"] == "cpu"
        assert kwargs["pin_memory"]
        return FakeTensor()

    runner = model_runner.ModelRunner.__new__(model_runner.ModelRunner)
    runner.block_size = Sequence.block_size
    seq = Sequence([1, 2], SamplingParams())
    seq.block_table = [0]
    seq.num_scheduled_tokens = 2
    monkeypatch.setattr(torch, "tensor", factory)
    try:
        runner.prepare_prefill([seq])
        runner.prepare_decode([seq])
        runner.prepare_sample([seq])
    finally:
        reset_context()
    assert len(allocations) == 11
