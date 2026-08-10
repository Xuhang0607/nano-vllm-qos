from nanovllm.engine.qos import RequestQoS
from nanovllm.sampling_params import SamplingParams
from nanovllm.serve.engine_worker import (
    GenerationCancelled,
    GenerationFailed,
    GenerationFinished,
    InferenceWorker,
    TextDelta,
)
from nanovllm.serve.mock_engine import MockEngine


def collect(handle, timeout=3):
    text = ""
    while True:
        event = handle.events.get(timeout=timeout)
        if isinstance(event, TextDelta):
            text += event.text
        if isinstance(event, (GenerationFinished, GenerationCancelled)):
            return text, event


def test_worker_streams_two_requests_through_one_engine_loop():
    engine = MockEngine(step_delay_s=0)
    worker = InferenceWorker(lambda: engine, "nano-test", "test")
    worker.start()
    try:
        first = worker.submit(
            [{"role": "user", "content": "first"}],
            SamplingParams(max_tokens=10),
            RequestQoS(priority=0),
        )
        second = worker.submit(
            [{"role": "user", "content": "second"}],
            SamplingParams(max_tokens=10),
            RequestQoS(priority=5),
        )
        first_text, first_result = collect(first)
        second_text, second_result = collect(second)
    finally:
        worker.stop()

    assert isinstance(first_result, GenerationFinished)
    assert isinstance(second_result, GenerationFinished)
    assert first_text == first_result.text
    assert second_text == second_result.text
    assert worker.metrics()["worker_alive"] is False


def test_worker_cancels_active_request():
    engine = MockEngine(step_delay_s=0.01)
    worker = InferenceWorker(lambda: engine, "nano-test", "test")
    worker.start()
    try:
        handle = worker.submit(
            [{"role": "user", "content": "cancel me"}],
            SamplingParams(max_tokens=128),
            RequestQoS(),
        )
        first = handle.events.get(timeout=2)
        assert isinstance(first, TextDelta)
        handle.cancel()
        _, result = collect(handle)
    finally:
        worker.stop()

    assert isinstance(result, GenerationCancelled)
    assert engine.get_scheduler_metrics()["cancelled_requests"] == 1


def test_worker_rejects_prompt_and_output_beyond_context_window():
    engine = MockEngine(step_delay_s=0)
    engine.config = type("Config", (), {"max_model_len": 32, "eos": -1})()
    worker = InferenceWorker(lambda: engine, "nano-test", "test")
    worker.start()
    try:
        handle = worker.submit(
            [{"role": "user", "content": "this prompt is already too long"}],
            SamplingParams(max_tokens=16),
            RequestQoS(),
        )
        result = handle.events.get(timeout=2)
    finally:
        worker.stop()

    assert isinstance(result, GenerationFailed)
    assert "context length exceeded" in result.message
