import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread

from nanovllm.engine.qos import RequestQoS
from nanovllm.sampling_params import SamplingParams
from nanovllm.serve.engine_worker import (
    GenerationCancelled,
    GenerationFailed,
    GenerationFinished,
    GenerationEventQueue,
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


def test_async_event_delivery_works_with_executor_exhausted():
    async def exercise():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        release = Event()
        blocked = loop.run_in_executor(None, release.wait)
        queues = [GenerationEventQueue() for _ in range(128)]
        pending = [asyncio.create_task(queue.get_async()) for queue in queues]
        await asyncio.sleep(0)
        producer = Thread(target=queues[-1].put, args=(TextDelta("ready"),))
        try:
            producer.start()
            event = await asyncio.wait_for(pending[-1], timeout=1)
            assert event.text == "ready"
            assert not blocked.done()
        finally:
            producer.join()
            release.set()
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await blocked
        assert all(not queue._waiters for queue in queues)

    asyncio.run(exercise())


def test_cancelled_async_wait_does_not_consume_later_events():
    async def exercise():
        queue = GenerationEventQueue()
        task = asyncio.create_task(queue.get_async())
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        queue.put(TextDelta("first"))
        queue.put(TextDelta("second"))
        assert (await queue.get_async()).text == "first"
        assert (await queue.get_async()).text == "second"
        assert not queue._waiters

    asyncio.run(exercise())


def test_non_streaming_worker_decodes_only_once(monkeypatch):
    engine = MockEngine(step_delay_s=0)
    decode = engine.tokenizer.decode
    calls = []

    def counted_decode(tokens, **kwargs):
        calls.append(tuple(tokens))
        return decode(tokens, **kwargs)

    monkeypatch.setattr(engine.tokenizer, "decode", counted_decode)
    worker = InferenceWorker(lambda: engine, "nano-test", "test")
    worker.start()
    try:
        handle = worker.submit([{"role": "user", "content": "test"}],
                               SamplingParams(max_tokens=32), RequestQoS(), stream=False)
        event = handle.events.get(timeout=3)
        assert isinstance(event, GenerationFinished)
        assert len(event.token_ids) == 32
        assert len(calls) == 1
        assert event.text == decode(event.token_ids)
    finally:
        worker.stop()
