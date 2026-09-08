from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Event, Lock, Thread
from time import time
from uuid import uuid4

from nanovllm.engine.qos import RequestQoS
from nanovllm.sampling_params import SamplingParams

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class TextDelta:
    text: str


@dataclass(slots=True, frozen=True)
class GenerationFinished:
    text: str
    token_ids: tuple[int, ...]
    prompt_tokens: int
    finish_reason: str
    metrics: dict


@dataclass(slots=True, frozen=True)
class GenerationFailed:
    message: str


@dataclass(slots=True, frozen=True)
class GenerationCancelled:
    pass


GenerationEvent = (
    TextDelta | GenerationFinished | GenerationFailed | GenerationCancelled
)


class GenerationEventQueue(Queue[GenerationEvent]):
    """Thread-safe delivery with cancellable async waits and no per-request thread."""

    def __init__(self):
        super().__init__()
        self._waiter_lock = Lock()
        self._waiters = set()

    @staticmethod
    def _wake(future):
        if not future.done():
            future.set_result(None)

    def put(self, item, block=True, timeout=None):
        super().put(item, block=block, timeout=timeout)
        with self._waiter_lock:
            waiters = tuple(self._waiters)
        for loop, future in waiters:
            try:
                loop.call_soon_threadsafe(self._wake, future)
            except RuntimeError:  # The client event loop may have closed during shutdown.
                with self._waiter_lock:
                    self._waiters.discard((loop, future))

    async def get_async(self):
        loop = asyncio.get_running_loop()
        while True:
            future = loop.create_future()
            waiter = (loop, future)
            with self._waiter_lock:
                # Register before checking the queue so a concurrent put cannot lose a wakeup.
                self._waiters.add(waiter)
            try:
                try:
                    return self.get_nowait()
                except Empty:
                    await future
            finally:
                with self._waiter_lock:
                    self._waiters.discard(waiter)


@dataclass(slots=True)
class GenerationHandle:
    request_id: str
    created: int
    events: GenerationEventQueue = field(default_factory=GenerationEventQueue)
    cancelled: Event = field(default_factory=Event)
    stream: bool = True

    def cancel(self):
        self.cancelled.set()


@dataclass(slots=True)
class _QueuedRequest:
    handle: GenerationHandle
    messages: tuple[dict[str, str], ...]
    sampling_params: SamplingParams
    qos: RequestQoS


@dataclass(slots=True)
class _ActiveRequest:
    queued: _QueuedRequest
    seq_id: int
    prompt_tokens: int
    token_ids: list[int] = field(default_factory=list)
    emitted_text: str = ""


_STOP = object()


class InferenceWorker:
    """Own one engine thread so CUDA calls never leak into HTTP worker threads."""

    def __init__(
        self,
        engine_factory: Callable[[], object],
        served_model: str,
        backend_name: str = "nano-vllm",
    ):
        self.engine_factory = engine_factory
        self.served_model = served_model
        self.backend_name = backend_name
        self._incoming = Queue()
        self._ready = Event()
        self._metrics_lock = Lock()
        self._metrics = {
            "backend": backend_name,
            "served_model": served_model,
            "serving_active_requests": 0,
            "serving_queued_requests": 0,
        }
        self._thread = None
        self._startup_error = None

    @property
    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, timeout=600):
        if self.is_alive:
            return
        self._ready.clear()
        self._startup_error = None
        self._thread = Thread(
            target=self._run,
            name="nanovllm-inference",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=timeout):
            raise TimeoutError("nano-vLLM engine startup timed out")
        if self._startup_error is not None:
            raise RuntimeError(
                "nano-vLLM engine failed to start"
            ) from self._startup_error

    def stop(self, timeout=30):
        if self._thread is None:
            return
        self._incoming.put(_STOP)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError("nano-vLLM inference worker did not stop")
        self._thread = None

    def submit(
        self,
        messages: list[dict[str, str]],
        sampling_params: SamplingParams,
        qos: RequestQoS,
        stream: bool = True,
    ) -> GenerationHandle:
        if not self.is_alive:
            raise RuntimeError("nano-vLLM inference worker is not running")
        handle = GenerationHandle(
            request_id=f"chatcmpl-nv-{uuid4().hex}",
            created=int(time()),
            stream=stream,
        )
        self._incoming.put(
            _QueuedRequest(
                handle=handle,
                messages=tuple(messages),
                sampling_params=sampling_params,
                qos=qos,
            )
        )
        return handle

    def metrics(self):
        with self._metrics_lock:
            result = self._metrics.copy()
        result["serving_queued_requests"] = self._incoming.qsize()
        result["worker_alive"] = self.is_alive
        return result

    def _set_metrics(self, metrics, active_count):
        snapshot = dict(metrics)
        snapshot.update(
            {
                "backend": self.backend_name,
                "served_model": self.served_model,
                "serving_active_requests": active_count,
            }
        )
        with self._metrics_lock:
            self._metrics = snapshot

    @staticmethod
    def _normalize_messages(messages):
        return [
            {
                "role": "system" if message["role"] == "developer" else message["role"],
                "content": message["content"],
            }
            for message in messages
        ]

    @staticmethod
    def _prompt_token_ids(tokenizer, messages):
        token_ids = tokenizer.apply_chat_template(
            InferenceWorker._normalize_messages(messages),
            tokenize=True,
            add_generation_prompt=True,
        )
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        token_ids = list(token_ids)
        if not token_ids:
            raise ValueError("chat template produced an empty prompt")
        return token_ids

    def _admit(self, engine, queued, active):
        if queued.handle.cancelled.is_set():
            queued.handle.events.put(GenerationCancelled())
            return
        try:
            prompt_ids = self._prompt_token_ids(engine.tokenizer, queued.messages)
            max_model_len = getattr(getattr(engine, "config", None), "max_model_len", None)
            requested_tokens = queued.sampling_params.max_tokens
            if (
                max_model_len is not None
                and len(prompt_ids) + requested_tokens > max_model_len
            ):
                available_tokens = max(0, max_model_len - len(prompt_ids))
                raise ValueError(
                    "context length exceeded: "
                    f"prompt has {len(prompt_ids)} tokens and max_tokens requests "
                    f"{requested_tokens}, but max_model_len is {max_model_len}; "
                    f"at most {available_tokens} completion tokens are available"
                )
            seq_id = engine.add_request(
                prompt_ids,
                queued.sampling_params,
                queued.qos,
            )
            active[seq_id] = _ActiveRequest(
                queued=queued,
                seq_id=seq_id,
                prompt_tokens=len(prompt_ids),
            )
        except Exception as exc:  # noqa: BLE001 - isolate malformed requests
            queued.handle.events.put(GenerationFailed(f"{type(exc).__name__}: {exc}"))

    def _drain_incoming(self, engine, active, block):
        try:
            item = self._incoming.get(block=block, timeout=0.1 if block else None)
        except Empty:
            return False
        if item is _STOP:
            return True
        self._admit(engine, item, active)
        while True:
            try:
                item = self._incoming.get_nowait()
            except Empty:
                return False
            if item is _STOP:
                return True
            self._admit(engine, item, active)

    @staticmethod
    def _decode_stable_delta(tokenizer, state):
        decoded = tokenizer.decode(state.token_ids, skip_special_tokens=True)
        stable = decoded.split("\ufffd", 1)[0]
        if not stable.startswith(state.emitted_text):
            return ""
        delta = stable[len(state.emitted_text) :]
        state.emitted_text = stable
        return delta

    @staticmethod
    def _finish(engine, state, token_ids):
        state.token_ids = list(token_ids)
        text = engine.tokenizer.decode(token_ids, skip_special_tokens=True)
        if state.queued.handle.stream and text.startswith(state.emitted_text):
            tail = text[len(state.emitted_text) :]
            if tail:
                state.queued.handle.events.put(TextDelta(tail))
                state.emitted_text = text
        metrics = engine.get_request_metrics(state.seq_id) or {}
        eos_token_id = getattr(getattr(engine, "config", None), "eos", None)
        stopped_by_eos = bool(token_ids) and token_ids[-1] == eos_token_id
        finish_reason = (
            "stop"
            if stopped_by_eos
            else (
                "length"
                if len(token_ids) >= state.queued.sampling_params.max_tokens
                else "stop"
            )
        )
        state.queued.handle.events.put(
            GenerationFinished(
                text=text,
                token_ids=tuple(token_ids),
                prompt_tokens=state.prompt_tokens,
                finish_reason=finish_reason,
                metrics=metrics,
            )
        )

    @staticmethod
    def _cancel_requests(engine, active):
        for seq_id, state in tuple(active.items()):
            if not state.queued.handle.cancelled.is_set():
                continue
            engine.cancel_request(seq_id)
            state.queued.handle.events.put(GenerationCancelled())
            del active[seq_id]

    def _fail_active(self, engine, active, exc):
        message = f"{type(exc).__name__}: {exc}"
        for seq_id, state in tuple(active.items()):
            try:
                engine.cancel_request(seq_id)
            except Exception:  # noqa: BLE001 - preserve the original engine error
                LOGGER.exception("failed to cancel request %s", seq_id)
            state.queued.handle.events.put(GenerationFailed(message))
        active.clear()

    def _run(self):
        engine = None
        active = {}
        stopping = False
        try:
            engine = self.engine_factory()
        except Exception as exc:  # noqa: BLE001 - forwarded through start()
            self._startup_error = exc
            self._ready.set()
            return
        self._set_metrics(engine.get_scheduler_metrics(), 0)
        self._ready.set()

        try:
            while not stopping:
                stopping = self._drain_incoming(
                    engine,
                    active,
                    block=not active,
                )
                self._cancel_requests(engine, active)
                if stopping:
                    break
                if not active:
                    self._set_metrics(engine.get_scheduler_metrics(), 0)
                    continue
                try:
                    outputs, _ = engine.step()
                    for seq_id, token_id in engine.take_step_token_events():
                        state = active.get(seq_id)
                        if state is None:
                            continue
                        state.token_ids.append(token_id)
                        if state.queued.handle.stream:
                            delta = self._decode_stable_delta(engine.tokenizer, state)
                            if delta:
                                state.queued.handle.events.put(TextDelta(delta))
                    for seq_id, token_ids in outputs:
                        state = active.pop(seq_id, None)
                        if state is not None:
                            self._finish(engine, state, token_ids)
                    self._set_metrics(engine.get_scheduler_metrics(), len(active))
                except Exception as exc:  # noqa: BLE001 - report failures per request
                    LOGGER.exception("nano-vLLM engine step failed")
                    self._fail_active(engine, active, exc)
                    self._set_metrics(engine.get_scheduler_metrics(), 0)
        finally:
            for seq_id, state in tuple(active.items()):
                try:
                    engine.cancel_request(seq_id)
                except Exception:  # noqa: BLE001 - shutdown must still close the engine
                    LOGGER.exception(
                        "failed to cancel request %s during shutdown", seq_id
                    )
                state.queued.handle.events.put(GenerationCancelled())
            if engine is not None:
                engine.exit()
