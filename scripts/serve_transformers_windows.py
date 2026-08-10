"""Run the nano-vLLM web console with a Windows-compatible HF backend.

This compatibility server intentionally does not claim nano-vLLM scheduling or
Paged KV behavior. It exists so the same UI and OpenAI-compatible wire protocol
can run a real local model when native Triton/FlashAttention wheels are absent.
"""

import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from time import perf_counter, time
from typing import List, Literal, Optional
from uuid import uuid4

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextStreamer,
)


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "nanovllm" / "serve" / "static"
_STOP = object()


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["developer", "system", "user", "assistant"]
    content: str = Field(min_length=1)


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_usage: bool = False


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    messages: List[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.7, gt=0)
    top_p: float = Field(default=0.8, gt=0, le=1)
    max_tokens: Optional[int] = Field(default=256, ge=1)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    n: int = Field(default=1, ge=1)
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    priority: int = Field(default=0, ge=0)
    request_class: str = Field(default="interactive", min_length=1)
    ttft_slo_ms: Optional[float] = Field(default=None, gt=0)
    tpot_slo_ms: Optional[float] = Field(default=None, gt=0)
    e2e_slo_ms: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_options(self):
        if self.n != 1:
            raise ValueError("the Transformers compatibility backend supports n=1 only")
        return self

    @property
    def output_limit(self):
        return self.max_completion_tokens or self.max_tokens or 256

    @property
    def include_usage(self):
        return bool(self.stream_options and self.stream_options.include_usage)


@dataclass
class GenerationOperation:
    request_id: str
    created: int
    messages: list
    temperature: float
    top_p: float
    max_tokens: int
    priority: int
    request_class: str
    arrival_time: float = field(default_factory=perf_counter)
    events: Queue = field(default_factory=Queue)
    cancelled: Event = field(default_factory=Event)

    def cancel(self):
        self.cancelled.set()


class CancelStoppingCriteria(StoppingCriteria):
    def __init__(self, cancelled):
        self.cancelled = cancelled

    def __call__(self, input_ids, scores, **kwargs):
        return self.cancelled.is_set()


class EventTextStreamer(TextStreamer):
    def __init__(self, tokenizer, operation):
        super().__init__(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        self.operation = operation
        self.first_text_time = None

    def on_finalized_text(self, text, stream_end=False):
        if not text:
            return
        if self.first_text_time is None:
            self.first_text_time = perf_counter()
        self.operation.events.put(("delta", text))


class RuntimeStats:
    def __init__(self, model_name, device_name):
        self.model_name = model_name
        self.device_name = device_name
        self.lock = Lock()
        self.waiting = 0
        self.running = 0
        self.completed = 0
        self.cancelled = 0
        self.failed = 0
        self.last_metrics = {}

    def enqueue(self):
        with self.lock:
            self.waiting += 1

    def start(self):
        with self.lock:
            self.waiting -= 1
            self.running += 1

    def finish(self, metrics, cancelled=False, failed=False):
        with self.lock:
            self.running -= 1
            if cancelled:
                self.cancelled += 1
            elif failed:
                self.failed += 1
            else:
                self.completed += 1
            self.last_metrics = dict(metrics)

    def skip_cancelled(self):
        with self.lock:
            self.waiting -= 1
            self.cancelled += 1

    def snapshot(self):
        with self.lock:
            return {
                "backend": "Transformers CUDA compatibility",
                "served_model": self.model_name,
                "device": self.device_name,
                "policy": "fcfs-transformers",
                "serving_active_requests": self.running,
                "serving_queued_requests": self.waiting,
                "running_requests": self.running,
                "waiting_requests": self.waiting,
                "completed_requests": self.completed,
                "cancelled_requests": self.cancelled,
                "failed_requests": self.failed,
                "prefix_cache_backend": "unavailable",
                "prefix_cache_hit_blocks": 0,
                "prefix_cache_block_hit_rate": 0.0,
                "remote_restore_completed": 0,
                "kv_restored_tokens": 0,
                **self.last_metrics,
            }


class TransformersWorker:
    """Serialize model execution on one persistent CUDA-owning thread."""

    def __init__(self, model_path, served_model, enable_thinking=False, device="cuda"):
        self.model_path = str(model_path)
        self.served_model = served_model
        self.enable_thinking = enable_thinking
        self.device = device
        self.queue = Queue()
        self.ready = Event()
        self.thread = None
        self.startup_error = None
        self.tokenizer = None
        self.model = None
        self.stats = RuntimeStats(served_model, device)

    @property
    def is_alive(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, timeout=300):
        if self.is_alive:
            return
        self.ready.clear()
        self.thread = Thread(target=self._run, name="qwen3-transformers", daemon=True)
        self.thread.start()
        if not self.ready.wait(timeout=timeout):
            raise TimeoutError("Qwen3 model startup timed out")
        if self.startup_error is not None:
            raise RuntimeError("Qwen3 model failed to load") from self.startup_error

    def stop(self, timeout=30):
        if self.thread is None:
            return
        self.queue.put(_STOP)
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            raise TimeoutError("Transformers worker did not stop")
        self.thread = None

    def submit(self, payload):
        operation = GenerationOperation(
            request_id="chatcmpl-qwen3-" + uuid4().hex,
            created=int(time()),
            messages=[message.model_dump() for message in payload.messages],
            temperature=payload.temperature,
            top_p=payload.top_p,
            max_tokens=payload.output_limit,
            priority=payload.priority,
            request_class=payload.request_class,
        )
        self.stats.enqueue()
        self.queue.put(operation)
        return operation

    def metrics(self):
        snapshot = self.stats.snapshot()
        snapshot["worker_alive"] = self.is_alive
        if self.device == "cuda" and torch.cuda.is_available():
            snapshot["gpu_memory_allocated_mib"] = round(
                torch.cuda.memory_allocated() / 1024**2,
                1,
            )
        return snapshot

    @staticmethod
    def _normalize_messages(messages):
        return [
            {
                "role": "system" if message["role"] == "developer" else message["role"],
                "content": message["content"],
            }
            for message in messages
        ]

    def _load(self):
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            dtype=dtype,
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.eval()

    def _generate(self, operation):
        started = perf_counter()
        self.stats.start()
        try:
            streamer = EventTextStreamer(self.tokenizer, operation)
            prompt = self.tokenizer.apply_chat_template(
                self._normalize_messages(operation.messages),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
                return_dict=True,
                return_tensors="pt",
            )
            prompt_tokens = int(prompt["input_ids"].shape[-1])
            model_inputs = {
                key: value.to(self.device)
                for key, value in prompt.items()
                if hasattr(value, "to")
            }
            with torch.inference_mode():
                output = self.model.generate(
                    **model_inputs,
                    streamer=streamer,
                    max_new_tokens=operation.max_tokens,
                    do_sample=True,
                    temperature=operation.temperature,
                    top_p=operation.top_p,
                    stopping_criteria=StoppingCriteriaList(
                        [CancelStoppingCriteria(operation.cancelled)]
                    ),
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            completion_ids = output[0, prompt_tokens:].tolist()
            finished = perf_counter()
            first_text = streamer.first_text_time or finished
            queue_ms = (started - operation.arrival_time) * 1000.0
            ttft_ms = (first_text - operation.arrival_time) * 1000.0
            e2e_ms = (finished - operation.arrival_time) * 1000.0
            tpot_ms = (
                (e2e_ms - ttft_ms) / (len(completion_ids) - 1)
                if len(completion_ids) > 1
                else None
            )
            metrics = {
                "request_class": operation.request_class,
                "priority": operation.priority,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(completion_ids),
                "queue_ms": queue_ms,
                "ttft_ms": ttft_ms,
                "tpot_ms": tpot_ms,
                "e2e_ms": e2e_ms,
                "preemptions": 0,
                "kv_restore_wait_ms": 0.0,
                "kv_restored_tokens": 0,
                "kv_restore_failures": 0,
            }
            if operation.cancelled.is_set():
                self.stats.finish(metrics, cancelled=True)
                operation.events.put(("cancelled", None))
                return
            eos = bool(completion_ids) and completion_ids[-1] == self.tokenizer.eos_token_id
            finish_reason = "stop" if eos else (
                "length" if len(completion_ids) >= operation.max_tokens else "stop"
            )
            text = self.tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            self.stats.finish(metrics)
            operation.events.put((
                "finished",
                {
                    "text": text,
                    "token_ids": completion_ids,
                    "prompt_tokens": prompt_tokens,
                    "finish_reason": finish_reason,
                    "metrics": metrics,
                },
            ))
        except Exception as exc:
            metrics = {"e2e_ms": (perf_counter() - operation.arrival_time) * 1000.0}
            self.stats.finish(metrics, failed=True)
            operation.events.put(("failed", "%s: %s" % (type(exc).__name__, exc)))

    def _run(self):
        try:
            self._load()
        except Exception as exc:
            self.startup_error = exc
            self.ready.set()
            return
        self.ready.set()
        while True:
            operation = self.queue.get()
            if operation is _STOP:
                break
            if operation.cancelled.is_set():
                self.stats.skip_cancelled()
                operation.events.put(("cancelled", None))
                continue
            self._generate(operation)
        del self.model
        self.model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()


def error_payload(message, error_type="invalid_request_error", param=None, code=None):
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }


def sse(payload):
    if payload == "[DONE]":
        return "data: [DONE]\n\n"
    return "data: %s\n\n" % json.dumps(payload, ensure_ascii=False)


def usage(result):
    completion_tokens = len(result["token_ids"])
    return {
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": completion_tokens,
        "total_tokens": result["prompt_tokens"] + completion_tokens,
    }


def create_app(worker, api_key=None):
    @asynccontextmanager
    async def lifespan(app):
        await asyncio.to_thread(worker.start)
        try:
            yield
        finally:
            await asyncio.to_thread(worker.stop)

    app = FastAPI(
        title="Qwen3 Transformers compatibility server",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request, exc):
        first = exc.errors()[0]
        location = first.get("loc", ())
        param = ".".join(str(item) for item in location[1:]) or None
        return JSONResponse(
            error_payload(first.get("msg", "invalid request"), param=param),
            status_code=422,
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(_request, exc):
        payload = exc.detail if isinstance(exc.detail, dict) else error_payload(str(exc.detail))
        return JSONResponse(payload, status_code=exc.status_code, headers=exc.headers)

    @app.middleware("http")
    async def authenticate(request, call_next):
        if (
            api_key
            and request.url.path.startswith("/v1/")
            and request.headers.get("authorization") != "Bearer " + api_key
        ):
            return JSONResponse(
                error_payload(
                    "Incorrect API key provided",
                    error_type="authentication_error",
                    code="invalid_api_key",
                ),
                status_code=401,
            )
        return await call_next(request)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    async def health():
        return {
            "status": "ok" if worker.is_alive else "unavailable",
            "model": worker.served_model,
            "backend": "Transformers CUDA compatibility",
        }

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{
                "id": worker.served_model,
                "object": "model",
                "created": int(time()),
                "owned_by": "Qwen",
            }],
        }

    @app.get("/v1/metrics")
    async def metrics():
        return worker.metrics()

    @app.post("/v1/chat/completions")
    async def chat(payload: ChatRequest, request: Request):
        if payload.model != worker.served_model:
            raise HTTPException(
                404,
                error_payload(
                    "The model '%s' does not exist" % payload.model,
                    param="model",
                    code="model_not_found",
                ),
            )
        operation = worker.submit(payload)

        if payload.stream:
            async def stream_events():
                completed = False
                try:
                    yield sse({
                        "id": operation.request_id,
                        "object": "chat.completion.chunk",
                        "created": operation.created,
                        "model": worker.served_model,
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }],
                    })
                    while True:
                        event_type, value = await asyncio.to_thread(operation.events.get)
                        if event_type == "delta":
                            yield sse({
                                "id": operation.request_id,
                                "object": "chat.completion.chunk",
                                "created": operation.created,
                                "model": worker.served_model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": value},
                                    "finish_reason": None,
                                }],
                            })
                        elif event_type == "finished":
                            completed = True
                            yield sse({
                                "id": operation.request_id,
                                "object": "chat.completion.chunk",
                                "created": operation.created,
                                "model": worker.served_model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": value["finish_reason"],
                                }],
                                "x_nanovllm_metrics": value["metrics"],
                            })
                            if payload.include_usage:
                                yield sse({
                                    "id": operation.request_id,
                                    "object": "chat.completion.chunk",
                                    "created": operation.created,
                                    "model": worker.served_model,
                                    "choices": [],
                                    "usage": usage(value),
                                })
                            yield sse("[DONE]")
                            return
                        elif event_type == "failed":
                            yield sse(error_payload(value, "server_error"))
                            yield sse("[DONE]")
                            return
                        elif event_type == "cancelled":
                            return
                finally:
                    if not completed:
                        operation.cancel()

            return StreamingResponse(
                stream_events(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-Request-Id": operation.request_id,
                },
            )

        completed = False
        try:
            while True:
                if await request.is_disconnected():
                    operation.cancel()
                    raise HTTPException(499, error_payload("Client disconnected"))
                try:
                    event_type, value = await asyncio.to_thread(
                        operation.events.get,
                        True,
                        0.1,
                    )
                except Empty:
                    continue
                if event_type == "finished":
                    completed = True
                    return JSONResponse({
                        "id": operation.request_id,
                        "object": "chat.completion",
                        "created": operation.created,
                        "model": worker.served_model,
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": value["text"]},
                            "finish_reason": value["finish_reason"],
                        }],
                        "usage": usage(value),
                        "x_nanovllm_metrics": value["metrics"],
                    })
                if event_type == "failed":
                    raise HTTPException(500, error_payload(value, "server_error"))
                if event_type == "cancelled":
                    raise HTTPException(499, error_payload("Generation cancelled"))
        finally:
            if not completed:
                operation.cancel()

    return app


def parse_args():
    parser = argparse.ArgumentParser(
        description="Serve a real local Qwen3 model on Windows with Transformers",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default="Qwen3-0.6B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--api-key")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model).resolve()
    if not (model_path / "model.safetensors").is_file():
        raise SystemExit("model.safetensors was not found under %s" % model_path)
    worker = TransformersWorker(
        model_path,
        args.served_model_name,
        enable_thinking=args.enable_thinking,
        device=args.device,
    )
    app = create_app(worker, api_key=args.api_key)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
