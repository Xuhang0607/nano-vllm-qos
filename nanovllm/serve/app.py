import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from time import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from nanovllm.engine.qos import RequestQoS
from nanovllm.sampling_params import SamplingParams
from nanovllm.serve.engine_worker import (
    GenerationCancelled,
    GenerationFailed,
    GenerationFinished,
    InferenceWorker,
    TextDelta,
)
from nanovllm.serve.protocol import ChatCompletionRequest

STATIC_DIR = Path(__file__).with_name("static")


def _error(message, error_type="invalid_request_error", param=None, code=None):
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }


def _sse(payload):
    if payload == "[DONE]":
        return "data: [DONE]\n\n"
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _usage(event):
    completion_tokens = len(event.token_ids)
    return {
        "prompt_tokens": event.prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": event.prompt_tokens + completion_tokens,
    }


def create_app(
    engine_factory,
    served_model: str,
    backend_name="nano-vllm",
    api_key: str | None = None,
):
    worker = InferenceWorker(engine_factory, served_model, backend_name)

    @asynccontextmanager
    async def lifespan(app):
        await asyncio.to_thread(worker.start)
        try:
            yield
        finally:
            await asyncio.to_thread(worker.stop)

    app = FastAPI(
        title="nano-vLLM OpenAI-compatible server",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.state.inference_worker = worker

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request, exc):
        first = exc.errors()[0]
        location = first.get("loc", ())
        param = ".".join(str(item) for item in location[1:]) or None
        return JSONResponse(
            _error(first.get("msg", "invalid request"), param=param),
            status_code=422,
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(_request, exc):
        detail = exc.detail
        if isinstance(detail, dict) and "error" in detail:
            payload = detail
        else:
            payload = _error(str(detail))
        return JSONResponse(payload, status_code=exc.status_code, headers=exc.headers)

    @app.middleware("http")
    async def authenticate(request, call_next):
        if (
            api_key
            and request.url.path.startswith("/v1/")
            and request.headers.get("authorization") != f"Bearer {api_key}"
        ):
            return JSONResponse(
                _error(
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
            "model": served_model,
            "backend": backend_name,
        }

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": served_model,
                    "object": "model",
                    "created": int(time()),
                    "owned_by": "nano-vllm",
                }
            ],
        }

    @app.get("/v1/metrics")
    async def metrics():
        return worker.metrics()

    def submit(payload):
        if payload.model != served_model:
            raise HTTPException(
                404,
                _error(
                    f"The model '{payload.model}' does not exist",
                    param="model",
                    code="model_not_found",
                ),
            )
        max_model_len = worker.metrics().get("max_model_len")
        if max_model_len is not None and payload.output_limit > max_model_len:
            raise HTTPException(
                400,
                _error(
                    f"max_tokens must not exceed max_model_len ({max_model_len})",
                    param="max_tokens",
                    code="context_length_exceeded",
                ),
            )
        return worker.submit(
            [message.model_dump() for message in payload.messages],
            SamplingParams(
                temperature=payload.temperature,
                max_tokens=payload.output_limit,
            ),
            RequestQoS(
                priority=payload.priority,
                ttft_slo_ms=payload.ttft_slo_ms,
                tpot_slo_ms=payload.tpot_slo_ms,
                e2e_slo_ms=payload.e2e_slo_ms,
                request_class=payload.request_class,
            ),
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatCompletionRequest, request: Request):
        handle = submit(payload)
        if payload.stream:

            async def stream_events():
                completed = False
                try:
                    yield _sse(
                        {
                            "id": handle.request_id,
                            "object": "chat.completion.chunk",
                            "created": handle.created,
                            "model": served_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": ""},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                    while True:
                        event = await asyncio.to_thread(handle.events.get)
                        if isinstance(event, TextDelta):
                            yield _sse(
                                {
                                    "id": handle.request_id,
                                    "object": "chat.completion.chunk",
                                    "created": handle.created,
                                    "model": served_model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": event.text},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            )
                        elif isinstance(event, GenerationFinished):
                            completed = True
                            yield _sse(
                                {
                                    "id": handle.request_id,
                                    "object": "chat.completion.chunk",
                                    "created": handle.created,
                                    "model": served_model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {},
                                            "finish_reason": event.finish_reason,
                                        }
                                    ],
                                    "x_nanovllm_metrics": event.metrics,
                                }
                            )
                            if payload.include_usage:
                                yield _sse(
                                    {
                                        "id": handle.request_id,
                                        "object": "chat.completion.chunk",
                                        "created": handle.created,
                                        "model": served_model,
                                        "choices": [],
                                        "usage": _usage(event),
                                    }
                                )
                            yield _sse("[DONE]")
                            return
                        elif isinstance(event, GenerationFailed):
                            yield _sse(_error(event.message, "server_error"))
                            yield _sse("[DONE]")
                            return
                        elif isinstance(event, GenerationCancelled):
                            return
                finally:
                    if not completed:
                        handle.cancel()

            return StreamingResponse(
                stream_events(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-Request-Id": handle.request_id,
                },
            )

        completed = False
        try:
            while True:
                if await request.is_disconnected():
                    handle.cancel()
                    raise HTTPException(499, _error("Client disconnected"))
                event = await asyncio.to_thread(handle.events.get)
                if isinstance(event, GenerationFinished):
                    completed = True
                    return JSONResponse(
                        {
                            "id": handle.request_id,
                            "object": "chat.completion",
                            "created": handle.created,
                            "model": served_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": event.text,
                                    },
                                    "finish_reason": event.finish_reason,
                                }
                            ],
                            "usage": _usage(event),
                            "x_nanovllm_metrics": event.metrics,
                        },
                        headers={"X-Request-Id": handle.request_id},
                    )
                if isinstance(event, GenerationFailed):
                    raise HTTPException(500, _error(event.message, "server_error"))
                if isinstance(event, GenerationCancelled):
                    raise HTTPException(499, _error("Generation cancelled"))
        finally:
            if not completed:
                handle.cancel()

    return app
