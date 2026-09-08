import json

from fastapi.testclient import TestClient

from nanovllm.serve.app import create_app
from nanovllm.serve.mock_engine import MockEngine


def make_client(api_key=None):
    app = create_app(
        lambda: MockEngine(step_delay_s=0),
        served_model="nano-test",
        backend_name="test backend",
        api_key=api_key,
    )
    return TestClient(app)


def request_body(**overrides):
    body = {
        "model": "nano-test",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 12,
        "temperature": 0.8,
    }
    body.update(overrides)
    return body


def sse_payloads(response):
    return [
        line.removeprefix("data: ")
        for line in response.iter_lines()
        if line.startswith("data: ")
    ]


def test_health_models_and_frontend_are_served():
    with make_client() as client:
        health = client.get("/health")
        models = client.get("/v1/models")
        frontend = client.get("/")

    assert health.json() == {
        "status": "ok",
        "model": "nano-test",
        "backend": "test backend",
    }
    assert models.json()["data"][0]["id"] == "nano-test"
    assert "nano-vLLM Console" in frontend.text


def test_non_streaming_chat_completion_matches_openai_shape():
    with make_client() as client:
        response = client.post("/v1/chat/completions", json=request_body())

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"].startswith("chatcmpl-nv-")
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "nano-test"
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert payload["choices"][0]["message"]["content"]
    assert payload["usage"]["completion_tokens"] == 12
    assert payload["usage"]["total_tokens"] == (payload["usage"]["prompt_tokens"] + 12)
    assert payload["x_nanovllm_metrics"]["ttft_ms"] >= 0


def test_non_streaming_wait_survives_periodic_disconnect_check_timeout():
    app = create_app(lambda: MockEngine(step_delay_s=0.02),
                     served_model="nano-test", backend_name="test backend")
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=request_body(max_tokens=20))
    assert response.status_code == 200
    assert response.json()["usage"]["completion_tokens"] == 20


def test_streaming_chat_completion_emits_role_text_finish_usage_and_done():
    with make_client() as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json=request_body(
                stream=True,
                stream_options={"include_usage": True},
            ),
        ) as response:
            payloads = sse_payloads(response)

    assert response.status_code == 200
    assert payloads[-1] == "[DONE]"
    chunks = [json.loads(item) for item in payloads[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    text = "".join(
        chunk["choices"][0]["delta"].get("content", "")
        for chunk in chunks
        if chunk["choices"]
    )
    assert text
    finish_chunks = [
        chunk
        for chunk in chunks
        if chunk["choices"] and chunk["choices"][0]["finish_reason"] is not None
    ]
    assert finish_chunks[0]["choices"][0]["finish_reason"] == "length"
    usage_chunk = chunks[-1]
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"]["completion_tokens"] == 12


def test_api_key_and_validation_errors_use_openai_error_envelope():
    with make_client(api_key="secret") as client:
        unauthorized = client.get("/v1/models")
        invalid = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer secret"},
            json=request_body(n=2),
        )

    assert unauthorized.status_code == 401
    assert unauthorized.json()["error"]["type"] == "authentication_error"
    assert invalid.status_code == 422
    assert invalid.json()["error"]["param"] is None


def test_unsupported_sampling_options_are_not_silently_ignored():
    with make_client() as client:
        response = client.post(
            "/v1/chat/completions",
            json=request_body(top_p=0.8),
        )

    assert response.status_code == 422
    assert "top_p=1" in response.json()["error"]["message"]


def test_unknown_model_returns_model_not_found_error():
    with make_client() as client:
        response = client.post(
            "/v1/chat/completions",
            json=request_body(model="missing"),
        )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_output_limit_cannot_exceed_the_served_context_window():
    with make_client() as client:
        response = client.post(
            "/v1/chat/completions",
            json=request_body(max_tokens=65536),
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_length_exceeded"
    assert response.json()["error"]["param"] == "max_tokens"
