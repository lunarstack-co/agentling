"""The native Ollama reference adapter in examples/providers, run offline."""

import json
import re
from typing import Any

import httpx
import pytest

from agentling import (
    ChatMessage,
    Delta,
    ModelAuthError,
    ModelBadRequestError,
    ModelConnectionError,
    ModelError,
    ModelRateLimitError,
    ModelServerError,
    ModelTimeoutError,
    ToolCall,
    ToolDefinition,
    Usage,
    agglomerate_deltas,
)
from examples.providers.ollama_native import OllamaModel

_USER = [ChatMessage(role="user", content="hi")]


def ndjson_transport(
    chunks: list[dict[str, Any]],
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
    body: str | None = None,
    requests: list[httpx.Request] | None = None,
) -> httpx.MockTransport:
    """Answer every request with these chunks, one JSON object per line."""

    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        text = (
            body if body is not None else "".join(json.dumps(c) + "\n" for c in chunks)
        )
        return httpx.Response(status, headers=headers, content=text.encode())

    return httpx.MockTransport(handler)


def _model(transport: httpx.AsyncBaseTransport, **kwargs: Any) -> OllamaModel:
    return OllamaModel("qwen3:8b", transport=transport, **kwargs)


async def _collect(
    model: OllamaModel, messages: list[ChatMessage] | None = None
) -> list[Delta]:
    return [d async for d in model.stream(messages or _USER)]


def _chunk(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "qwen3:8b",
        "message": {"role": "assistant", **message},
        "done": False,
    }


def _done(
    message: dict[str, Any] | None = None, reason: str = "stop", **counts: int
) -> dict[str, Any]:
    return {
        "model": "qwen3:8b",
        "message": {"role": "assistant", "content": "", **(message or {})},
        "done": True,
        "done_reason": reason,
        **({"prompt_eval_count": 12, "eval_count": 3} if not counts else counts),
    }


async def test_text_stream() -> None:
    chunks = [_chunk({"content": "Hel"}), _chunk({"content": "lo"}), _done()]

    deltas = await _collect(_model(ndjson_transport(chunks)))

    message = agglomerate_deltas(deltas)
    assert message.content == "Hello"
    assert message.finish_reason == "stop"
    assert message.usage == Usage(12, 3)
    assert all(d.content or d.usage or d.finish_reason for d in deltas)


async def test_tool_call_on_the_done_chunk() -> None:
    call = {"function": {"name": "echo", "arguments": {"text": "ping"}}}
    chunks = [_done({"tool_calls": [call]})]

    message = agglomerate_deltas(await _collect(_model(ndjson_transport(chunks))))

    (tool_call,) = message.tool_calls
    assert (tool_call.name, tool_call.arguments) == ("echo", {"text": "ping"})
    assert re.fullmatch(r"agentling_call_0_[0-9a-f]{8}", tool_call.id)
    assert message.finish_reason == "tool_calls"


async def test_tool_call_ids_and_indexes_are_kept() -> None:
    def call(call_id: str, index: int) -> dict[str, Any]:
        return {
            "id": call_id,
            "function": {"index": index, "name": "echo", "arguments": {}},
        }

    chunks = [
        _chunk({"content": "", "tool_calls": [call("call_b", 1)]}),
        _chunk({"content": "", "tool_calls": [call("call_a", 0)]}),
        _done(),
    ]

    message = agglomerate_deltas(await _collect(_model(ndjson_transport(chunks))))

    assert [c.id for c in message.tool_calls] == ["call_a", "call_b"]


async def test_thinking_is_streamed_as_reasoning() -> None:
    chunks = [
        _chunk({"thinking": "hm"}),
        _chunk({"thinking": "m"}),
        _chunk({"content": "ok"}),
        _done(),
    ]

    message = agglomerate_deltas(await _collect(_model(ndjson_transport(chunks))))

    assert message.reasoning == "hmm"
    assert message.content == "ok"


async def test_missing_counts_default_to_zero() -> None:
    chunks = [_chunk({"content": "x"}), _done(eval_count=4)]

    message = agglomerate_deltas(await _collect(_model(ndjson_transport(chunks))))

    assert message.usage == Usage(0, 4)


async def test_request_body() -> None:
    requests: list[httpx.Request] = []
    model = _model(
        ndjson_transport([_done()], requests=requests),
        api_key="k",
        extra_body={"options": {"temperature": 0}},
    )
    history = [
        ChatMessage("system", "sys"),
        ChatMessage("user", "hi"),
        ChatMessage(
            "assistant",
            reasoning="r",
            tool_calls=[ToolCall("c1", "echo", {"text": "ping"})],
        ),
        ChatMessage("tool", "ping", tool_call_id="c1"),
    ]
    definition = ToolDefinition("echo", "Echo.", {"type": "object"})

    [_ async for _ in model.stream(history, [definition])]

    (request,) = requests
    assert request.url.path == "/api/chat"
    assert request.headers["authorization"] == "Bearer k"
    body = json.loads(request.content)
    assert body["stream"] is True
    assert body["options"] == {"temperature": 0}
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["messages"][2] == {
        "role": "assistant",
        "content": "",
        "thinking": "r",
        "tool_calls": [{"function": {"name": "echo", "arguments": {"text": "ping"}}}],
    }
    assert body["messages"][3] == {
        "role": "tool",
        "content": "ping",
        "tool_name": "echo",
    }
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo.",
                "parameters": {"type": "object"},
            },
        }
    ]


async def test_no_authorization_header_without_a_key() -> None:
    requests: list[httpx.Request] = []

    await _collect(_model(ndjson_transport([_done()], requests=requests)))

    assert "authorization" not in requests[0].headers


@pytest.mark.parametrize(
    ("done_reason", "expected"),
    [("stop", "stop"), ("length", "length"), ("load", "unknown")],
)
async def test_done_reasons_are_mapped(done_reason: str, expected: str) -> None:
    chunks = [_chunk({"content": "x"}), _done(reason=done_reason)]

    message = agglomerate_deltas(await _collect(_model(ndjson_transport(chunks))))

    assert message.finish_reason == expected


def _error(status: int, text: str, **headers: str) -> httpx.MockTransport:
    return ndjson_transport(
        [], status=status, headers=headers, body=json.dumps({"error": text})
    )


def _connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused", request=request)


def _read_timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("slow", request=request)


@pytest.mark.parametrize(
    ("transport", "error_type", "status", "retry_after"),
    [
        (_error(404, "model 'x' not found"), ModelBadRequestError, 404, None),
        (_error(400, "missing request body"), ModelBadRequestError, 400, None),
        (_error(401, "unauthorized"), ModelAuthError, 401, None),
        (_error(403, "forbidden"), ModelAuthError, 403, None),
        (_error(429, "slow", **{"retry-after": "3"}), ModelRateLimitError, 429, 3.0),
        (_error(500, "boom"), ModelServerError, 500, None),
        (_error(503, "busy"), ModelServerError, 503, None),
        (httpx.MockTransport(_connect_error), ModelConnectionError, None, None),
        (httpx.MockTransport(_read_timeout), ModelTimeoutError, None, None),
    ],
)
async def test_failures_map_to_model_errors(
    transport: httpx.AsyncBaseTransport,
    error_type: type[ModelError],
    status: int | None,
    retry_after: float | None,
) -> None:
    with pytest.raises(error_type) as caught:
        await _collect(_model(transport))

    assert type(caught.value) is error_type
    assert caught.value.status == status
    assert caught.value.retry_after == retry_after


async def test_missing_model_message_is_kept() -> None:
    with pytest.raises(ModelBadRequestError, match="model 'x' not found"):
        await _collect(_model(_error(404, "model 'x' not found")))


async def test_mid_stream_error_line() -> None:
    chunks = [_chunk({"content": "Partial"}), {"error": "boom"}]

    with pytest.raises(ModelServerError, match="boom"):
        await _collect(_model(ndjson_transport(chunks)))


async def test_stream_without_a_done_chunk_raises() -> None:
    with pytest.raises(ModelServerError, match="before Ollama sent its final chunk"):
        await _collect(_model(ndjson_transport([_chunk({"content": "x"})])))


@pytest.mark.parametrize(
    "line",
    [
        '{"message": "hi", "done": true}',
        '{"message": {"tool_calls": ["x"]}, "done": true, "done_reason": "stop"}',
    ],
    ids=["message_not_an_object", "tool_call_not_an_object"],
)
async def test_malformed_chunks_raise_model_errors(line: str) -> None:
    transport = ndjson_transport([], body=line + "\n")

    with pytest.raises(ModelServerError, match="Malformed stream chunk") as caught:
        await _collect(_model(transport))

    assert caught.value.retryable is False
