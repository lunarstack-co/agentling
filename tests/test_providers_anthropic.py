"""The Anthropic Messages reference adapter in examples/providers, run offline."""

import json
from typing import Any

import httpx
import pytest

from agentling import (
    ChatMessage,
    Delta,
    ModelAuthError,
    ModelBadRequestError,
    ModelConnectionError,
    ModelContextLengthError,
    ModelError,
    ModelRateLimitError,
    ModelServerError,
    ModelTimeoutError,
    ToolCall,
    ToolDefinition,
    Usage,
    agglomerate_deltas,
)
from examples.providers.anthropic_messages import AnthropicMessagesModel

_USER = [ChatMessage(role="user", content="hi")]


def _sse(events: list[dict[str, Any]]) -> bytes:
    lines = [f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events]
    return "".join(lines).encode()


def anthropic_transport(
    events: list[dict[str, Any]],
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
    body: str | None = None,
    requests: list[httpx.Request] | None = None,
) -> httpx.MockTransport:
    """Answer every request with these events as an SSE stream (or an error)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        content = body.encode() if body is not None else _sse(events)
        return httpx.Response(status, headers=headers, content=content)

    return httpx.MockTransport(handler)


def _model(
    transport: httpx.AsyncBaseTransport, **kwargs: Any
) -> AnthropicMessagesModel:
    return AnthropicMessagesModel(
        "claude-test", api_key="k", transport=transport, **kwargs
    )


async def _collect(
    model: AnthropicMessagesModel,
    messages: list[ChatMessage] | None = None,
    tools: list[ToolDefinition] | None = None,
) -> list[Delta]:
    return [d async for d in model.stream(messages or _USER, tools or [])]


def _start(usage: dict[str, int] | None = None) -> dict[str, Any]:
    return {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "claude-test",
            "stop_reason": None,
            "usage": usage or {"input_tokens": 25, "output_tokens": 1},
        },
    }


def _block(index: int, block: dict[str, Any]) -> dict[str, Any]:
    return {"type": "content_block_start", "index": index, "content_block": block}


def _delta(index: int, delta: dict[str, Any]) -> dict[str, Any]:
    return {"type": "content_block_delta", "index": index, "delta": delta}


def _stop(index: int) -> dict[str, Any]:
    return {"type": "content_block_stop", "index": index}


def _end(stop_reason: str, usage: dict[str, int] | None = None) -> list[dict[str, Any]]:
    message_delta: dict[str, Any] = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
    }
    if usage is not None:
        message_delta["usage"] = usage
    return [message_delta, {"type": "message_stop"}]


def _text_events(*texts: str, stop: str = "end_turn") -> list[dict[str, Any]]:
    return [
        _start(),
        _block(0, {"type": "text", "text": ""}),
        {"type": "ping"},
        *[_delta(0, {"type": "text_delta", "text": t}) for t in texts],
        _stop(0),
        *_end(stop, {"output_tokens": 15}),
    ]


def _tool_events(index: int = 0) -> list[dict[str, Any]]:
    return [
        _block(
            index, {"type": "tool_use", "id": "toolu_1", "name": "echo", "input": {}}
        ),
        _delta(index, {"type": "input_json_delta", "partial_json": ""}),
        _delta(index, {"type": "input_json_delta", "partial_json": '{"text":'}),
        _delta(index, {"type": "input_json_delta", "partial_json": ' "ping"}'}),
        _stop(index),
    ]


async def test_text_stream() -> None:
    deltas = await _collect(_model(anthropic_transport(_text_events("Hel", "lo"))))

    message = agglomerate_deltas(deltas)
    assert message.content == "Hello"
    assert message.finish_reason == "stop"
    assert message.usage == Usage(25, 15)
    assert sum(d.finish_reason is not None for d in deltas) == 1
    assert all(d.content or d.usage or d.finish_reason for d in deltas)


async def test_tool_use_stream() -> None:
    events = [_start(), *_tool_events(), *_end("tool_use", {"output_tokens": 9})]

    deltas = await _collect(_model(anthropic_transport(events)))

    message = agglomerate_deltas(deltas)
    assert message.tool_calls == [ToolCall("toolu_1", "echo", {"text": "ping"})]
    assert message.finish_reason == "tool_calls"
    fragments = [tc.arguments for d in deltas for tc in d.tool_calls]
    assert "" not in fragments


def _thinking_turn() -> list[dict[str, Any]]:
    return [
        _start(),
        _block(0, {"type": "thinking", "thinking": "", "signature": ""}),
        _delta(0, {"type": "thinking_delta", "thinking": "a"}),
        _delta(0, {"type": "thinking_delta", "thinking": "b"}),
        _delta(0, {"type": "signature_delta", "signature": "sig1"}),
        _stop(0),
        _block(1, {"type": "redacted_thinking", "data": "opaque"}),
        _stop(1),
        _block(2, {"type": "text", "text": ""}),
        _delta(2, {"type": "text_delta", "text": "Calling echo."}),
        _stop(2),
        *_tool_events(index=3),
        *_end("tool_use", {"output_tokens": 30}),
    ]


async def test_thinking_blocks_round_trip() -> None:
    requests: list[httpx.Request] = []
    model = _model(anthropic_transport(_thinking_turn(), requests=requests))

    message = agglomerate_deltas(await _collect(model))

    thinking = {"type": "thinking", "thinking": "ab", "signature": "sig1"}
    redacted = {"type": "redacted_thinking", "data": "opaque"}
    assert message.reasoning == "ab"
    assert message.provider_data == {"anthropic": {"blocks": [thinking, redacted]}}

    history = [
        ChatMessage("system", "sys"),
        ChatMessage("user", "hi"),
        message,
        ChatMessage("tool", "ping", tool_call_id="toolu_1"),
    ]
    await _collect(model, history)

    sent = json.loads(requests[1].content)["messages"]
    assert sent[1]["content"] == [
        thinking,
        redacted,
        {"type": "text", "text": "Calling echo."},
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "echo",
            "input": {"text": "ping"},
        },
    ]


async def test_usage_includes_cache_tokens() -> None:
    usage = {
        "input_tokens": 10,
        "cache_read_input_tokens": 30,
        "cache_creation_input_tokens": 5,
        "output_tokens": 1,
    }
    events = [
        _start(usage),
        _block(0, {"type": "text", "text": "x"}),
        _stop(0),
        *_end("end_turn", {"output_tokens": 7}),
    ]

    message = agglomerate_deltas(await _collect(_model(anthropic_transport(events))))

    assert message.usage == Usage(
        input_tokens=45, output_tokens=7, cached_input_tokens=30
    )


async def test_message_delta_without_usage_keeps_start_numbers() -> None:
    events = [_start(), _block(0, {"type": "text", "text": "x"}), *_end("end_turn")]

    message = agglomerate_deltas(await _collect(_model(anthropic_transport(events))))

    assert message.usage == Usage(25, 1)


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("tool_use", "tool_calls"),
        ("max_tokens", "length"),
        ("refusal", "content_filter"),
        ("pause_turn", "unknown"),
    ],
)
async def test_stop_reasons_are_mapped(stop_reason: str, expected: str) -> None:
    events = _text_events("x", stop=stop_reason)

    message = agglomerate_deltas(await _collect(_model(anthropic_transport(events))))

    assert message.finish_reason == expected


async def test_request_body() -> None:
    requests: list[httpx.Request] = []
    model = _model(
        anthropic_transport(_text_events("ok"), requests=requests),
        extra_body={"temperature": 0},
    )
    history = [
        ChatMessage("system", "sys"),
        ChatMessage("user", "hi"),
        ChatMessage(
            "assistant",
            tool_calls=[
                ToolCall("c1", "add", {"a": 1}),
                ToolCall("c2", "add", {"a": 2}),
            ],
        ),
        ChatMessage("tool", "1", tool_call_id="c1"),
        ChatMessage("tool", "2", tool_call_id="c2"),
        ChatMessage("user", "note"),
    ]
    definition = ToolDefinition("add", "Add.", {"type": "object"})

    await _collect(model, history, [definition])

    (request,) = requests
    assert request.headers["x-api-key"] == "k"
    assert request.headers["anthropic-version"] == "2023-06-01"
    body = json.loads(request.content)
    assert body["system"] == "sys"
    assert body["max_tokens"] == 4096
    assert body["stream"] is True
    assert body["temperature"] == 0
    assert body["messages"][0] == {
        "role": "user",
        "content": [{"type": "text", "text": "hi"}],
    }
    assert body["messages"][2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "1"},
            {"type": "tool_result", "tool_use_id": "c2", "content": "2"},
            {"type": "text", "text": "note"},
        ],
    }
    assert len(body["messages"]) == 3
    assert body["tools"] == [
        {"name": "add", "description": "Add.", "input_schema": {"type": "object"}}
    ]


def _error(
    status: int, message: str, kind: str = "api_error", **headers: str
) -> httpx.MockTransport:
    body = json.dumps({"type": "error", "error": {"type": kind, "message": message}})
    return anthropic_transport([], status=status, headers=headers, body=body)


def _connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused", request=request)


def _read_timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("slow", request=request)


@pytest.mark.parametrize(
    ("transport", "error_type", "status", "retry_after"),
    [
        (_error(401, "invalid x-api-key"), ModelAuthError, 401, None),
        (_error(403, "no access"), ModelAuthError, 403, None),
        (
            _error(429, "slow down", "rate_limit_error", **{"retry-after": "3"}),
            ModelRateLimitError,
            429,
            3.0,
        ),
        (_error(500, "boom"), ModelServerError, 500, None),
        (_error(504, "timeout"), ModelServerError, 504, None),
        (_error(529, "Overloaded", "overloaded_error"), ModelServerError, 529, None),
        (
            _error(400, "prompt is too long: 210000 tokens > 200000 maximum"),
            ModelContextLengthError,
            400,
            None,
        ),
        (_error(400, "bad field"), ModelBadRequestError, 400, None),
        (_error(404, "not found"), ModelBadRequestError, 404, None),
        (_error(413, "too large"), ModelBadRequestError, 413, None),
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


async def test_error_message_keeps_only_the_api_message() -> None:
    with pytest.raises(ModelAuthError) as caught:
        await _collect(_model(_error(401, "invalid x-api-key")))

    assert str(caught.value) == "HTTP 401 from the model endpoint: invalid x-api-key"


async def test_mid_stream_error_event() -> None:
    events = [
        _start(),
        _block(0, {"type": "text", "text": ""}),
        _delta(0, {"type": "text_delta", "text": "Partial"}),
        {
            "type": "error",
            "error": {"type": "overloaded_error", "message": "Overloaded"},
        },
    ]

    with pytest.raises(ModelServerError, match="Overloaded") as caught:
        await _collect(_model(anthropic_transport(events)))

    assert caught.value.status is None


async def test_unknown_events_are_ignored() -> None:
    events = _text_events("Hello")
    events.insert(3, {"type": "future_event", "detail": 1})
    events.insert(4, _delta(0, {"type": "citations_delta", "citation": {}}))

    deltas = await _collect(_model(anthropic_transport(events)))

    assert agglomerate_deltas(deltas).content == "Hello"
    assert all(d.content or d.usage or d.finish_reason for d in deltas)


async def test_adapter_makes_one_request() -> None:
    requests: list[httpx.Request] = []
    transport = anthropic_transport(
        [], status=529, body='{"type": "error"}', requests=requests
    )

    with pytest.raises(ModelServerError):
        await _collect(_model(transport))

    assert len(requests) == 1


async def test_body_without_an_event_stream_raises() -> None:
    transport = anthropic_transport([], body='{"content": []}')

    with pytest.raises(ModelServerError, match="without an event stream") as caught:
        await _collect(_model(transport))

    assert caught.value.retryable is False


@pytest.mark.parametrize(
    "events",
    [
        # A tool fragment for a block that was never opened.
        [_start(), _delta(0, {"type": "input_json_delta", "partial_json": "{"})],
        [
            {
                "type": "content_block_start",
                "content_block": {"type": "text", "text": "x"},
            }
        ],
        [{"type": "content_block_delta", "index": 0}],
    ],
    ids=["orphan_tool_fragment", "missing_index", "missing_delta"],
)
async def test_malformed_events_raise_model_errors(
    events: list[dict[str, Any]],
) -> None:
    with pytest.raises(ModelServerError, match="Malformed stream event") as caught:
        await _collect(_model(anthropic_transport(events)))

    assert caught.value.retryable is False
