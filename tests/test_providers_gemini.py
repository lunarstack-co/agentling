"""Gemini's native reference adapter, exercised offline over httpx."""

from __future__ import annotations

import copy
import json
from pathlib import Path
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
from agentling.testing import check_model, read_fixture
from examples.providers.gemini_native import GeminiModel
from examples.providers.record import record

_USER = [ChatMessage("user", "hi")]
_FIXTURE = Path("examples/providers/fixtures/gemini_native.jsonl")


def _chunk(*parts: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "index": 0,
        "content": {"role": "model", "parts": list(parts)},
    }
    if finish:
        candidate["finishReason"] = finish
    return {"candidates": [candidate]}


def _transport(
    events: list[Any],
    *,
    requests: list[httpx.Request] | None = None,
    status: int = 200,
    body: str | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        content = (
            body
            if body is not None
            else "".join(f"data: {json.dumps(e)}\n\n" for e in events)
        )
        return httpx.Response(status, content=content, headers=headers)

    return httpx.MockTransport(handler)


def _model(transport: httpx.AsyncBaseTransport, **kwargs: Any) -> GeminiModel:
    return GeminiModel("gemini-test", api_key="test-key", transport=transport, **kwargs)


async def _collect(
    model: GeminiModel,
    messages: list[ChatMessage] | None = None,
    tools: list[ToolDefinition] | None = None,
) -> list[Delta]:
    return [d async for d in model.stream(messages or _USER, tools or [])]


async def test_text_thoughts_usage_and_metadata_stream() -> None:
    events = [
        {"modelVersion": "gemini-test"},
        _chunk({"thought": True, "text": "Thinking", "thoughtSignature": "signature"}),
        _chunk({"text": "Hel"}),
        _chunk({"text": "lo"}, finish="STOP"),
        {
            "usageMetadata": {
                "promptTokenCount": 20,
                "cachedContentTokenCount": 5,
                "candidatesTokenCount": 6,
                "thoughtsTokenCount": 8,
                "toolUsePromptTokenCount": 3,
            }
        },
    ]
    deltas = await _collect(_model(_transport(events)))
    result = agglomerate_deltas(deltas)
    assert result.content == "Hello"
    assert result.reasoning == "Thinking"
    assert result.usage == Usage(23, 14, cached_input_tokens=5, reasoning_tokens=8)
    assert result.finish_reason == "stop"
    assert sum(d.finish_reason is not None for d in deltas) == 1
    assert all(
        d.content or d.reasoning or d.tool_calls or d.usage or d.finish_reason
        for d in deltas
    )
    assert deltas[-1].provider_data == {
        "gemini": {
            "parts": [
                events[1]["candidates"][0]["content"]["parts"][0],
                {"text": "Hel"},
                {"text": "lo"},
            ]
        }
    }


async def test_parallel_calls_and_signatures_round_trip_without_native_ids() -> None:
    parts: list[dict[str, Any]] = [
        {"text": "Calling tools.", "thoughtSignature": "text-signature"},
        {
            "functionCall": {"name": "echo", "args": {"text": "a"}},
            "thoughtSignature": "call-signature",
        },
        {"functionCall": {"name": "echo", "args": {"text": "b"}}},
    ]
    requests: list[httpx.Request] = []
    model = _model(_transport([_chunk(*parts, finish="STOP")], requests=requests))
    deltas = await _collect(model)
    result = agglomerate_deltas(deltas)
    assert result.finish_reason == "tool_calls"
    assert [c.index for d in deltas for c in d.tool_calls] == [0, 1]
    assert [(c.name, c.arguments) for c in result.tool_calls] == [
        ("echo", {"text": "a"}),
        ("echo", {"text": "b"}),
    ]
    history = [
        ChatMessage("user", "call"),
        result,
        *[
            ChatMessage("tool", f"result-{i}", tool_call_id=c.id)
            for i, c in enumerate(result.tool_calls)
        ],
        ChatMessage("user", "continue"),
    ]
    before = copy.deepcopy(history)
    await _collect(model, history)
    contents = json.loads(requests[-1].content)["contents"]
    assert contents[1] == {"role": "model", "parts": parts}
    assert contents[2] == {
        "role": "user",
        "parts": [
            {"functionResponse": {"name": "echo", "response": {"result": "result-0"}}},
            {"functionResponse": {"name": "echo", "response": {"result": "result-1"}}},
            {"text": "continue"},
        ],
    }
    assert history == before


async def test_native_call_ids_are_returned_in_function_responses() -> None:
    requests: list[httpx.Request] = []
    model = _model(
        _transport(
            [
                _chunk(
                    {"functionCall": {"id": "native-1", "name": "echo", "args": {}}},
                    finish="STOP",
                )
            ],
            requests=requests,
        )
    )
    result = agglomerate_deltas(await _collect(model))
    assert result.tool_calls == [ToolCall("native-1", "echo", {})]
    await _collect(
        model, [*_USER, result, ChatMessage("tool", "done", tool_call_id="native-1")]
    )
    assert (
        json.loads(requests[-1].content)["contents"][-1]["parts"][0][
            "functionResponse"
        ]["id"]
        == "native-1"
    )


async def test_request_schema_system_extra_body_and_other_provider_data() -> None:
    requests: list[httpx.Request] = []
    model = GeminiModel(
        "models/gemini-test",
        api_key="key",
        base_url="https://test/v1beta/",
        transport=_transport(
            [_chunk({"text": "ok"}, finish="STOP")], requests=requests
        ),
        extra_body={"generationConfig": {"maxOutputTokens": 100}, "safetySettings": []},
    )
    messages = [
        ChatMessage("system", "sys1"),
        ChatMessage("system", "sys2"),
        *_USER,
        ChatMessage(
            "assistant",
            content="calling",
            tool_calls=[ToolCall("c1", "echo", {"text": "x"})],
            provider_data={"anthropic": {"blocks": [{"secret": "foreign"}]}},
        ),
        ChatMessage("tool", "x", tool_call_id="c1"),
    ]
    tools = [
        ToolDefinition(
            "echo",
            "Echo.",
            {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "additionalProperties": False,
            },
        )
    ]
    before = copy.deepcopy((messages, tools))
    await _collect(model, messages, tools)
    (request,) = requests
    assert (
        str(request.url)
        == "https://test/v1beta/models/gemini-test:streamGenerateContent?alt=sse"
    )
    assert request.headers["x-goog-api-key"] == "key"
    assert "key=" not in str(request.url)
    body = json.loads(request.content)
    assert body["systemInstruction"] == {"parts": [{"text": "sys1\n\nsys2"}]}
    assert body["tools"] == [
        {
            "functionDeclarations": [
                {
                    "name": "echo",
                    "description": "Echo.",
                    "parametersJsonSchema": tools[0].parameters,
                }
            ]
        }
    ]
    assert body["generationConfig"] == {"maxOutputTokens": 100}
    assert body["safetySettings"] == []
    assert body["contents"][1] == {
        "role": "model",
        "parts": [
            {"text": "calling"},
            {"functionCall": {"id": "c1", "name": "echo", "args": {"text": "x"}}},
        ],
    }
    assert "foreign" not in request.content.decode()
    assert (messages, tools) == before


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("STOP", "stop"),
        ("MAX_TOKENS", "length"),
        ("SAFETY", "content_filter"),
        ("RECITATION", "content_filter"),
        ("BLOCKLIST", "content_filter"),
        ("PROHIBITED_CONTENT", "content_filter"),
        ("SPII", "content_filter"),
        ("MALFORMED_FUNCTION_CALL", "error"),
        ("UNEXPECTED_TOOL_CALL", "error"),
        ("OTHER", "unknown"),
        ("FUTURE_REASON", "unknown"),
    ],
)
async def test_finish_reason_mapping(reason: str, expected: str) -> None:
    result = agglomerate_deltas(
        await _collect(_model(_transport([_chunk({"text": "x"}, finish=reason)])))
    )
    assert result.finish_reason == expected
    assert result.usage is None


async def test_prompt_block_is_a_content_filter_finish() -> None:
    result = agglomerate_deltas(
        await _collect(
            _model(_transport([{"promptFeedback": {"blockReason": "SAFETY"}}]))
        )
    )
    assert result.finish_reason == "content_filter"
    assert result.content == ""


async def test_usage_is_cumulative_and_second_candidate_is_ignored() -> None:
    events = [
        {"usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 1}},
        {
            "candidates": [
                {"index": 1, "content": {"parts": [{"text": "ignore"}]}},
                {
                    "index": 0,
                    "content": {"parts": [{"text": "answer"}]},
                    "finishReason": "STOP",
                },
            ],
            "usageMetadata": {"candidatesTokenCount": 4},
        },
    ]
    result = agglomerate_deltas(await _collect(_model(_transport(events))))
    assert result.content == "answer"
    assert result.usage == Usage(7, 4)


@pytest.mark.parametrize(
    "status,error_type,message",
    [
        (400, ModelBadRequestError, "bad field"),
        (401, ModelAuthError, "bad key"),
        (403, ModelAuthError, "denied"),
        (404, ModelBadRequestError, "not found"),
        (408, ModelTimeoutError, "timeout"),
        (429, ModelRateLimitError, "slow down"),
        (500, ModelServerError, "internal"),
        (503, ModelServerError, "unavailable"),
        (504, ModelServerError, "deadline"),
        (
            400,
            ModelContextLengthError,
            "The input token count exceeds the maximum number of tokens allowed",
        ),
    ],
)
async def test_http_errors_are_classified_without_retries(
    status: int, error_type: type[ModelError], message: str
) -> None:
    requests: list[httpx.Request] = []
    transport = _transport(
        [],
        status=status,
        body=json.dumps(
            {"error": {"code": status, "message": message}, "request": "private body"}
        ),
        requests=requests,
        headers={"retry-after": "3"},
    )
    with pytest.raises(error_type) as caught:
        await _collect(_model(transport))
    assert type(caught.value) is error_type
    assert caught.value.status == status
    assert caught.value.retry_after == (3.0 if status == 429 else None)
    assert len(requests) == 1
    assert "private body" not in str(caught.value)


async def test_invalid_api_key_reported_as_400_is_auth_error() -> None:
    error = {
        "error": {
            "code": 400,
            "message": "API key not valid",
            "details": [{"reason": "API_KEY_INVALID"}],
        }
    }
    with pytest.raises(ModelAuthError):
        await _collect(_model(_transport([], status=400, body=json.dumps(error))))


@pytest.mark.parametrize("retry_after,expected", [("-2", 0.0), ("bad", None)])
async def test_retry_after_is_parsed(retry_after: str, expected: float | None) -> None:
    with pytest.raises(ModelRateLimitError) as caught:
        await _collect(
            _model(_transport([], status=429, headers={"retry-after": retry_after}))
        )
    assert caught.value.retry_after == expected


@pytest.mark.parametrize(
    "exception,error_type",
    [
        (httpx.ConnectError, ModelConnectionError),
        (httpx.ReadTimeout, ModelTimeoutError),
        (httpx.RemoteProtocolError, ModelConnectionError),
    ],
)
async def test_transport_errors_are_classified(
    exception: type[httpx.TransportError], error_type: type[ModelError]
) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise exception("failure", request=request)

    with pytest.raises(error_type):
        await _collect(_model(httpx.MockTransport(fail)))


async def test_midstream_error_is_classified() -> None:
    events = [
        _chunk({"text": "partial"}),
        {"error": {"code": 503, "message": "unavailable"}},
    ]
    with pytest.raises(ModelServerError, match="unavailable"):
        await _collect(_model(_transport(events)))


@pytest.mark.parametrize(
    "events",
    [
        [[]],
        [{"candidates": [None]}],
        [_chunk({"functionCall": {"args": {}}})],
        [_chunk({"functionCall": {"name": "echo", "args": []}})],
        [_chunk({"text": 1})],
        [_chunk({"functionCall": {"id": 1, "name": "echo"}})],
        [{"candidates": [{"finishReason": ["STOP"]}]}],
    ],
)
async def test_malformed_stream_events_fail_as_model_errors(events: list[Any]) -> None:
    with pytest.raises(ModelServerError, match="Malformed") as caught:
        await _collect(_model(_transport(events)))
    assert caught.value.retryable is False


@pytest.mark.parametrize(
    "body",
    ["data: invalid\n\n", "data: null\n\n", '{"candidates": []}', "data: {}\n\n"],
)
async def test_invalid_or_incomplete_streams_fail(body: str) -> None:
    with pytest.raises(ModelServerError) as caught:
        await _collect(_model(_transport([], body=body)))
    assert caught.value.retryable is False


async def test_orphan_tool_response_is_bad_request() -> None:
    with pytest.raises(ModelBadRequestError, match="matching function call"):
        await _collect(
            _model(_transport([])),
            [ChatMessage("tool", "result", tool_call_id="missing")],
        )


@pytest.mark.parametrize("count", [True, -1, "3"])
async def test_malformed_usage_does_not_escape_as_invalid_usage(count: Any) -> None:
    events = [
        {
            **_chunk({"text": "answer"}, finish="STOP"),
            "usageMetadata": {"cachedContentTokenCount": count},
        }
    ]
    with pytest.raises(ModelServerError, match="Malformed") as caught:
        await _collect(_model(_transport(events)))
    assert caught.value.retryable is False


def test_key_environment_and_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert GeminiModel("m").api_key == "google-key"
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    assert GeminiModel("m").api_key == "gemini-key"
    assert GeminiModel("m", api_key="override").api_key == "override"
    assert GeminiModel("m", api_key="").api_key == ""


async def test_recorder_redacts_google_key_and_replays_contract(tmp_path: Path) -> None:
    responses = read_fixture(_FIXTURE)[:2]

    def handler(request: httpx.Request) -> httpx.Response:
        exchange = responses.pop(0)
        return httpx.Response(200, content="\n".join(exchange.body_lines))

    out = tmp_path / "gemini.jsonl"
    report = await record(
        "gemini_native",
        model="gemini-3.8-flash",
        api_key="private-google-key",
        out=out,
        inner=httpx.MockTransport(handler),
    )
    assert report.ok, str(report)
    assert "private-google-key" not in out.read_text()
    assert {e.request["headers"]["x-goog-api-key"] for e in read_fixture(out)} == {
        "‹redacted›"
    }
    replay = await check_model(lambda t: GeminiModel("m", transport=t), recorded=out)
    assert replay.ok, str(replay)
