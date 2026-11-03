"""The OpenAI-compatible reference adapter in examples/providers, run offline."""

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

import examples.cli_math_tutor as math_tutor
import examples.cli_notes_agent as notes_agent
import examples.cli_repo_assistant as repo_assistant
from agentling import (
    Agent,
    ChatMessage,
    Delta,
    ModelAuthError,
    ModelBadRequestError,
    ModelConnectionError,
    ModelContextLengthError,
    ModelError,
    ModelRateLimitError,
    ModelServerError,
    ToolCall,
    ToolDefinition,
    Usage,
    agglomerate_deltas,
)
from examples.providers.openai_chat import OpenAIChatModel

_USER = [ChatMessage(role="user", content="hi")]


def _sse(chunks: list[dict[str, Any]]) -> bytes:
    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def sse_transport(
    chunks: list[dict[str, Any]],
    status: int = 200,
    headers: dict[str, str] | None = None,
    body: str | None = None,
    requests: list[httpx.Request] | None = None,
) -> httpx.MockTransport:
    """Answer every request with these chunks as an SSE stream (or an error)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        content = body.encode() if body is not None else _sse(chunks)
        return httpx.Response(status, headers=headers, content=content)

    return httpx.MockTransport(handler)


def _model(transport: httpx.AsyncBaseTransport) -> OpenAIChatModel:
    return OpenAIChatModel("test-model", api_key="k", transport=transport)


async def _collect(model: OpenAIChatModel, **kwargs: Any) -> list[Delta]:
    return [delta async for delta in model.stream(_USER, **kwargs)]


def _choice(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}


async def test_text_stream() -> None:
    chunks = [
        _choice({"role": "assistant", "content": ""}),
        _choice({"content": "Hel"}),
        _choice({"content": "lo"}),
        _choice({}, finish_reason="stop"),
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 1},
            },
        },
    ]

    deltas = await _collect(_model(sse_transport(chunks)))

    assert [d.content for d in deltas[:2]] == ["Hel", "lo"]
    assert all(d.content or d.finish_reason or d.usage for d in deltas)
    message = agglomerate_deltas(deltas)
    assert message.content == "Hello"
    assert message.finish_reason == "stop"
    assert message.usage == Usage(5, 2, cached_input_tokens=1)
    assert sum(d.finish_reason is not None for d in deltas) == 1


async def test_tool_call_without_ids() -> None:
    chunks = [
        _choice({"role": "assistant"}),
        _choice(
            {"tool_calls": [{"index": 0, "function": {"name": "add", "arguments": ""}}]}
        ),
        _choice({"tool_calls": [{"index": 0, "function": {"arguments": '{"a": 1'}}]}),
        _choice({"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]}),
        _choice({}, finish_reason="tool_calls"),
    ]

    message = agglomerate_deltas(await _collect(_model(sse_transport(chunks))))

    assert message.finish_reason == "tool_calls"
    (call,) = message.tool_calls
    assert call.name == "add"
    assert call.arguments == {"a": 1}
    assert call.id.startswith("agentling_call_0_")


async def test_reasoning_is_streamed_separately() -> None:
    chunks = [_choice({"reasoning_content": "hmm"}), _choice({"content": "ok"})]
    message = agglomerate_deltas(await _collect(_model(sse_transport(chunks))))
    assert (message.reasoning, message.content) == ("hmm", "ok")


# The body Gemini's OpenAI-compatible endpoint returns for an invalid key.
_GEMINI_BAD_KEY = json.dumps(
    [
        {
            "error": {
                "code": 400,
                "message": "API key not valid. Please pass a valid API key.",
                "status": "INVALID_ARGUMENT",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                        "reason": "API_KEY_INVALID",
                        "domain": "googleapis.com",
                    }
                ],
            }
        }
    ]
)


def _connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused", request=request)


@pytest.mark.parametrize(
    ("transport", "error_type", "status", "retry_after"),
    [
        (sse_transport([], status=401, body="no"), ModelAuthError, 401, None),
        (sse_transport([], status=403, body="no"), ModelAuthError, 403, None),
        (
            sse_transport([], status=429, body="slow", headers={"Retry-After": "3"}),
            ModelRateLimitError,
            429,
            3.0,
        ),
        (sse_transport([], status=500, body="oops"), ModelServerError, 500, None),
        (sse_transport([], status=503, body="oops"), ModelServerError, 503, None),
        (sse_transport([], status=404, body="?"), ModelBadRequestError, 404, None),
        (
            sse_transport(
                [], status=400, body='{"error": {"code": "context_length_exceeded"}}'
            ),
            ModelContextLengthError,
            400,
            None,
        ),
        (sse_transport([], status=400, body="bad"), ModelBadRequestError, 400, None),
        (
            sse_transport([], status=400, body=_GEMINI_BAD_KEY),
            ModelAuthError,
            400,
            None,
        ),
        (httpx.MockTransport(_connect_error), ModelConnectionError, None, None),
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


_OPENAI_BAD_KEY = json.dumps(
    {
        "error": {
            "message": "Incorrect API key provided: sk-abc***xyz.",
            "type": "invalid_request_error",
            "code": "invalid_api_key",
        }
    }
)


async def test_error_message_keeps_only_the_backend_message() -> None:
    transport = sse_transport([], status=401, body=_OPENAI_BAD_KEY)

    with pytest.raises(ModelAuthError) as caught:
        await _collect(_model(transport))

    assert str(caught.value) == (
        "HTTP 401 from the model endpoint: Incorrect API key provided: sk-abc***xyz."
    )
    assert "invalid_api_key" not in str(caught.value)


async def test_error_message_falls_back_to_a_short_body_excerpt() -> None:
    transport = sse_transport([], status=502, body="<html>" + "x" * 1000)

    with pytest.raises(ModelServerError) as caught:
        await _collect(_model(transport))

    assert len(str(caught.value)) < 260


@pytest.mark.parametrize(
    ("error_chunk", "error_type", "status"),
    [
        # OpenAI: an error object and no choices, after the 200 status.
        (
            {"error": {"message": "boom", "type": "server_error"}},
            ModelServerError,
            None,
        ),
        # OpenRouter: the HTTP status in `code`, and finish_reason "error".
        (
            {
                "error": {"code": 429, "message": "boom"},
                "choices": [
                    {"index": 0, "delta": {"content": ""}, "finish_reason": "error"}
                ],
            },
            ModelRateLimitError,
            429,
        ),
        ({"error": "boom"}, ModelServerError, None),
    ],
)
async def test_mid_stream_error_chunk_raises(
    error_chunk: dict[str, Any], error_type: type[ModelError], status: int | None
) -> None:
    transport = sse_transport([_choice({"content": "Partial "}), error_chunk])

    with pytest.raises(error_type, match="boom") as caught:
        await _collect(_model(transport))

    assert type(caught.value) is error_type
    assert caught.value.status == status


async def test_mid_stream_error_fails_the_run_instead_of_answering() -> None:
    chunks = [_choice({"content": "Partial "}), {"error": {"message": "boom"}}]
    model = _model(sse_transport(chunks))

    with pytest.raises(ModelServerError, match="boom"):
        await Agent(model=model, retry=None).run("x")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("stop", "stop"),
        ("function_call", "tool_calls"),
        ("error", "error"),
        ("eos", "unknown"),
    ],
)
async def test_finish_reasons_are_mapped(raw: str, expected: str) -> None:
    chunks = [_choice({"content": "x"}, finish_reason=raw)]
    message = agglomerate_deltas(await _collect(_model(sse_transport(chunks))))
    assert message.finish_reason == expected


@pytest.mark.parametrize("line", ["data: null", 'data: "x"', "data: [1]", "data: {bad"])
async def test_malformed_chunks_raise_model_errors(line: str) -> None:
    with pytest.raises(ModelServerError) as caught:
        await _collect(_model(sse_transport([], body=line + "\n\n")))

    assert caught.value.retryable is False


async def test_undecodable_body_raises_a_model_error() -> None:
    transport = sse_transport(
        [], body="data: {}\n\n", headers={"Content-Encoding": "gzip"}
    )

    with pytest.raises(ModelServerError, match="Bad response"):
        await _collect(_model(transport))


async def test_redirect_is_reported_not_followed() -> None:
    location = "https://api.example.com/v1/chat/completions"
    transport = sse_transport([], status=301, body="", headers={"Location": location})

    with pytest.raises(
        ModelBadRequestError, match="redirects to https://api"
    ) as caught:
        await _collect(_model(transport))

    assert caught.value.status == 301


async def test_body_without_an_event_stream_raises() -> None:
    body = json.dumps({"choices": [{"message": {"content": "hi"}}]})
    transport = sse_transport(
        [], body=body, headers={"Content-Type": "application/json"}
    )

    with pytest.raises(ModelServerError, match="without an event stream") as caught:
        await _collect(_model(transport))

    assert caught.value.retryable is False


@pytest.mark.parametrize(
    ("header", "expected"),
    [("-5", 0.0), ("Wed, 21 Oct 2015 07:28:00 GMT", None)],
)
async def test_retry_after_header_parsing(header: str, expected: float | None) -> None:
    transport = sse_transport(
        [], status=429, body="slow", headers={"Retry-After": header}
    )

    with pytest.raises(ModelRateLimitError) as caught:
        await _collect(_model(transport))

    assert caught.value.retry_after == expected


def _tool_chunk(*calls: dict[str, Any]) -> dict[str, Any]:
    return _choice({"tool_calls": list(calls)})


async def _tool_calls(chunks: list[dict[str, Any]]) -> list[ToolCall]:
    deltas = await _collect(_model(sse_transport(chunks)))
    return agglomerate_deltas(deltas).tool_calls


async def test_whole_calls_without_index_stay_separate() -> None:
    calls = await _tool_calls(
        [
            _tool_chunk(
                {"id": "a", "function": {"name": "f", "arguments": '{"x": 1}'}}
            ),
            _tool_chunk(
                {"id": "b", "function": {"name": "g", "arguments": '{"y": 2}'}}
            ),
        ]
    )

    assert calls == [
        ToolCall(id="a", name="f", arguments={"x": 1}),
        ToolCall(id="b", name="g", arguments={"y": 2}),
    ]


async def test_null_index_without_ids_starts_a_call_per_name() -> None:
    calls = await _tool_calls(
        [
            _tool_chunk({"index": None, "function": {"name": "f", "arguments": "{}"}}),
            _tool_chunk({"index": None, "function": {"name": "g", "arguments": "{}"}}),
        ]
    )

    assert [call.name for call in calls] == ["f", "g"]
    assert calls[0].id.startswith("agentling_call_0_")
    assert calls[1].id.startswith("agentling_call_1_")


async def test_fragments_without_index_continue_the_current_call() -> None:
    calls = await _tool_calls(
        [
            _tool_chunk({"id": "a", "function": {"name": "f", "arguments": '{"x"'}}),
            _tool_chunk({"function": {"arguments": ": 1"}}),
            # Some servers repeat the id on every fragment of the same call.
            _tool_chunk({"id": "a", "function": {"arguments": "}"}}),
        ]
    )

    assert calls == [ToolCall(id="a", name="f", arguments={"x": 1})]


async def test_request_body() -> None:
    requests: list[httpx.Request] = []
    model = _model(sse_transport([_choice({"content": "ok"})], requests=requests))
    messages = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content="hi"),
        ChatMessage(
            role="assistant",
            tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1})],
        ),
        ChatMessage(role="tool", content="1", tool_call_id="c1"),
    ]
    definition = ToolDefinition("add", "Add.", {"type": "object"})

    [_ async for _ in model.stream(messages, [definition])]

    (request,) = requests
    assert request.headers["authorization"] == "Bearer k"
    body = json.loads(request.content)
    assert body["model"] == "test-model"
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["messages"][2]["tool_calls"][0]["function"] == {
        "name": "add",
        "arguments": '{"a": 1}',
    }
    assert body["messages"][3] == {"role": "tool", "content": "1", "tool_call_id": "c1"}
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add.",
                "parameters": {"type": "object"},
            },
        }
    ]


async def test_extra_body_is_merged_last() -> None:
    requests: list[httpx.Request] = []
    transport = sse_transport([_choice({"content": "ok"})], requests=requests)
    model = OpenAIChatModel(
        "m",
        api_key="k",
        transport=transport,
        extra_body={"temperature": 0, "stream_options": {"include_usage": False}},
    )

    await _collect(model)

    body = json.loads(requests[0].content)
    assert body["temperature"] == 0
    assert body["stream_options"] == {"include_usage": False}


async def test_runs_a_full_agent_turn() -> None:
    chunks = [_choice({"content": "42"}), _choice({}, finish_reason="stop")]
    assert await Agent(model=_model(sse_transport(chunks))).run("x") == "42"


@pytest.mark.parametrize(
    "build_agent",
    [math_tutor.build_agent, repo_assistant.build_agent, notes_agent.build_agent],
)
def test_api_examples_default_to_the_reference_adapter(
    build_agent: Callable[[], Agent], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTLING_EXAMPLE_MODEL", "m")

    agent = build_agent()

    assert isinstance(agent.model, OpenAIChatModel)
    assert agent.model.model == "m"
