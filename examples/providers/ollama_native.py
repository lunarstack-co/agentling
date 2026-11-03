"""A model adapter for Ollama's native /api/chat endpoint.

Streams text, thinking and tool calls over httpx and turns HTTP failures into
agentling's ModelError classes. The agent loop does the retrying, so this
adapter never retries. For Ollama's OpenAI-compatible /v1 endpoint, use
openai_chat.py instead.

Ollama differs from the OpenAI shape in ways this file handles:

- The stream is newline-delimited JSON, not server-sent events.
- Tool-call arguments arrive as a JSON object, often all at once.
- Tool results name the tool (`tool_name`) rather than the call id.
- `done_reason` says "stop" even when the model called a tool.
- A prompt longer than the context window is truncated, not rejected, so
  this adapter never raises ModelContextLengthError.

    model = OllamaModel("qwen3:8b")
    agent = Agent(model=model, tools=[...])
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from agentling import (
    ChatMessage,
    Delta,
    FinishReason,
    ModelAuthError,
    ModelBadRequestError,
    ModelConnectionError,
    ModelError,
    ModelRateLimitError,
    ModelServerError,
    ModelTimeoutError,
    ToolCallDelta,
    ToolDefinition,
    Usage,
)

_FINISH_REASONS: dict[str, FinishReason] = {"stop": "stop", "length": "length"}


class OllamaModel:
    """Streams chat responses from Ollama's native API."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        api_key: str | None = None,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        """Create the adapter.

        Args:
            model: The model tag, for example "qwen3:8b".
            base_url: Where Ollama listens, without /api.
            api_key: Sent as a bearer token, for a hosted Ollama or a proxy.
                A local server needs none.
            timeout: httpx timeout in seconds for connecting and each read.
                The first request can be slow while the model loads.
            transport: An httpx transport to use instead of the network
                (tests pass an httpx.MockTransport).
            extra_body: Fields merged into every request body last, so they
                win: `options` such as temperature or num_ctx, `think`,
                `keep_alive`.
        """

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.transport = transport
        self.extra_body = dict(extra_body or {})

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition] = (),
    ) -> AsyncIterator[Delta]:
        """Stream one assistant turn as Deltas."""

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {**request_body(self.model, messages, tools), **self.extra_body}
        decoder = _TurnDecoder()
        try:
            async with httpx.AsyncClient(
                transport=self.transport, timeout=self.timeout
            ) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/api/chat", json=body, headers=headers
                ) as response:
                    if response.status_code != 200:
                        await response.aread()
                        raise error_for_response(response)
                    async for line in response.aiter_lines():
                        for delta in decoder.decode(line):
                            yield delta
            if not decoder.done:
                raise ModelServerError(
                    "The stream ended before Ollama sent its final chunk.",
                    retryable=False,
                )
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError(f"Request timed out: {exc}") from exc
        except httpx.TransportError as exc:
            raise ModelConnectionError(
                f"Could not reach {self.base_url}: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelServerError(f"Bad response from {self.base_url}: {exc}") from exc


def request_body(
    model: str, messages: Sequence[ChatMessage], tools: Sequence[ToolDefinition]
) -> dict[str, Any]:
    """Build the /api/chat request for one streamed turn."""

    body: dict[str, Any] = {
        "model": model,
        "messages": _wire_messages(messages),
        "stream": True,
    }
    if tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]
    return body


def _wire_messages(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """Convert the conversation, naming the tool behind each tool result."""

    tool_names: dict[str, str] = {}
    wire: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool":
            wire.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_name": tool_names.get(message.tool_call_id or "", ""),
                }
            )
            continue
        entry: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.role == "assistant":
            if message.reasoning:
                entry["thinking"] = message.reasoning
            if message.tool_calls:
                entry["tool_calls"] = [
                    {"function": {"name": call.name, "arguments": call.arguments}}
                    for call in message.tool_calls
                ]
                tool_names.update({call.id: call.name for call in message.tool_calls})
        wire.append(entry)
    return wire


class _TurnDecoder:
    """Turns one turn's NDJSON lines into Deltas."""

    def __init__(self) -> None:
        self.done = False
        self._calls = 0

    def decode(self, line: str) -> list[Delta]:
        """Turn one line into zero or more Deltas."""

        if not line.strip():
            return []
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ModelServerError(
                f"Invalid stream chunk: {line[:200]!r}", retryable=False
            ) from exc
        if not isinstance(chunk, dict):
            raise ModelServerError(
                f"Unexpected stream chunk: {line[:200]!r}", retryable=False
            )
        if chunk.get("error"):
            # Ollama reports a failure after the 200 status as a final line.
            raise ModelServerError(
                f"The model endpoint reported: {str(chunk['error'])[:200]}"
            )

        message = chunk.get("message") or {}
        if not isinstance(message, dict) or not all(
            isinstance(call, dict) for call in message.get("tool_calls") or []
        ):
            raise ModelServerError(
                f"Malformed stream chunk: {line[:200]!r}", retryable=False
            )
        deltas: list[Delta] = []
        if message.get("content") or message.get("thinking"):
            deltas.append(
                Delta(
                    content=message.get("content") or None,
                    reasoning=message.get("thinking") or None,
                )
            )
        for call in message.get("tool_calls") or []:
            deltas.append(Delta(tool_calls=[self._tool_call(call)]))
        if chunk.get("done"):
            self.done = True
            deltas.append(self._final(chunk))
        return deltas

    def _tool_call(self, call: dict[str, Any]) -> ToolCallDelta:
        function = call.get("function") or {}
        index = function.get("index")
        if not isinstance(index, int):
            index = self._calls
        self._calls += 1
        return ToolCallDelta(
            index=index,
            id=call.get("id") or None,
            name=function.get("name"),
            arguments=json.dumps(function.get("arguments") or {}),
        )

    def _final(self, chunk: dict[str, Any]) -> Delta:
        reason: FinishReason = (
            "tool_calls"
            if self._calls
            else _FINISH_REASONS.get(chunk.get("done_reason") or "", "unknown")
        )
        return Delta(
            usage=Usage(
                input_tokens=chunk.get("prompt_eval_count") or 0,
                output_tokens=chunk.get("eval_count") or 0,
            ),
            finish_reason=reason,
        )


def error_for_response(response: httpx.Response) -> ModelError:
    """Map a failed HTTP response to the ModelError class the loop expects."""

    status = response.status_code
    message = f"HTTP {status} from the model endpoint: {_error_text(response)}"
    if status in (401, 403):
        return ModelAuthError(message, status=status)
    if status == 429:
        return ModelRateLimitError(
            message, status=status, retry_after=_retry_after(response)
        )
    if status >= 500:
        return ModelServerError(message, status=status)
    # A missing model is a 404, a bad request or a model without tool
    # support a 400. None of them goes away by trying again.
    return ModelBadRequestError(message, status=status)


def _error_text(response: httpx.Response) -> str:
    """Ollama's own error message, or a short excerpt of the body."""

    try:
        payload = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        return str(payload["error"])[:200]
    return response.text[:200]


def _retry_after(response: httpx.Response) -> float | None:
    """Read Retry-After in seconds, never below zero."""

    try:
        return max(0.0, float(response.headers["retry-after"]))
    except (KeyError, ValueError):
        return None
