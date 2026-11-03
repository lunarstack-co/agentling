"""A model adapter for OpenAI-compatible chat-completions endpoints.

Works with OpenAI and with servers that copy its API (LM Studio, vLLM, Ollama's
/v1, OpenRouter and others): point `base_url` at them. It streams text and tool
calls over httpx, reports usage and finish reasons, and turns HTTP failures
into agentling's ModelError classes. The agent loop does the retrying, so this
adapter never retries.

    model = OpenAIChatModel("gpt-4o-mini")  # reads OPENAI_API_KEY
    agent = Agent(model=model, tools=[...])
"""

from __future__ import annotations

import json
import os
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
    ModelContextLengthError,
    ModelError,
    ModelRateLimitError,
    ModelServerError,
    ModelTimeoutError,
    ToolCallDelta,
    ToolDefinition,
    Usage,
)

_FINISH_REASONS: dict[str, FinishReason] = {
    "stop": "stop",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "length": "length",
    "content_filter": "content_filter",
    "error": "error",
}


class OpenAIChatModel:
    """Streams chat completions from an OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        """Create the adapter.

        Args:
            model: The model name the endpoint expects.
            api_key: Sent as a bearer token. Defaults to OPENAI_API_KEY; local
                servers usually need none.
            base_url: The endpoint's base URL, up to and including /v1.
            timeout: httpx timeout in seconds for connecting and each read.
            transport: An httpx transport to use instead of the network
                (tests pass an httpx.MockTransport).
            extra_body: Fields merged into every request body last, so they
                win: sampling settings such as temperature, or a
                server's own options.
        """

        self.model = model
        self.api_key = (
            api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        )
        self.base_url = base_url.rstrip("/")
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
        decoder = _TurnDecoder()
        try:
            # One client per turn keeps the example short. Hold a long-lived
            # client on the instance if you need connection reuse.
            async with httpx.AsyncClient(
                transport=self.transport, timeout=self.timeout
            ) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json={
                        **request_body(self.model, messages, tools),
                        **self.extra_body,
                    },
                    headers=headers,
                ) as response:
                    if response.status_code != 200:
                        await response.aread()
                        raise error_for_response(response)
                    async for line in response.aiter_lines():
                        delta = decoder.decode(line)
                        if delta is not None:
                            yield delta
                    if not decoder.saw_event:
                        content_type = response.headers.get("content-type", "none")
                        raise ModelServerError(
                            "The endpoint answered without an event stream "
                            f"(Content-Type: {content_type}). Check that base_url "
                            "points at an OpenAI-compatible API, usually ending "
                            "in /v1.",
                            retryable=False,
                        )
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError(f"Request timed out: {exc}") from exc
        except httpx.TransportError as exc:
            raise ModelConnectionError(
                f"Could not reach {self.base_url}: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            # The rest of httpx's errors, such as a body that fails to decode.
            raise ModelServerError(f"Bad response from {self.base_url}: {exc}") from exc


def request_body(
    model: str, messages: Sequence[ChatMessage], tools: Sequence[ToolDefinition]
) -> dict[str, Any]:
    """Build the chat-completions request for one streamed turn."""

    body: dict[str, Any] = {
        "model": model,
        "messages": [_wire_message(message) for message in messages],
        "stream": True,
        "stream_options": {"include_usage": True},
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


def _wire_message(message: ChatMessage) -> dict[str, Any]:
    """Convert one ChatMessage to the chat-completions message shape."""

    if message.role == "tool":
        return {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    if message.role == "assistant":
        wire: dict[str, Any] = {"role": "assistant", "content": message.content or None}
        if message.tool_calls:
            wire["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    # The API expects the arguments as a JSON string.
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in message.tool_calls
            ]
        return wire
    return {"role": message.role, "content": message.content}


class _TurnDecoder:
    """Turns one turn's server-sent-event lines into Deltas.

    Remembers whether any event arrived, and which tool call a fragment
    belongs to when the server leaves out `index`.
    """

    def __init__(self) -> None:
        self.saw_event = False
        self._latest_call = -1
        self._call_by_id: dict[str, int] = {}

    def decode(self, line: str) -> Delta | None:
        """Turn one line into a Delta, or None when it carries nothing."""

        if not line.startswith("data:"):
            return None
        self.saw_event = True
        data = line[len("data:") :].strip()
        if not data or data == "[DONE]":
            return None
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ModelServerError(
                f"Invalid stream chunk: {data[:200]!r}", retryable=False
            ) from exc
        if not isinstance(chunk, dict):
            raise ModelServerError(
                f"Unexpected stream chunk: {data[:200]!r}", retryable=False
            )
        if chunk.get("error"):
            # OpenAI and OpenRouter report a failure that happens after the
            # 200 status as a chunk carrying an error.
            raise _stream_error(chunk["error"])
        return self._delta(chunk)

    def _delta(self, chunk: dict[str, Any]) -> Delta | None:
        """Convert a parsed chunk, dropping ones that carry nothing.

        The first chunk usually holds only the role. The loop treats any Delta
        as output and stops retrying, so an empty one must not be yielded.
        """

        choices = chunk.get("choices") or []
        choice: dict[str, Any] = choices[0] if choices else {}
        wire_delta: dict[str, Any] = choice.get("delta") or {}
        raw_reason = choice.get("finish_reason")
        raw_usage = chunk.get("usage")

        delta = Delta(
            content=wire_delta.get("content") or None,
            # DeepSeek and others stream reasoning separately from the answer.
            reasoning=wire_delta.get("reasoning_content") or None,
            tool_calls=[
                self._tool_call(call) for call in wire_delta.get("tool_calls") or []
            ],
            usage=_usage(raw_usage) if raw_usage else None,
            finish_reason=_FINISH_REASONS.get(raw_reason, "unknown")
            if raw_reason
            else None,
        )
        if not (
            delta.content
            or delta.reasoning
            or delta.tool_calls
            or delta.usage
            or delta.finish_reason
        ):
            return None
        return delta

    def _tool_call(self, call: dict[str, Any]) -> ToolCallDelta:
        function = call.get("function") or {}
        call_id = call.get("id") or None
        name = function.get("name") or None
        return ToolCallDelta(
            index=self._index_for(call.get("index"), call_id, name),
            id=call_id,
            name=name,
            arguments=function.get("arguments"),
        )

    def _index_for(self, index: object, call_id: str | None, name: str | None) -> int:
        """The fragment's call index, or a stand-in when the server sends none.

        Without an index, a fragment with an id this turn hasn't seen, or with
        a name and no id, starts the next call. Anything else continues the
        latest call.
        """

        if isinstance(index, int):
            if call_id is not None:
                self._call_by_id.setdefault(call_id, index)
            self._latest_call = max(self._latest_call, index)
            return index
        if call_id is not None:
            if call_id not in self._call_by_id:
                self._latest_call += 1
                self._call_by_id[call_id] = self._latest_call
            return self._call_by_id[call_id]
        if name is not None:
            self._latest_call += 1
        return max(self._latest_call, 0)


def _usage(raw: dict[str, Any]) -> Usage:
    """Read token counts, including the optional detail breakdowns."""

    prompt_details = raw.get("prompt_tokens_details") or {}
    completion_details = raw.get("completion_tokens_details") or {}
    return Usage(
        input_tokens=raw.get("prompt_tokens") or 0,
        output_tokens=raw.get("completion_tokens") or 0,
        cached_input_tokens=prompt_details.get("cached_tokens") or 0,
        reasoning_tokens=completion_details.get("reasoning_tokens") or 0,
    )


def error_for_response(response: httpx.Response) -> ModelError:
    """Map a failed HTTP response to the ModelError class the loop expects."""

    if response.has_redirect_location:
        # Following it would turn this POST into a GET (httpx does what
        # browsers do), so report where the endpoint moved instead.
        return ModelBadRequestError(
            f"HTTP {response.status_code}: the endpoint redirects to "
            f"{response.headers['location']}. Update base_url to match.",
            status=response.status_code,
        )
    return _classify(
        response.status_code,
        _error_text(response),
        response.text,
        _retry_after(response),
    )


def _stream_error(error: object) -> ModelError:
    """Classify an error the endpoint reported inside the stream."""

    if not isinstance(error, dict):
        return ModelServerError(f"The model endpoint reported: {str(error)[:200]}")
    message = str(error.get("message") or "an error with no message")[:200]
    code = error.get("code")
    # OpenRouter puts the HTTP status in `code`; OpenAI uses a string there.
    if isinstance(code, int):
        return _classify(code, message, json.dumps(error), retry_after=None)
    return ModelServerError(f"The model endpoint reported: {message}")


def _classify(
    status: int, message: str, body: str, retry_after: float | None
) -> ModelError:
    """Pick the ModelError class for a status code and its error body."""

    message = f"HTTP {status} from the model endpoint: {message}"
    # Gemini's OpenAI-compatible endpoint reports a bad key as a 400.
    if status in (401, 403) or "API_KEY_INVALID" in body:
        return ModelAuthError(message, status=status)
    if status == 429:
        return ModelRateLimitError(message, status=status, retry_after=retry_after)
    if status >= 500:
        return ModelServerError(message, status=status)
    if "context_length_exceeded" in body:
        return ModelContextLengthError(message, status=status)
    return ModelBadRequestError(message, status=status)


def _error_text(response: httpx.Response) -> str:
    """The backend's own error message, or a short excerpt of the body.

    Bodies can echo parts of the request (OpenAI's 401 includes a masked key),
    and exception messages end up in logs, so only the message field is kept.
    """

    try:
        payload = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return str(error["message"])[:200]
    return response.text[:200]


def _retry_after(response: httpx.Response) -> float | None:
    """Read Retry-After in seconds, never below zero.

    The HTTP-date form is ignored.
    """

    try:
        return max(0.0, float(response.headers["retry-after"]))
    except (KeyError, ValueError):
        return None
