"""A copyable adapter for Gemini's native streaming generateContent API.

Uses httpx, not Google's SDK, and never retries; the agent loop owns retries.
Raw model parts round-trip under provider_data["gemini"]["parts"], preserving
thought signatures on text and function calls without merging or reordering them.

    model = GeminiModel("gemini-3.8-flash")  # reads GEMINI_API_KEY
    agent = Agent(model=model, tools=[...])
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import AsyncIterator, Sequence
from typing import Any
from urllib.parse import quote

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
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "IMAGE_SAFETY": "content_filter",
    "IMAGE_PROHIBITED_CONTENT": "content_filter",
    "MALFORMED_FUNCTION_CALL": "error",
    "UNEXPECTED_TOOL_CALL": "error",
    "TOO_MANY_TOOL_CALLS": "error",
    "MISSING_THOUGHT_SIGNATURE": "error",
    "MALFORMED_RESPONSE": "error",
}
_NAMESPACE = "gemini"


class GeminiModel:
    """Stream one candidate from Gemini's native REST API."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        """Configure the endpoint, optional request overrides and test transport.

        `model` is a model name, optionally prefixed with `models/`. The key
        defaults to GEMINI_API_KEY, then GOOGLE_API_KEY. `extra_body` is merged
        last, for settings such as generationConfig and safetySettings.
        """
        self.model = model.removeprefix("models/")
        self.api_key = (
            api_key
            if api_key is not None
            else os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport
        self.extra_body = copy.deepcopy(extra_body or {})

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition] = (),
    ) -> AsyncIterator[Delta]:
        """Yield text, visible thoughts and complete function-call parts as Deltas."""
        body = {**request_body(messages, tools), **self.extra_body}
        headers = {"x-goog-api-key": self.api_key} if self.api_key else {}
        url = (
            f"{self.base_url}/models/{quote(self.model, safe='')}:streamGenerateContent"
        )
        decoder = _TurnDecoder()
        try:
            async with httpx.AsyncClient(
                transport=self.transport, timeout=self.timeout
            ) as client:
                async with client.stream(
                    "POST", url, params={"alt": "sse"}, json=body, headers=headers
                ) as response:
                    if response.status_code != 200:
                        await response.aread()
                        raise error_for_response(response)
                    async for line in response.aiter_lines():
                        for delta in decoder.decode(line):
                            yield delta
            yield decoder.finish()
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError(f"Request timed out: {exc}") from exc
        except httpx.TransportError as exc:
            raise ModelConnectionError(
                f"Could not reach {self.base_url}: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelServerError(f"Bad response from {self.base_url}: {exc}") from exc


def request_body(
    messages: Sequence[ChatMessage], tools: Sequence[ToolDefinition]
) -> dict[str, Any]:
    """Map system instructions, history and JSON Schema tool declarations."""
    body: dict[str, Any] = {"contents": _wire_messages(messages)}
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if tools:
        body["tools"] = [
            {
                "functionDeclarations": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parametersJsonSchema": copy.deepcopy(tool.parameters),
                    }
                    for tool in tools
                ]
            }
        ]
    return body


def _model_parts(message: ChatMessage) -> list[dict[str, Any]]:
    data = (message.provider_data or {}).get(_NAMESPACE)
    if isinstance(data, dict) and isinstance(data.get("parts"), list):
        return copy.deepcopy(data["parts"])
    parts: list[dict[str, Any]] = []
    if message.content:
        parts.append({"text": message.content})
    parts.extend(
        {
            "functionCall": {
                "id": c.id,
                "name": c.name,
                "args": copy.deepcopy(c.arguments),
            }
        }
        for c in message.tool_calls
    )
    return parts


def _wire_messages(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    wire: list[dict[str, Any]] = []
    # Agentling synthesises absent IDs; don't send those back as native IDs.
    calls: dict[str, tuple[str, str | None]] = {}
    for message in messages:
        if message.role == "system":
            continue
        role = "model" if message.role == "assistant" else "user"
        if message.role == "assistant":
            parts = _model_parts(message)
            native_calls = [p["functionCall"] for p in parts if "functionCall" in p]
            for call, native in zip(message.tool_calls, native_calls):
                calls[call.id] = (call.name, native.get("id"))
        elif message.role == "tool":
            if message.tool_call_id not in calls:
                raise ModelBadRequestError("Tool result has no matching function call.")
            name, native_id = calls[message.tool_call_id]
            response: dict[str, Any] = {
                "name": name,
                "response": {"result": message.content},
            }
            if native_id:
                response["id"] = native_id
            parts = [{"functionResponse": response}]
        else:
            parts = [{"text": message.content}] if message.content else []
        if not parts:
            continue
        if wire and wire[-1]["role"] == role:
            wire[-1]["parts"].extend(parts)
        else:
            wire.append({"role": role, "parts": parts})
    return wire


class _TurnDecoder:
    """Keep cumulative metadata for the final Delta; stream useful parts now."""

    def __init__(self) -> None:
        self._parts: list[dict[str, Any]] = []
        self._tool_count = 0
        self._usage: dict[str, int] = {}
        self._finish_reason: str | None = None
        self._saw_event = False

    def decode(self, line: str) -> list[Delta]:
        if not line.startswith("data:"):
            return []
        self._saw_event = True
        try:
            event = json.loads(line[5:].strip())
            if not isinstance(event, dict):
                raise TypeError("event must be an object")
            if "error" in event:
                error = event["error"]
                raise _classify(int(error.get("code", 500)), error)
            usage = event.get("usageMetadata")
            if isinstance(usage, dict):
                for key in (
                    "promptTokenCount",
                    "toolUsePromptTokenCount",
                    "candidatesTokenCount",
                    "thoughtsTokenCount",
                    "cachedContentTokenCount",
                ):
                    if key in usage:
                        value = usage[key]
                        if type(value) is not int or value < 0:
                            raise TypeError(
                                "token counts must be non-negative integers"
                            )
                        self._usage[key] = value
            if (event.get("promptFeedback") or {}).get("blockReason"):
                self._finish_reason = "SAFETY"
            candidates = event.get("candidates") or []
            # Agentling represents one answer, even if extra_body requests several.
            candidate: dict[str, Any] = next(
                (c for c in candidates if c.get("index", 0) == 0), {}
            )
            if candidate.get("finishReason"):
                if not isinstance(candidate["finishReason"], str):
                    raise TypeError("finish reason must be a string")
                self._finish_reason = candidate["finishReason"]
            result: list[Delta] = []
            for part in (candidate.get("content") or {}).get("parts", []):
                if not isinstance(part, dict):
                    raise TypeError("part must be an object")
                self._parts.append(copy.deepcopy(part))
                if part.get("text"):
                    if not isinstance(part["text"], str):
                        raise TypeError("text must be a string")
                    if part.get("thought"):
                        result.append(Delta(reasoning=part["text"]))
                    else:
                        result.append(Delta(content=part["text"]))
                if "functionCall" in part:
                    call = part["functionCall"]
                    if not isinstance(call.get("name"), str) or not call["name"]:
                        raise TypeError("function call must have a name")
                    if call.get("id") is not None and not isinstance(call["id"], str):
                        raise TypeError("function call id must be a string")
                    args = call.get("args", {})
                    if not isinstance(args, dict):
                        raise TypeError("function arguments must be an object")
                    result.append(
                        Delta(
                            tool_calls=[
                                ToolCallDelta(
                                    index=self._tool_count,
                                    id=call.get("id"),
                                    name=call["name"],
                                    arguments=json.dumps(args),
                                )
                            ]
                        )
                    )
                    self._tool_count += 1
            return result
        except (ValueError, TypeError, AttributeError) as exc:
            raise ModelServerError(
                "Malformed Gemini stream event.", retryable=False
            ) from exc

    def finish(self) -> Delta:
        if not self._saw_event or self._finish_reason is None:
            raise ModelServerError(
                "Gemini stream ended without a finish reason.", retryable=False
            )
        reason = _FINISH_REASONS.get(self._finish_reason, "unknown")
        if reason == "stop" and self._tool_count:
            reason = "tool_calls"
        usage = None
        if self._usage:
            # Gemini reports thoughts separately from candidate output tokens.
            usage = Usage(
                input_tokens=self._usage.get("promptTokenCount", 0)
                + self._usage.get("toolUsePromptTokenCount", 0),
                output_tokens=self._usage.get("candidatesTokenCount", 0)
                + self._usage.get("thoughtsTokenCount", 0),
                cached_input_tokens=self._usage.get("cachedContentTokenCount", 0),
                reasoning_tokens=self._usage.get("thoughtsTokenCount", 0),
            )
        return Delta(
            usage=usage,
            finish_reason=reason,
            provider_data={_NAMESPACE: {"parts": self._parts}} if self._parts else None,
        )


def error_for_response(response: httpx.Response) -> ModelError:
    """Classify HTTP errors without leaking request headers or whole bodies."""
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    return _classify(response.status_code, error, response.headers.get("retry-after"))


def _classify(status: int, error: Any, retry_after: str | None = None) -> ModelError:
    detail = error if isinstance(error, dict) else {}
    message = str(detail.get("message") or "The model endpoint rejected the request.")[
        :200
    ]
    text = f"HTTP {status} from the model endpoint: {message}"
    if status in (401, 403) or "API_KEY_INVALID" in str(detail.get("details", "")):
        return ModelAuthError(text, status=status)
    if status == 429:
        try:
            delay = max(0.0, float(retry_after)) if retry_after is not None else None
        except ValueError:
            delay = None
        return ModelRateLimitError(text, status=status, retry_after=delay)
    if status == 408:
        return ModelTimeoutError(text, status=status)
    if status >= 500:
        return ModelServerError(text, status=status)
    if status == 400 and any(
        marker in message.casefold()
        for marker in (
            "input token count",
            "maximum number of tokens",
            "context length",
        )
    ):
        return ModelContextLengthError(text, status=status)
    return ModelBadRequestError(text, status=status)
