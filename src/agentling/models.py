"""Provider-neutral message, tool and streaming types, and the Model protocol.

Agentling ships no model adapters: an app passes any object with a matching
stream() method, written for whichever backend it uses.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from .errors import ModelOutputError

Role = Literal["system", "user", "assistant", "tool"]

# Adapters map their backend's stop reasons onto these, "unknown" for the rest.
FinishReason = Literal[
    "stop", "tool_calls", "length", "content_filter", "error", "unknown"
]


@dataclass(frozen=True)
class ToolDefinition:
    """What a model is told about a tool. Adapters render it in their wire format."""

    name: str
    description: str
    # JSON Schema for the arguments object.
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    """A tool call requested by the model, in Agentling's own format."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolCallDelta:
    """A piece of a streamed tool call, grouped with the others by `index`.

    `id` may be missing (some local servers never send one); Agentling then
    makes one up.
    """

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str | None = None


@dataclass
class Delta:
    """A single streamed model chunk."""

    content: str | None = None
    tool_calls: list[ToolCallDelta] = field(default_factory=list)
    usage: Usage | None = None
    reasoning: str | None = None
    finish_reason: FinishReason | None = None
    # Adapter-owned data to send back on later turns (for example thinking
    # signatures), keyed by adapter namespace. Must be JSON-serialisable.
    provider_data: dict[str, Any] | None = None


@dataclass(frozen=True)
class Usage:
    """Token usage returned by a model provider."""

    input_tokens: int
    output_tokens: int
    # Breakdowns some backends report; already included in the two totals.
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed (input + output)."""
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        """Sum two usages field-wise (used to accumulate a run's total)."""
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


@dataclass
class ChatMessage:
    """A chat message in Agentling's own format; adapters convert it."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    usage: Usage | None = None
    reasoning: str | None = None
    finish_reason: FinishReason | None = None
    provider_data: dict[str, Any] | None = None


class Model(Protocol):
    """The one method Agentling needs from a model backend.

    Write an adapter for your backend that follows these rules:

    1. Yield no `Delta` until it carries content, reasoning, a tool-call
       fragment, usage or a finish_reason. Swallow metadata-only chunks: the
       loop only retries a turn that failed before its first Delta.
    2. Yield `usage` at least once when the backend reports it (the last one
       wins), and none when it doesn't.
    3. Yield `finish_reason` exactly once, on or before the last Delta.
    4. Raise only `ModelError` subclasses, with `retryable` set correctly. Never
       let a vendor or transport exception escape, and never retry yourself.
    5. Never mutate the `messages` or `tools` you were given.
    6. Render the system message and tool results however the backend requires
       (a top-level parameter, grouped blocks, and so on).
    7. Read back only your own `provider_data` namespace and ignore the others.
       Everything you put there must be JSON-serialisable. It is kept on
       tool-calling turns, which are the ones sent back to you.
    """

    def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition] = (),
    ) -> AsyncIterator[Delta]:
        """Stream the assistant response as incremental deltas."""
        ...


async def deltas_from_message(message: ChatMessage) -> AsyncIterator[Delta]:
    """Yield a whole message as one Delta, for backends that don't stream.

    In stream(): `async for delta in deltas_from_message(reply): yield delta`.
    """

    yield Delta(
        content=message.content or None,
        tool_calls=[
            ToolCallDelta(
                index=index,
                id=call.id,
                name=call.name,
                arguments=json.dumps(call.arguments),
            )
            for index, call in enumerate(message.tool_calls)
        ],
        usage=message.usage,
        reasoning=message.reasoning,
        finish_reason=message.finish_reason,
        provider_data=message.provider_data,
    )


@dataclass
class _ToolCallFragment:
    """Collects the pieces of one streamed tool call."""

    id: str | None = None
    name: str | None = None
    arguments: str = ""


def parse_tool_arguments(arguments: str) -> dict[str, Any]:
    """Parse tool-call arguments; anything but a JSON object is ModelOutputError."""

    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError as exc:
        raise ModelOutputError(f"Invalid tool call arguments: {arguments!r}") from exc

    if not isinstance(parsed, dict):
        raise ModelOutputError(
            f"Tool call arguments must be a JSON object: {arguments!r}"
        )

    return parsed


def agglomerate_deltas(deltas: Iterable[Delta]) -> ChatMessage:
    """Reassemble streamed deltas into a single ChatMessage.

    Content, reasoning and each tool call's arguments (grouped by `index`) are
    concatenated. For usage, finish_reason and provider_data the last one wins.
    """

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    fragments: dict[int, _ToolCallFragment] = {}
    usage: Usage | None = None
    finish_reason: FinishReason | None = None
    provider_data: dict[str, Any] | None = None

    for delta in deltas:
        if delta.content:
            content_parts.append(delta.content)
        if delta.reasoning:
            reasoning_parts.append(delta.reasoning)

        for tool_call in delta.tool_calls:
            fragment = fragments.setdefault(tool_call.index, _ToolCallFragment())
            if tool_call.id:
                fragment.id = tool_call.id
            if tool_call.name:
                fragment.name = tool_call.name
            if tool_call.arguments:
                fragment.arguments += tool_call.arguments

        if delta.usage is not None:
            usage = delta.usage
        if delta.finish_reason is not None:
            finish_reason = delta.finish_reason
        if delta.provider_data is not None:
            provider_data = delta.provider_data

    tool_calls: list[ToolCall] = []
    for index, fragment in sorted(fragments.items()):
        if not fragment.name:
            raise ModelOutputError("Streamed tool call is missing a name.")
        tool_calls.append(
            ToolCall(
                # The id pairs the call with its result message, so a backend
                # that sends none (or "") still needs one.
                id=fragment.id or f"agentling_call_{index}_{secrets.token_hex(4)}",
                name=fragment.name,
                arguments=parse_tool_arguments(fragment.arguments),
            )
        )

    return ChatMessage(
        role="assistant",
        content="".join(content_parts),
        tool_calls=tool_calls,
        usage=usage,
        reasoning="".join(reasoning_parts) if reasoning_parts else None,
        finish_reason=finish_reason,
        provider_data=provider_data,
    )
