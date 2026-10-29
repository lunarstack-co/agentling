"""Typed steps and the agent's memory.

The loop records each turn as a TaskStep, ActionStep or FinalStep rather than as
raw messages. Steps keep usage, timing and tool results, and render themselves
into the messages the model sees. Memory holds the steps and saves to JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

from .errors import MemoryLoadError
from .models import ChatMessage, ToolCall, Usage

# Serialization format version written by to_dict and checked by from_dict.
_MEMORY_VERSION = 1


@dataclass
class ToolResult:
    """The outcome of a single executed tool call."""

    tool_call_id: str
    name: str
    content: str
    is_error: bool = False
    # "validation", "execution", "timeout" or "unknown_tool"; None on success.
    # Picks the recovery hint the model sees.
    error_kind: str | None = None


@dataclass
class TaskStep:
    """A task that started (or, in multi-turn, continued) the run."""

    task: str

    def to_messages(self) -> list[ChatMessage]:
        """Render as a single user message."""

        return [ChatMessage(role="user", content=self.task)]


@dataclass
class ActionStep:
    """One loop iteration: the model's message, its tool results, usage and timing."""

    model_message: ChatMessage
    tool_results: list[ToolResult] = field(default_factory=list)
    # The loop doesn't set this; it is kept for callers and saved runs.
    error: str | None = None
    usage: Usage | None = None
    duration: float | None = None

    def to_messages(self) -> list[ChatMessage]:
        """Render the assistant turn, then one tool message per result."""

        messages: list[ChatMessage] = [self.model_message]

        for result in self.tool_results:
            content = result.content
            # Only a validation error is fixed by changing the arguments.
            if result.is_error:
                if result.error_kind == "validation":
                    hint = "Fix the arguments and try again."
                else:
                    hint = "Consider a different approach or tool."
                content = f"Error from {result.name!r}: {result.content}. {hint}"
            messages.append(
                ChatMessage(
                    role="tool", content=content, tool_call_id=result.tool_call_id
                )
            )

        return messages


@dataclass
class FinalStep:
    """The terminal answer produced by the run."""

    answer: str

    def to_messages(self) -> list[ChatMessage]:
        """Render as the assistant's final message (keeps multi-turn history whole)."""

        return [ChatMessage(role="assistant", content=self.answer)]


# A run's history is a sequence of these three step kinds.
Step = TaskStep | ActionStep | FinalStep


@dataclass
class Memory:
    """Ordered history of an agent run, as typed steps."""

    # The tag saved with each step so from_dict() can rebuild the right class.
    _KIND: ClassVar[dict[type, str]] = {
        TaskStep: "task",
        ActionStep: "action",
        FinalStep: "final",
    }

    steps: list[Step] = field(default_factory=list)

    def add(self, step: Step) -> None:
        """Append a step to the run's history."""

        self.steps.append(step)

    def reset(self) -> None:
        """Clear all steps (used by run(reset=True) to start fresh)."""

        self.steps.clear()

    def to_messages(self, system_prompt: str) -> list[ChatMessage]:
        """Render the full conversation: system prompt, then every step."""

        # The system prompt isn't stored as a step, so it is always first and
        # never saved twice.
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=system_prompt)
        ]
        for step in self.steps:
            messages.extend(step.to_messages())
        return messages

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict, tagging each step with its kind."""

        # asdict() drops the nested types; from_dict() rebuilds them using the
        # "type" tag. "version" lets the format change later.
        return {
            "version": _MEMORY_VERSION,
            "steps": [
                {"type": self._KIND[type(step)], "data": asdict(step)}
                for step in self.steps
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Memory:
        """Rebuild a Memory from to_dict() output.

        Raises MemoryLoadError for an unknown version or a malformed payload.
        """

        if not isinstance(data, dict):
            raise MemoryLoadError("Serialized memory must be a JSON object.")

        version = data.get("version")
        if version != _MEMORY_VERSION:
            raise MemoryLoadError(
                f"Unsupported memory version {version!r} (expected {_MEMORY_VERSION})."
            )

        entries = data.get("steps", [])
        if not isinstance(entries, list):
            raise MemoryLoadError("Memory 'steps' must be a list.")

        try:
            steps = [_step_from_dict(entry) for entry in entries]
        except (KeyError, TypeError) as exc:
            raise MemoryLoadError(f"Malformed memory step: {exc}") from exc

        return cls(steps=steps)

    def dump_json(self) -> str:
        """Serialize the memory to a JSON string."""

        return json.dumps(self.to_dict())

    @classmethod
    def load_json(cls, raw: str) -> Memory:
        """Rebuild a Memory from dump_json() output.

        Invalid JSON raises json.JSONDecodeError; a bad shape raises
        MemoryLoadError.
        """

        return cls.from_dict(json.loads(raw))


def _chat_message_from_dict(d: dict[str, Any]) -> ChatMessage:
    """Reconstruct a ChatMessage (and its nested ToolCalls/Usage) from a dict."""

    return ChatMessage(
        role=d["role"],
        content=d["content"],
        tool_calls=[ToolCall(**tc) for tc in d["tool_calls"]],
        tool_call_id=d["tool_call_id"],
        usage=Usage(**d["usage"]) if d["usage"] else None,
    )


def _step_from_dict(entry: dict[str, Any]) -> Step:
    """Reconstruct a Step from a {"type", "data"} entry, dispatching on the tag."""

    kind, data = entry["type"], entry["data"]

    if kind == "task":
        return TaskStep(**data)
    if kind == "final":
        return FinalStep(**data)
    if kind == "action":
        return ActionStep(
            model_message=_chat_message_from_dict(data["model_message"]),
            tool_results=[ToolResult(**tr) for tr in data["tool_results"]],
            error=data["error"],
            usage=Usage(**data["usage"]) if data["usage"] else None,
            duration=data["duration"],
        )

    raise MemoryLoadError(f"Unknown step type: {kind!r}")
