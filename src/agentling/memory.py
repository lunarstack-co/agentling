"""Typed steps and the agent's memory.

The loop records each turn as a TaskStep, ActionStep or FinalStep rather than as
raw messages. Steps keep usage, timing and tool results, and render themselves
into the messages the model sees. Memory holds the steps and saves to JSON.

The saved format is versioned. Version 2 added fields to Usage and ChatMessage;
version 1 saves still load, with the new fields at their defaults.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, ClassVar

from .errors import MemoryLoadError
from .models import ChatMessage, ToolCall, Usage

# Serialization format version written by to_dict, and the ones from_dict reads.
_MEMORY_VERSION = 2
_READABLE_VERSIONS = (1, 2)


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
    """The terminal answer produced by the run.

    Only the answer text is kept. reasoning and provider_data from a text-only
    final turn are not stored; tool-calling turns keep them on ActionStep.
    """

    answer: str

    def to_messages(self) -> list[ChatMessage]:
        """Render as the assistant's final message."""

        return [ChatMessage(role="assistant", content=self.answer)]


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

        # asdict() loses the step classes; from_dict() rebuilds them from "type".
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

        Reads format versions 1 and 2. Keys a step doesn't define are ignored
        and missing optional ones take their defaults. Raises MemoryLoadError
        for an unknown version, an unknown step type, or a missing required
        field.
        """

        if not isinstance(data, dict):
            raise MemoryLoadError("Serialized memory must be a JSON object.")

        version = data.get("version")
        # type() rather than isinstance(): True == 1 and 1.0 == 1 in Python.
        if type(version) is not int or version not in _READABLE_VERSIONS:
            raise MemoryLoadError(
                f"Unsupported memory version {version!r} "
                f"(expected one of {_READABLE_VERSIONS})."
            )

        entries = data.get("steps", [])
        if not isinstance(entries, list):
            raise MemoryLoadError("Memory 'steps' must be a list.")

        try:
            steps = [_step_from_dict(entry) for entry in entries]
        except (AttributeError, KeyError, TypeError) as exc:
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


def _known(cls: type[Any], data: dict[str, Any]) -> dict[str, Any]:
    """Keep only the keys that are fields of the dataclass `cls`."""

    names = {f.name for f in fields(cls)}
    return {key: value for key, value in data.items() if key in names}


def _usage_from_dict(d: dict[str, Any] | None) -> Usage | None:
    """Reconstruct a Usage, or None when none was saved."""

    return Usage(**_known(Usage, d)) if d else None


def _chat_message_from_dict(d: dict[str, Any]) -> ChatMessage:
    """Reconstruct a ChatMessage (and its nested ToolCalls/Usage) from a dict."""

    return ChatMessage(
        **{
            **_known(ChatMessage, d),
            "tool_calls": [
                ToolCall(**_known(ToolCall, tc)) for tc in d.get("tool_calls", [])
            ],
            "usage": _usage_from_dict(d.get("usage")),
        }
    )


def _step_from_dict(entry: dict[str, Any]) -> Step:
    """Reconstruct a Step from a {"type", "data"} entry, dispatching on the tag."""

    kind, data = entry["type"], entry["data"]

    if kind == "task":
        return TaskStep(**_known(TaskStep, data))
    if kind == "final":
        return FinalStep(**_known(FinalStep, data))
    if kind == "action":
        return ActionStep(
            **{
                **_known(ActionStep, data),
                "model_message": _chat_message_from_dict(data["model_message"]),
                "tool_results": [
                    ToolResult(**_known(ToolResult, tr))
                    for tr in data.get("tool_results", [])
                ],
                "usage": _usage_from_dict(data.get("usage")),
            }
        )

    raise MemoryLoadError(f"Unknown step type: {kind!r}")
