"""Events the agent loop yields while a run is streaming.

There are five: TextDelta, ToolCallEvent, ToolResultEvent, StepEvent and
FinalEvent. Memory is the saved record of a run; events are the live view of it,
and StepEvent carries each step as it is recorded.

Within a step the order is fixed: its TextDelta chunks, then every
ToolCallEvent, then every ToolResultEvent in the same order, then the StepEvent.
A run that ends normally emits exactly one FinalEvent, last. A run that fails
raises instead.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import IO, Literal

from .memory import Step, ToolResult
from .models import ToolCall, Usage

RunStatus = Literal["completed", "interrupted", "max_steps"]


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A chunk of streamed assistant text."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolCallEvent:
    """Emitted before the agent executes a tool call."""

    tool_call: ToolCall


@dataclass(frozen=True, slots=True)
class ToolResultEvent:
    """Emitted after a tool call completes, successfully or with an error."""

    result: ToolResult


@dataclass(frozen=True, slots=True)
class StepEvent:
    """Emitted after a step has been recorded to memory."""

    step: Step


@dataclass(frozen=True, slots=True)
class FinalEvent:
    """Emitted once when the run ends, carrying the answer, usage, and status.

    status is "completed" (the model finished), "interrupted" (a caller
    interrupted it) or "max_steps" (the step limit forced a final answer).
    """

    answer: str
    usage: Usage | None = None
    status: RunStatus = "completed"


Event = TextDelta | ToolCallEvent | ToolResultEvent | StepEvent | FinalEvent


def _truncate(text: str, limit: int = 500) -> str:
    """Shorten long tool output so a live stream stays readable."""

    return text if len(text) <= limit else text[:limit] + "..."


async def print_events(
    events: AsyncIterator[Event], *, file: IO[str] | None = None
) -> str:
    """Render an agent's event stream as it arrives.

    Takes the iterator from run(..., stream=True). Text prints as it streams,
    each tool call and result gets its own line, and the final answer prints
    last. Returns the final answer. Writes to `file`, or stdout if it is None.
    """

    answer = ""
    mid_line = False  # True while streamed text has left the cursor mid-line.

    async for event in events:
        match event:
            case TextDelta(text=text):
                print(text, end="", flush=True, file=file)
                mid_line = True
            case ToolCallEvent(tool_call=call):
                if mid_line:
                    print(file=file)
                    mid_line = False
                print(f"-> {call.name}({json.dumps(call.arguments)})", file=file)
            case ToolResultEvent(result=result):
                status = "error" if result.is_error else "ok"
                print(f"<- [{status}] {_truncate(result.content)}", file=file)
            case FinalEvent(answer=final):
                if mid_line:
                    print(file=file)
                    mid_line = False
                answer = final
                print(f"= {final}", file=file)
            case StepEvent():
                pass  # Its contents already surfaced through the events above.

    return answer
