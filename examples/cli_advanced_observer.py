"""Prints every event a streamed run produces, and counts each type.

Runs offline with a scripted model:

    uv run python -m examples.cli_advanced_observer

Also sets a context_manager that trims the prompt, and a model_timeout.
"""

from __future__ import annotations

import asyncio

from agentling import (
    Agent,
    ChatMessage,
    FinalEvent,
    StepEvent,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
    Usage,
    tool,
)
from agentling.testing import ScriptedModel


@tool
def health_check() -> str:
    """Report a one-line service health status."""
    return "all systems nominal"


def _assistant(
    content: str = "", tool_calls: list[ToolCall] | None = None
) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls or [],
        usage=Usage(1, 1),
    )


def keep_recent(messages: list[ChatMessage]) -> list[ChatMessage]:
    """A context_manager that keeps the system prompt and the recent tail."""

    if len(messages) <= 6:
        return messages
    return [messages[0], *messages[-5:]]


def build_agent() -> Agent:
    """Build an offline agent that calls one tool, then answers."""

    model = ScriptedModel(
        [
            _assistant(
                tool_calls=[ToolCall(id="c1", name="health_check", arguments={})]
            ),
            _assistant(content="All systems nominal."),
        ]
    )
    return Agent(
        model=model,
        tools=[health_check],
        context_manager=keep_recent,
        model_timeout=30.0,
    )


async def observe(agent: Agent) -> dict[str, int]:
    """Stream a run, print each event, and tally how many of each type arrived."""

    counts: dict[str, int] = {}
    session = agent.start()
    async for event in session.run("Check system health.", stream=True):
        counts[type(event).__name__] = counts.get(type(event).__name__, 0) + 1
        match event:
            case TextDelta(text=text):
                print(text, end="", flush=True)
            case ToolCallEvent(tool_call=call):
                print(f"\n-> {call.name}()")
            case ToolResultEvent(result=result):
                print(f"<- {result.content}")
            case StepEvent():
                print("[step recorded]")
            case FinalEvent(answer=answer, status=status):
                print(f"\n= [{status}] {answer}")
    return counts


async def main() -> None:
    counts = await observe(build_agent())
    print("\nEvent counts:", counts)


if __name__ == "__main__":
    asyncio.run(main())
