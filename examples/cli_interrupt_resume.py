"""Interrupt a streamed run after a tool result, then resume the same task.

Runs offline with a scripted model:

    uv run python -m examples.cli_interrupt_resume
"""

from __future__ import annotations

import asyncio

from agentling import Agent, ChatMessage, FinalEvent, ToolCall, ToolResultEvent, tool
from agentling.testing import ScriptedModel


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b


def build_agent() -> Agent:
    """Build an offline agent scripted to add twice and answer."""
    model = ScriptedModel(
        [
            ChatMessage(
                "assistant",
                tool_calls=[ToolCall("c1", "add", {"a": 19, "b": 23})],
            ),
            ChatMessage(
                "assistant",
                tool_calls=[ToolCall("c2", "add", {"a": 42, "b": 3})],
            ),
            ChatMessage("assistant", "The result is 45."),
        ]
    )
    return Agent(model=model, tools=[add])


async def main() -> None:
    session = build_agent().start()
    interrupted = False
    final: FinalEvent | None = None
    async for event in session.run("Add 19 and 23, then add 3.", stream=True):
        if isinstance(event, ToolResultEvent) and not interrupted:
            session.interrupt()
            interrupted = True
        elif isinstance(event, FinalEvent):
            final = event

    if final is None or final.status != "interrupted":
        raise RuntimeError("Expected an interrupted run")
    print(final.status)
    # Exhaust the original stream before resuming so the session is released.
    answer = await session.resume()
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
