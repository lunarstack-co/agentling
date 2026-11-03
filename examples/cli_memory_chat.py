"""Saves a session's memory to JSON, loads it into a new session, and continues.

Runs offline with a scripted model:

    uv run python -m examples.cli_memory_chat

The second session continues with reset=False and streams, so the
FinalEvent's status is printed with the answer.
"""

from __future__ import annotations

import asyncio

from agentling import (
    Agent,
    ChatMessage,
    FinalEvent,
    Memory,
    Usage,
)
from agentling.testing import ScriptedModel


def _assistant(content: str) -> ChatMessage:
    return ChatMessage(role="assistant", content=content, usage=Usage(1, 1))


def build_agent() -> Agent:
    """Build an offline agent scripted for a two-turn conversation."""

    model = ScriptedModel(
        [
            _assistant("Nice to meet you, Sam."),
            _assistant("Your name is Sam."),
        ]
    )
    return Agent(model=model)


async def main() -> None:
    agent = build_agent()

    first = agent.start()
    await first.run("Hi, my name is Sam.")
    saved = first.memory.dump_json()
    print(f"Saved {len(first.memory.steps)} steps to JSON.\n")

    # A fresh session, or another process, can restore and continue.
    second = agent.start()
    second.memory = Memory.load_json(saved)

    answer, status = "", "unknown"
    async for event in second.run("What is my name?", reset=False, stream=True):
        if isinstance(event, FinalEvent):
            answer, status = event.answer, event.status
    print(f"[{status}] {answer}")


if __name__ == "__main__":
    asyncio.run(main())
