"""Memory-persistence demo: dump a session, reload it, and continue.

Runs fully offline with a scripted model, so no API key is needed:

    uv run python examples/cli_memory_chat.py

The first session learns a fact; its typed memory is serialized to JSON; a fresh
session reloads that memory and continues the conversation with reset=False. The
continuation is streamed so the terminal FinalEvent's status is shown too.
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

    # First session: learn something, then serialize the run to JSON.
    first = agent.start()
    await first.run("Hi, my name is Sam.")
    saved = first.memory.dump_json()
    print(f"Saved {len(first.memory.steps)} steps to JSON.\n")

    # Later, in a fresh session (or another process), restore and continue.
    second = agent.start()
    second.memory = Memory.load_json(saved)

    answer, status = "", "unknown"
    async for event in second.run("What is my name?", reset=False, stream=True):
        if isinstance(event, FinalEvent):
            answer, status = event.answer, event.status
    print(f"[{status}] {answer}")


if __name__ == "__main__":
    asyncio.run(main())
