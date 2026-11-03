"""An adapter for a backend that returns whole replies and can't stream.

Runs offline:

    uv run python -m examples.custom_model_generate_only

The Model protocol only has stream(), so the adapter wraps each reply with
deltas_from_message. CannedBackend stands in for the real backend; replace its
generate() with your HTTP call.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Sequence

from agentling import (
    Agent,
    ChatMessage,
    Delta,
    ToolCall,
    ToolDefinition,
    Usage,
    deltas_from_message,
    tool,
)


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b


class CannedBackend:
    """Answers from a few fixed rules instead of a model."""

    async def generate(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolDefinition]
    ) -> ChatMessage:
        last = messages[-1]
        names = {t.name for t in tools}
        if last.role == "tool":
            return _reply(messages, content=f"The answer is {last.content}.")
        numbers = [int(n) for n in re.findall(r"-?\d+", last.content)]
        if "add" in names and len(numbers) >= 2:
            call = ToolCall("call_1", "add", {"a": numbers[0], "b": numbers[1]})
            return _reply(messages, tool_calls=[call])
        echo = re.search(r"text set to (\w+)", last.content)
        if "echo" in names and echo:
            call = ToolCall("call_1", "echo", {"text": echo.group(1)})
            return _reply(messages, tool_calls=[call])
        word = re.search(r"exactly the word: (\w+)", last.content)
        return _reply(messages, content=word.group(1) if word else "I can only add.")


def _reply(
    messages: Sequence[ChatMessage],
    *,
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
) -> ChatMessage:
    prompt_words = sum(len(m.content.split()) for m in messages)
    return ChatMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls or [],
        usage=Usage(input_tokens=prompt_words, output_tokens=len(content.split()) + 1),
        finish_reason="tool_calls" if tool_calls else "stop",
    )


class GenerateOnlyModel:
    """The adapter: implements stream() on top of a whole-reply backend."""

    def __init__(self, backend: CannedBackend) -> None:
        self.backend = backend

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition] = (),
    ) -> AsyncIterator[Delta]:
        reply = await self.backend.generate(messages, tools)
        async for delta in deltas_from_message(reply):
            yield delta


def build_agent() -> Agent:
    """Build an agent on the generate-only model with one tool."""

    return Agent(model=GenerateOnlyModel(CannedBackend()), tools=[add])


async def main() -> None:
    answer = await build_agent().run("What is 2 + 3?")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
