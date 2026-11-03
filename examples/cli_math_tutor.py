"""A small agent with two tools: a math tutor.

Needs OPENAI_API_KEY. AGENTLING_EXAMPLE_MODEL overrides the model name
(default gpt-4o-mini):

    uv run python -m examples.cli_math_tutor

build_agent() takes an optional model so tests run it offline.
"""

from __future__ import annotations

import asyncio
import os

from agentling import Agent, Model, tool
from examples.providers.openai_chat import OpenAIChatModel


@tool
def add(a: float, b: float) -> float:
    """Add two numbers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a * b


def build_agent(model: Model | None = None) -> Agent:
    """Build the tutor agent, defaulting to an OpenAI-compatible model."""

    return Agent(
        model=model
        or OpenAIChatModel(os.environ.get("AGENTLING_EXAMPLE_MODEL", "gpt-4o-mini")),
        tools=[add, multiply],
        instructions=(
            "You are a patient math tutor. Use the tools to compute, then "
            "explain the result in one sentence."
        ),
    )


async def main() -> None:
    answer = await build_agent().run("What is 6 times 7, then plus 3?")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
