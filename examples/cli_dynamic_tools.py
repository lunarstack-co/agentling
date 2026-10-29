"""Dynamic-tools demo: add a tool built from a JSON Schema, then remove it.

Runs fully offline with a scripted model, so no API key is needed:

    uv run python examples/cli_dynamic_tools.py

The session gains the tool mid-conversation, the model uses it, then the tool
is removed and the model recovers from the unknown-tool observation.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from agentling import (
    ActionStep,
    Agent,
    AgentSession,
    ChatMessage,
    Delta,
    Tool,
    ToolCall,
    ToolCallDelta,
    Usage,
)

WEATHER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "city": {"type": "string", "description": "The city to look up."},
        "units": {"type": ["string", "null"], "enum": ["c", "f", None]},
    },
    "required": ["city"],
}


class JsonSchemaTool(Tool):
    """A tool defined by a name, a description and a JSON Schema."""

    validate_arguments = False

    def __init__(self, name: str, description: str, schema: dict[str, Any]) -> None:
        self.name = name
        self.description = description
        self.parameters = schema

    async def forward(self, **kwargs: Any) -> Any:
        units = kwargs.get("units") or "c"
        temperature = 18 if units == "c" else 64
        return f"{kwargs['city']}: {temperature}°{units.upper()}, light rain"


class ScriptedModel:
    """A deterministic offline model that replays fixed assistant turns."""

    def __init__(self, turns: Sequence[ChatMessage]) -> None:
        self._turns = list(turns)
        self._index = 0

    async def generate(
        self, messages: Sequence[ChatMessage], tools: Sequence[Any] | None = None
    ) -> ChatMessage:
        turn = self._turns[self._index]
        self._index += 1
        return turn

    async def stream(
        self, messages: Sequence[ChatMessage], tools: Sequence[Any] | None = None
    ) -> AsyncIterator[Delta]:
        turn = self._turns[self._index]
        self._index += 1
        if turn.content:
            yield Delta(content=turn.content)
        for index, call in enumerate(turn.tool_calls):
            yield Delta(
                tool_calls=[
                    ToolCallDelta(
                        index=index,
                        id=call.id,
                        name=call.name,
                        arguments=json.dumps(call.arguments),
                    )
                ]
            )
        yield Delta(usage=turn.usage)


def _assistant(
    content: str = "", tool_calls: list[ToolCall] | None = None
) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls or [],
        usage=Usage(1, 1),
    )


def _weather_call(call_id: str) -> ChatMessage:
    arguments = {"city": "Lagos", "units": None}
    return _assistant(
        tool_calls=[ToolCall(id=call_id, name="get_weather", arguments=arguments)]
    )


def build_agent() -> Agent:
    """Build an offline agent scripted for two turns: tool present, then gone."""

    model = ScriptedModel(
        [
            _weather_call("c1"),
            _assistant(content="It's 18°C with light rain in Lagos."),
            _weather_call("c2"),
            _assistant(content="The weather tool is gone, so I can't check again."),
        ]
    )
    return Agent(model=model)


def weather_tool() -> JsonSchemaTool:
    return JsonSchemaTool(
        name="get_weather",
        description="Get the current weather for a city.",
        schema=WEATHER_SCHEMA,
    )


async def run_demo(session: AgentSession) -> tuple[str, str]:
    """Add the tool, ask once, remove it, ask again. Returns both answers."""

    session.add_tools([weather_tool()])
    first = await session.run("What's the weather in Lagos?")

    session.remove_tools(["get_weather"])
    second = await session.run("And now?", reset=False)
    return first, second


async def main() -> None:
    session = build_agent().start()
    first, second = await run_demo(session)

    print(f"With the tool:    {first}")
    print(f"Without the tool: {second}\n")
    print("Tool results:")
    for step in session.memory.steps:
        if isinstance(step, ActionStep):
            for result in step.tool_results:
                flag = result.error_kind or "ok"
                print(f"  [{flag}] {result.name}: {result.content}")


if __name__ == "__main__":
    asyncio.run(main())
