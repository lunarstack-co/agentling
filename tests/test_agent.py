import asyncio
import io
import re
import threading
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing
from pathlib import Path
from typing import Any

import pytest

import agentling.events
from agentling.agent import Agent, AgentSession, _truncate_middle
from agentling.errors import ModelError
from agentling.events import (
    Event,
    FinalEvent,
    StepEvent,
    TextDelta,
    ToolCallEvent,
    ToolResultEvent,
    print_events,
)
from agentling.memory import ActionStep, FinalStep, Memory, Step, TaskStep
from agentling.models import (
    ChatMessage,
    Delta,
    ToolCall,
    ToolCallDelta,
    ToolDefinition,
    Usage,
)
from agentling.skills import Skill
from agentling.testing import ScriptedModel
from agentling.tools import Tool, ToolCallError, tool


def _assistant(
    content: str = "", tool_calls: list[ToolCall] | None = None
) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls or [],
        usage=Usage(1, 1),
    )


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a * b


@tool
async def slow(seconds: float) -> str:
    """Sleep for a while, then return.

    Args:
        seconds: How long to sleep.
    """
    await asyncio.sleep(seconds)
    return "slept"


_REVIEWER_SKILL = Skill(
    name="reviewer",
    description="Review code for bugs and style issues.",
    instructions="Look for off-by-one errors and missing null checks.",
    path=Path("."),
    tools=[],
)

_CALC_SKILL = Skill(
    name="calc",
    description="Do arithmetic with the multiply tool.",
    instructions="Use the multiply tool to compute products.",
    path=Path("."),
    tools=[f"{__name__}:multiply"],
)


# --------------------------------------------------------------------------- #
# Termination paths
# --------------------------------------------------------------------------- #
async def test_forgiving_termination_returns_content() -> None:
    model = ScriptedModel([_assistant(content="42")])
    agent = Agent(model=model)
    session = agent.start()

    answer = await session.run("What is 6 times 7?")

    assert answer == "42"
    assert isinstance(session.memory.steps[0], TaskStep)
    assert isinstance(session.memory.steps[-1], FinalStep)


async def test_explicit_final_answer() -> None:
    model = ScriptedModel(
        [
            _assistant(
                tool_calls=[
                    ToolCall(id="c1", name="final_answer", arguments={"answer": "done"})
                ]
            )
        ]
    )
    agent = Agent(model=model)

    assert await agent.run("finish up") == "done"


# --------------------------------------------------------------------------- #
# Tool execution
# --------------------------------------------------------------------------- #
async def test_tool_call_then_answer() -> None:
    model = ScriptedModel(
        [
            _assistant(
                tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})]
            ),
            _assistant(content="The sum is 5."),
        ]
    )
    agent = Agent(model=model, tools=[add])
    session = agent.start()

    answer = await session.run("add 2 and 3")

    assert answer == "The sum is 5."
    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    assert action.tool_results[0].content == "5"
    assert action.tool_results[0].is_error is False


async def test_multi_step_tool_task() -> None:
    model = ScriptedModel(
        [
            _assistant(
                tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})]
            ),
            _assistant(
                tool_calls=[ToolCall(id="c2", name="add", arguments={"a": 5, "b": 4})]
            ),
            _assistant(
                tool_calls=[
                    ToolCall(id="c3", name="final_answer", arguments={"answer": "9"})
                ]
            ),
        ]
    )
    agent = Agent(model=model, tools=[add])

    assert await agent.run("compute a couple of sums") == "9"
    assert len(model.calls) == 3


async def test_tool_error_becomes_observation() -> None:
    @tool
    def boom() -> str:
        """Always fails."""
        raise RuntimeError("kaboom")

    model = ScriptedModel(
        [
            _assistant(tool_calls=[ToolCall(id="c1", name="boom", arguments={})]),
            _assistant(content="recovered"),
        ]
    )
    agent = Agent(model=model, tools=[boom])
    session = agent.start()

    assert await session.run("try boom") == "recovered"
    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    assert action.tool_results[0].is_error is True


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #
async def test_streaming_yields_events() -> None:
    model = ScriptedModel(
        [
            _assistant(
                tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1, "b": 1})]
            ),
            _assistant(content="2"),
        ]
    )
    agent = Agent(model=model, tools=[add])

    events = [event async for event in agent.run("add", stream=True)]

    assert any(isinstance(e, ToolCallEvent) for e in events)
    assert any(isinstance(e, ToolResultEvent) for e in events)
    assert any(isinstance(e, StepEvent) for e in events)
    assert isinstance(events[-1], FinalEvent)
    assert events[-1].answer == "2"


async def test_streaming_emits_text_deltas() -> None:
    model = ScriptedModel([_assistant(content="hello world")])
    agent = Agent(model=model)

    texts = [
        event.text
        async for event in agent.run("hi", stream=True)
        if isinstance(event, TextDelta)
    ]
    assert "".join(texts) == "hello world"


# --------------------------------------------------------------------------- #
# max_steps forced answer
# --------------------------------------------------------------------------- #
async def test_max_steps_forces_a_final_answer() -> None:
    # The model never ends the run itself; it always calls a tool.
    looping = [
        _assistant(
            tool_calls=[ToolCall(id=f"c{i}", name="add", arguments={"a": 1, "b": 1})]
        )
        for i in range(2)
    ]
    forced = _assistant(content="forced final")
    model = ScriptedModel([*looping, forced])
    agent = Agent(model=model, tools=[add], max_steps=2)

    assert await agent.run("loop forever") == "forced final"
    assert len(model.calls) == 3  # 2 loop steps + 1 forced tool-free answer


def _tool_turn(call_id: str, name: str, **arguments: Any) -> ChatMessage:
    """A model turn that requests a single tool call."""
    return _assistant(tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)])


# --------------------------------------------------------------------------- #
# Loop detector
# --------------------------------------------------------------------------- #
_REPEAT_MARKER = "already made this exact tool call"


async def test_loop_detector_nudges_on_repeat() -> None:
    model = ScriptedModel(
        [
            _tool_turn("c1", "add", a=2, b=3),
            _tool_turn("c2", "add", a=2, b=3),  # identical (name, args)
            _assistant(content="done"),
        ]
    )
    agent = Agent(model=model, tools=[add])
    session = agent.start()
    await session.run("repeat the same call")

    first, second = session.memory.steps[1], session.memory.steps[2]
    assert isinstance(first, ActionStep)
    assert isinstance(second, ActionStep)
    assert _REPEAT_MARKER not in first.tool_results[0].content
    assert _REPEAT_MARKER in second.tool_results[0].content


async def test_loop_detector_ignores_different_args() -> None:
    model = ScriptedModel(
        [
            _tool_turn("c1", "add", a=2, b=3),
            _tool_turn("c2", "add", a=4, b=5),  # different args -> not a loop
            _assistant(content="done"),
        ]
    )
    agent = Agent(model=model, tools=[add])
    session = agent.start()
    await session.run("two different sums")

    for step in session.memory.steps:
        if isinstance(step, ActionStep):
            assert _REPEAT_MARKER not in step.tool_results[0].content


# --------------------------------------------------------------------------- #
# Self-heal: unknown tool and always-raising tool
# --------------------------------------------------------------------------- #
async def test_unknown_tool_recovers() -> None:
    model = ScriptedModel([_tool_turn("c1", "ghost"), _assistant(content="recovered")])
    agent = Agent(model=model)  # only final_answer is registered
    session = agent.start()
    answer = await session.run("call a missing tool")

    assert answer == "recovered"
    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    assert action.tool_results[0].is_error is True
    assert "Unknown tool" in action.tool_results[0].content


async def test_always_raising_tool_recovers() -> None:
    @tool
    def boom() -> str:
        """Always raises."""
        raise RuntimeError("nope")

    model = ScriptedModel(
        [
            _tool_turn("c1", "boom"),
            _tool_turn("c2", "final_answer", answer="handled"),
        ]
    )
    agent = Agent(model=model, tools=[boom])
    session = agent.start()
    answer = await session.run("trigger boom")

    assert answer == "handled"
    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    assert action.tool_results[0].is_error is True
    assert "RuntimeError" in action.tool_results[0].content


# --------------------------------------------------------------------------- #
# Interruption + resume
# --------------------------------------------------------------------------- #
def _interrupt_after_first_step(session: AgentSession) -> None:
    """Make `session` interrupt itself once, after its first recorded step."""

    fired: list[bool] = []

    def interrupt_once(step: object) -> None:
        if not fired:
            fired.append(True)
            session.interrupt()

    session.step_callbacks.append(interrupt_once)


def _roles(memory: Memory) -> Sequence[str]:
    """Message roles in memory, without the system prompt."""
    return [message.role for message in memory.to_messages("system")][1:]


def _task_steps(session: AgentSession) -> int:
    return sum(isinstance(step, TaskStep) for step in session.memory.steps)


async def test_interrupt_stops_run_gracefully() -> None:
    model = ScriptedModel(
        [_tool_turn("c1", "add", a=1, b=1), _assistant(content="unreached")]
    )
    session = Agent(model=model, tools=[add]).start()
    _interrupt_after_first_step(session)

    assert await session.run("loop") == "Run interrupted."
    assert isinstance(session.memory.steps[-1], ActionStep)  # paused before a FinalStep
    assert len(model.calls) == 1


async def test_interrupt_while_idle_has_no_effect() -> None:
    model = ScriptedModel([_assistant(content="ok")])
    session = Agent(model=model).start()

    session.interrupt()

    assert await session.run("q") == "ok"
    assert _roles(session.memory) == ["user", "assistant"]
    assert len(model.calls) == 1


async def test_interrupt_after_a_finished_run_is_dropped() -> None:
    model = ScriptedModel([_assistant(content="a"), _assistant(content="b")])
    session = Agent(model=model).start()
    await session.run("first")

    session.interrupt()
    final = [event async for event in session.run("second", stream=True)][-1]

    assert isinstance(final, FinalEvent)
    assert (final.answer, final.status) == ("b", "completed")


def test_interrupt_docstring_says_idle_calls_do_nothing() -> None:
    assert "Has no effect when no run is active." in (
        AgentSession.interrupt.__doc__ or ""
    )


def test_the_interrupt_is_cleared_in_one_place() -> None:
    agent_py = Path(__file__).resolve().parents[1] / "src" / "agentling" / "agent.py"
    assert agent_py.read_text(encoding="utf-8").count("_interrupt.clear()") == 1


async def test_interrupt_then_resume_completes() -> None:
    model = ScriptedModel(
        [
            _tool_turn("c1", "add", a=1, b=1),
            _assistant(content="finished on resume"),
        ]
    )
    session = Agent(model=model, tools=[add]).start()
    _interrupt_after_first_step(session)

    assert await session.run("start") == "Run interrupted."
    before = list(session.memory.steps)

    assert await session.resume() == "finished on resume"
    assert [message.role for message in model.calls[1]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert _task_steps(session) == 1
    assert session.memory.steps[:2] == before


async def test_run_with_reset_false_still_adds_the_task() -> None:
    model = ScriptedModel(
        [_tool_turn("c1", "add", a=1, b=1), _assistant(content="finished")]
    )
    session = Agent(model=model, tools=[add]).start()
    _interrupt_after_first_step(session)

    assert await session.run("start") == "Run interrupted."
    assert await session.run("continue", reset=False) == "finished"
    assert _task_steps(session) == 2


async def test_resume_can_stream() -> None:
    model = ScriptedModel(
        [_tool_turn("c1", "add", a=1, b=1), _assistant(content="finished")]
    )
    session = Agent(model=model, tools=[add]).start()
    _interrupt_after_first_step(session)
    await session.run("start")

    events = [event async for event in session.resume(stream=True)]

    final = events[-1]
    assert isinstance(final, FinalEvent)
    assert (final.answer, final.status) == ("finished", "completed")
    assert _task_steps(session) == 1


async def test_resume_max_steps_counts_from_zero() -> None:
    model = ScriptedModel(
        [
            _tool_turn("c1", "add", a=1, b=1),
            _tool_turn("c2", "add", a=2, b=2),
            _assistant(content="forced"),
        ]
    )
    session = Agent(model=model, tools=[add]).start()
    _interrupt_after_first_step(session)
    await session.run("start")

    final = [event async for event in session.resume(stream=True, max_steps=1)][-1]

    assert isinstance(final, FinalEvent)
    assert (final.answer, final.status) == ("forced", "max_steps")
    assert sum(isinstance(step, ActionStep) for step in session.memory.steps) == 2


def test_resume_on_a_fresh_session_raises() -> None:
    session = Agent(model=ScriptedModel([])).start()

    # Raised by the call itself, before any coroutine or stream exists.
    with pytest.raises(ValueError, match="^Nothing to resume$"):
        _ = session.resume()
    with pytest.raises(ValueError, match="^Nothing to resume$"):
        session.resume(stream=True)


async def test_resume_after_a_finished_run_raises() -> None:
    session = Agent(model=ScriptedModel([_assistant(content="done")])).start()
    await session.run("q")

    with pytest.raises(ValueError, match="^Nothing to resume$"):
        await session.resume()


# --------------------------------------------------------------------------- #
# One run at a time per session
# --------------------------------------------------------------------------- #
_ALREADY_RUNNING = r"^Session is already running\. .*contextlib\.aclosing\.$"


async def test_is_running_tracks_the_run() -> None:
    model = ScriptedModel([_tool_turn("c1", "add", a=1, b=1), _assistant(content="ok")])
    session = Agent(model=model, tools=[add]).start()
    seen: list[bool] = []
    session.step_callbacks.append(lambda step: seen.append(session.is_running))

    assert session.is_running is False
    await session.run("go")

    assert seen == [True]
    assert session.is_running is False


async def test_is_running_resets_when_a_stream_is_closed_early() -> None:
    model = ScriptedModel([_assistant(content="a"), _assistant(content="b")])
    session = Agent(model=model).start()

    async with aclosing(session.run("first", stream=True)) as events:
        await anext(events)
        assert session.is_running is True
    assert session.is_running is False

    stream = session.run("second", stream=True)
    await anext(stream)
    assert session.is_running is True
    await stream.aclose()
    assert session.is_running is False


async def test_an_unstarted_stream_does_not_hold_the_session() -> None:
    session = Agent(model=ScriptedModel([_assistant(content="b")])).start()

    unstarted = session.run("a", stream=True)

    assert session.is_running is False
    assert await session.run("b") == "b"
    await unstarted.aclose()


async def test_a_second_run_while_running_is_refused() -> None:
    model = ScriptedModel([_assistant(content="first answer")])
    session = Agent(model=model).start()

    first = session.run("a", stream=True)
    assert isinstance(await anext(first), TextDelta)

    with pytest.raises(RuntimeError, match=_ALREADY_RUNNING):
        await session.run("b")

    final = [event async for event in first][-1]
    assert isinstance(final, FinalEvent)
    assert final.answer == "first answer"
    assert len(model.calls) == 1


async def test_a_second_stream_is_refused_on_its_first_step() -> None:
    model = ScriptedModel([_assistant(content="first answer")])
    session = Agent(model=model).start()
    first = session.run("a", stream=True)
    await anext(first)

    second = session.run("b", stream=True)
    with pytest.raises(RuntimeError, match=_ALREADY_RUNNING):
        await anext(second)

    final = [event async for event in first][-1]
    assert isinstance(final, FinalEvent)
    assert final.answer == "first answer"


async def test_resume_while_running_is_refused() -> None:
    model = ScriptedModel(
        [_tool_turn("c1", "add", a=1, b=1), _assistant(content="finished")]
    )
    session = Agent(model=model, tools=[add]).start()
    _interrupt_after_first_step(session)
    await session.run("start")

    async with aclosing(session.resume(stream=True)) as resumed:
        await anext(resumed)
        with pytest.raises(RuntimeError, match=_ALREADY_RUNNING):
            await session.resume()


async def test_running_again_from_a_callback_is_refused() -> None:
    model = ScriptedModel(
        [_tool_turn("c1", "add", a=1, b=1), _assistant(content="unreached")]
    )
    session = Agent(model=model, tools=[add]).start()

    async def run_again(step: Step) -> None:
        await session.run("again")

    session.step_callbacks.append(run_again)

    with pytest.raises(RuntimeError, match=_ALREADY_RUNNING):
        await session.run("go")
    assert session.is_running is False
    assert isinstance(session.memory.steps[-1], ActionStep)


async def test_run_and_resume_can_be_scheduled_as_tasks() -> None:
    # mypy checks this file, so these create_task calls also pin that the
    # non-streaming run() and resume() are typed as coroutines.
    model = ScriptedModel(
        [_tool_turn("c1", "add", a=1, b=1), _assistant(content="finished")]
    )
    session = Agent(model=model, tools=[add]).start()
    _interrupt_after_first_step(session)

    first: asyncio.Task[str] = asyncio.create_task(session.run("go"))
    assert await first == "Run interrupted."
    resumed: asyncio.Task[str] = asyncio.create_task(session.resume())
    assert await resumed == "finished"

    agent = Agent(model=ScriptedModel([_assistant(content="x")]))
    one_shot: asyncio.Task[str] = asyncio.create_task(agent.run("q"))
    assert await one_shot == "x"


async def test_sessions_run_one_after_another_and_agents_run_concurrently() -> None:
    session = Agent(
        model=ScriptedModel([_assistant(content="a"), _assistant(content="b")])
    ).start()
    assert await session.run("1") == "a"
    assert await session.run("2") == "b"

    agent = Agent(
        model=ScriptedModel([_assistant(content="x"), _assistant(content="y")])
    )
    assert sorted(await asyncio.gather(agent.run("1"), agent.run("2"))) == ["x", "y"]


# --------------------------------------------------------------------------- #
# final_answer ends the run without tool events
# --------------------------------------------------------------------------- #
def _event_names(events: list[Event]) -> list[str]:
    return [type(event).__name__ for event in events]


async def test_final_answer_emits_no_tool_events() -> None:
    model = ScriptedModel([_tool_turn("c1", "final_answer", answer="x")])
    session = Agent(model=model).start()

    events = [event async for event in session.run("go", stream=True)]

    assert _event_names(events) == ["StepEvent", "FinalEvent"]
    final = events[-1]
    assert isinstance(final, FinalEvent)
    assert final.answer == "x"
    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    assert action.model_message.tool_calls[0].name == "final_answer"
    assert action.tool_results[0].content == "x"


async def test_a_tool_and_final_answer_in_one_step() -> None:
    model = ScriptedModel(
        [
            _assistant(
                tool_calls=[
                    ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2}),
                    ToolCall(id="c2", name="final_answer", arguments={"answer": "3"}),
                ]
            )
        ]
    )
    session = Agent(model=model, tools=[add]).start()

    events = [event async for event in session.run("go", stream=True)]

    assert _event_names(events) == [
        "ToolCallEvent",
        "ToolResultEvent",
        "StepEvent",
        "FinalEvent",
    ]
    call, result = events[0], events[1]
    assert isinstance(call, ToolCallEvent) and call.tool_call.name == "add"
    assert isinstance(result, ToolResultEvent) and result.result.name == "add"
    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    assert len(action.tool_results) == 2


async def test_text_then_final_answer() -> None:
    model = ScriptedModel(
        [
            _assistant(
                content="thinking",
                tool_calls=[
                    ToolCall(id="c1", name="final_answer", arguments={"answer": "x"})
                ],
            )
        ]
    )

    events = [event async for event in Agent(model=model).run("go", stream=True)]

    assert _event_names(events) == ["TextDelta", "StepEvent", "FinalEvent"]


async def test_final_answer_with_bad_arguments_is_recorded_not_shown() -> None:
    model = ScriptedModel(
        [_tool_turn("c1", "final_answer", wrong=1), _assistant(content="recovered")]
    )
    session = Agent(model=model).start()

    events = [event async for event in session.run("go", stream=True)]

    assert _event_names(events) == ["StepEvent", "TextDelta", "FinalEvent"]
    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    assert action.tool_results[0].is_error is True


def test_event_docstring_explains_final_answer() -> None:
    assert "final_answer" in (agentling.events.__doc__ or "")


# --------------------------------------------------------------------------- #
# Step callbacks
# --------------------------------------------------------------------------- #
async def test_async_step_callback_runs_before_the_step_event() -> None:
    log: list[str] = []

    async def record(step: Step) -> None:
        await asyncio.sleep(0)
        log.append("callback")

    model = ScriptedModel([_tool_turn("c1", "add", a=1, b=1), _assistant(content="ok")])
    session = Agent(model=model, tools=[add], step_callbacks=[record]).start()

    async for event in session.run("go", stream=True):
        if isinstance(event, StepEvent):
            log.append("step")

    assert log == ["callback", "step"]
    assert session.step_callbacks == [record]


async def test_step_callbacks_run_in_registration_order() -> None:
    log: list[str] = []

    def first(step: Step) -> None:
        log.append("first")

    async def second(step: Step) -> None:
        await asyncio.sleep(0)
        log.append("second")

    def third(step: Step) -> None:
        log.append("third")

    model = ScriptedModel([_tool_turn("c1", "add", a=1, b=1), _assistant(content="ok")])
    session = Agent(model=model, tools=[add], step_callbacks=[first, second]).start()
    session.step_callbacks.append(third)

    await session.run("go")

    assert log == ["first", "second", "third"]


async def test_a_raising_callback_on_the_final_answer_step_keeps_the_answer() -> None:
    def boom(step: Step) -> None:
        raise ValueError("boom")

    model = ScriptedModel([_tool_turn("c1", "final_answer", answer="42")])
    session = Agent(model=model, step_callbacks=[boom]).start()

    with pytest.raises(ValueError, match="boom"):
        await session.run("go")

    assert [type(step) for step in session.memory.steps] == [
        TaskStep,
        ActionStep,
        FinalStep,
    ]
    assert session.memory.steps[-1] == FinalStep(answer="42")
    with pytest.raises(ValueError, match="^Nothing to resume$"):
        await session.resume()
    assert len(model.calls) == 1


async def test_a_raising_async_callback_stops_the_run_after_recording() -> None:
    async def boom(step: Step) -> None:
        raise ValueError("boom")

    model = ScriptedModel(
        [_tool_turn("c1", "add", a=1, b=1), _assistant(content="unreached")]
    )
    session = Agent(model=model, tools=[add], step_callbacks=[boom]).start()
    events: list[Event] = []

    with pytest.raises(ValueError, match="boom"):
        async for event in session.run("go", stream=True):
            events.append(event)

    assert isinstance(session.memory.steps[-1], ActionStep)
    assert not any(isinstance(event, StepEvent) for event in events)
    assert len(model.calls) == 1
    assert session.is_running is False


# --------------------------------------------------------------------------- #
# Skills: progressive disclosure
# --------------------------------------------------------------------------- #
def _load_skill_turn(call_id: str, skill_name: str) -> ChatMessage:
    """A model turn calling load_skill (whose own argument is named 'name')."""
    return _assistant(
        tool_calls=[
            ToolCall(id=call_id, name="load_skill", arguments={"name": skill_name})
        ]
    )


def test_skill_catalog_is_added_to_the_system_prompt() -> None:
    agent = Agent(model=ScriptedModel([]), skills=[_REVIEWER_SKILL])
    session = agent.start()

    assert "load_skill" in session.tools
    assert "reviewer" in agent.instructions
    assert "Review code for bugs and style issues." in agent.instructions
    # Only the name and description surface up front; the body stays hidden
    # until the skill is loaded.
    assert "off-by-one" not in agent.instructions


def test_no_skills_means_no_load_skill_tool() -> None:
    agent = Agent(model=ScriptedModel([]))

    assert "load_skill" not in agent.start().tools


async def test_load_skill_reveals_the_body() -> None:
    model = ScriptedModel(
        [
            _load_skill_turn("c1", "reviewer"),
            _assistant(content="done"),
        ]
    )
    agent = Agent(model=model, skills=[_REVIEWER_SKILL])
    session = agent.start()

    assert await session.run("review this") == "done"
    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    assert action.tool_results[0].content == _REVIEWER_SKILL.instructions
    assert action.tool_results[0].is_error is False


async def test_load_skill_registers_the_skills_tools() -> None:
    model = ScriptedModel(
        [
            _load_skill_turn("c1", "calc"),
            _tool_turn("c2", "multiply", a=6, b=7),
            _assistant(content="42"),
        ]
    )
    agent = Agent(model=model, skills=[_CALC_SKILL])
    session = agent.start()

    # multiply is hidden until the skill that provides it is loaded.
    assert "multiply" not in session.tools

    assert await session.run("compute six times seven") == "42"

    assert "multiply" in session.tools
    load = session.memory.steps[1]
    assert isinstance(load, ActionStep)
    assert "Tools now available: multiply." in load.tool_results[0].content
    product = session.memory.steps[2]
    assert isinstance(product, ActionStep)
    assert product.tool_results[0].content == "42"


async def test_load_unknown_skill_is_an_error_observation() -> None:
    model = ScriptedModel(
        [
            _load_skill_turn("c1", "ghost"),
            _assistant(content="recovered"),
        ]
    )
    agent = Agent(model=model, skills=[_REVIEWER_SKILL])
    session = agent.start()

    assert await session.run("load a missing skill") == "recovered"
    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    assert action.tool_results[0].is_error is True
    assert "Unknown skill" in action.tool_results[0].content


def test_skill_can_be_loaded_from_a_path(tmp_path: Path) -> None:
    folder = tmp_path / "greeter"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        "---\nname: greeter\ndescription: Greet warmly.\n---\nSay hi.\n",
        encoding="utf-8",
    )
    agent = Agent(model=ScriptedModel([]), skills=[folder])

    assert "greeter" in agent.skills
    assert "greeter" in agent.instructions


# --------------------------------------------------------------------------- #
# Construction guards
# --------------------------------------------------------------------------- #
def test_duplicate_tool_name_raises() -> None:
    with pytest.raises(ValueError, match="Duplicate tool name"):
        Agent(model=ScriptedModel([]), tools=[add, add])


def test_max_steps_below_one_raises() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        Agent(model=ScriptedModel([]), max_steps=0)


async def test_run_max_steps_below_one_raises() -> None:
    agent = Agent(model=ScriptedModel([_assistant(content="unused")]))

    with pytest.raises(ValueError, match="at least 1"):
        await agent.run("hi", max_steps=0)


# --------------------------------------------------------------------------- #
# Session isolation
# --------------------------------------------------------------------------- #
async def test_sessions_keep_separate_memory() -> None:
    agent = Agent(model=ScriptedModel([_assistant(content="one")]))
    s1 = agent.start()
    s2 = agent.start()

    await s1.run("first")

    assert s1.memory is not s2.memory
    assert len(s1.memory.steps) > 0
    assert s2.memory.steps == []  # a sibling session is untouched


async def test_skill_tools_do_not_leak_between_sessions() -> None:
    model = ScriptedModel(
        [
            _load_skill_turn("c1", "calc"),
            _tool_turn("c2", "multiply", a=2, b=2),
            _assistant(content="4"),
        ]
    )
    agent = Agent(model=model, skills=[_CALC_SKILL])
    loader = agent.start()
    other = agent.start()

    await loader.run("use calc")

    assert "multiply" in loader.tools  # registered by load_skill in this session
    assert "multiply" not in other.tools  # never leaks into a sibling session


async def test_interrupt_affects_only_its_session() -> None:
    model = ScriptedModel(
        [_tool_turn("c1", "add", a=1, b=1), _assistant(content="s2 done")]
    )
    agent = Agent(model=model, tools=[add])
    s1 = agent.start()
    s2 = agent.start()
    _interrupt_after_first_step(s1)

    assert await s1.run("stop") == "Run interrupted."
    # s1's interrupt flag is still set; s2 has its own and runs to completion.
    assert await s2.run("go") == "s2 done"


# --------------------------------------------------------------------------- #
# Malformed model output (recoverable)
# --------------------------------------------------------------------------- #
class _MalformedThenAnswerModel:
    """Streams one unparseable tool call, then a plain answer on the retry."""

    def __init__(self) -> None:
        self.calls = 0

    async def stream(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolDefinition] = ()
    ) -> AsyncIterator[Delta]:
        self.calls += 1
        if self.calls == 1:
            yield Delta(
                tool_calls=[
                    ToolCallDelta(index=0, id="c1", name="add", arguments="{bad json")
                ]
            )
        else:
            yield Delta(content="recovered")
        yield Delta(usage=Usage(1, 1))


class _AlwaysMalformedModel:
    """Every turn streams an unparseable tool call."""

    async def stream(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolDefinition] = ()
    ) -> AsyncIterator[Delta]:
        yield Delta(
            tool_calls=[
                ToolCallDelta(index=0, id="c1", name="add", arguments="{bad json")
            ]
        )
        yield Delta(usage=Usage(1, 1))


async def test_malformed_tool_call_is_recoverable() -> None:
    model = _MalformedThenAnswerModel()
    agent = Agent(model=model)

    assert await agent.run("do something") == "recovered"
    assert model.calls == 2  # first turn malformed, re-prompted, then answered


async def test_persistently_malformed_output_raises_model_error() -> None:
    agent = Agent(model=_AlwaysMalformedModel())

    with pytest.raises(ModelError):
        await agent.run("do something")


# --------------------------------------------------------------------------- #
# Timeouts and cancellation
# --------------------------------------------------------------------------- #
async def test_tool_timeout_becomes_observation() -> None:
    model = ScriptedModel(
        [_tool_turn("c1", "slow", seconds=5), _assistant(content="done")]
    )
    agent = Agent(model=model, tools=[slow], tool_timeout=0.01)
    session = agent.start()

    assert await session.run("call slow") == "done"
    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    assert action.tool_results[0].is_error is True
    assert "ToolTimeoutError" in action.tool_results[0].content


async def test_parallel_timeout_keeps_other_results() -> None:
    model = ScriptedModel(
        [
            _assistant(
                tool_calls=[
                    ToolCall(id="c1", name="slow", arguments={"seconds": 5}),
                    ToolCall(id="c2", name="add", arguments={"a": 2, "b": 3}),
                ]
            ),
            _assistant(content="both handled"),
        ]
    )
    agent = Agent(model=model, tools=[slow, add], tool_timeout=0.01)
    session = agent.start()

    assert await session.run("two tools") == "both handled"
    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    by_name = {result.name: result for result in action.tool_results}
    assert by_name["slow"].is_error is True
    assert by_name["add"].is_error is False
    assert by_name["add"].content == "5"


class _SlowStreamModel:
    """A model whose stream stalls before producing anything."""

    async def stream(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolDefinition] = ()
    ) -> AsyncIterator[Delta]:
        await asyncio.sleep(5)
        yield Delta(content="too late")


async def test_model_stream_timeout_raises_model_error() -> None:
    agent = Agent(model=_SlowStreamModel(), model_timeout=0.01, retry=None)

    with pytest.raises(ModelError):
        await agent.run("hi")


async def test_cancellation_propagates_during_a_tool() -> None:
    model = ScriptedModel(
        [_tool_turn("c1", "slow", seconds=5), _assistant(content="done")]
    )
    agent = Agent(model=model, tools=[slow])  # no timeout: relies on cancellation

    async def run_it() -> str:
        return await agent.run("call slow")

    task = asyncio.ensure_future(run_it())
    await asyncio.sleep(0.02)  # let the run reach the slow tool

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


class _InterruptMidStreamModel:
    """Interrupts its own session partway through streaming."""

    def __init__(self) -> None:
        self.session: AgentSession | None = None

    async def stream(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolDefinition] = ()
    ) -> AsyncIterator[Delta]:
        yield Delta(content="par")
        assert self.session is not None
        self.session.interrupt()
        yield Delta(content="tial")
        yield Delta(usage=Usage(1, 1))


async def test_interrupt_during_stream_stops_the_run() -> None:
    model = _InterruptMidStreamModel()
    agent = Agent(model=model)
    session = agent.start()
    model.session = session

    assert await session.run("stream") == "Run interrupted."


# --------------------------------------------------------------------------- #
# Tool execution hardening
# --------------------------------------------------------------------------- #
async def test_sync_tools_run_off_the_event_loop_thread() -> None:
    @tool
    def where() -> int:
        """Return the id of the thread the tool ran on."""
        return threading.get_ident()

    model = ScriptedModel([_tool_turn("c1", "where"), _assistant(content="ok")])
    agent = Agent(model=model, tools=[where])
    session = agent.start()

    await session.run("which thread")
    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    # The sync tool ran in a worker thread, not the event loop's thread.
    assert action.tool_results[0].content != str(threading.get_ident())


async def test_tool_output_is_truncated() -> None:
    @tool
    def big() -> str:
        """Return a large string."""
        return "x" * 1000

    model = ScriptedModel([_tool_turn("c1", "big"), _assistant(content="ok")])
    agent = Agent(model=model, tools=[big], max_tool_output_chars=100)
    session = agent.start()

    await session.run("go")
    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    content = action.tool_results[0].content
    assert "characters omitted" in content
    assert len(content) < 300


async def test_errors_are_shown_by_default() -> None:
    @tool
    def leak() -> str:
        """Raise with a message."""
        raise RuntimeError("visible detail")

    model = ScriptedModel([_tool_turn("c1", "leak"), _assistant(content="ok")])
    agent = Agent(model=model, tools=[leak])
    session = agent.start()

    await session.run("go")
    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    assert "visible detail" in action.tool_results[0].content


async def test_error_redaction_hides_the_message() -> None:
    @tool
    def leak() -> str:
        """Raise with a secret."""
        raise RuntimeError("token=SECRET123")

    model = ScriptedModel([_tool_turn("c1", "leak"), _assistant(content="ok")])
    agent = Agent(model=model, tools=[leak], redact_errors=True)
    session = agent.start()

    await session.run("go")
    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    content = action.tool_results[0].content
    assert "SECRET123" not in content  # message suppressed
    assert "RuntimeError" in content  # the type is still surfaced


async def test_tool_call_error_shown_even_when_redacting() -> None:
    @tool
    def bad() -> str:
        """Raise a ToolCallError."""
        raise ToolCallError("use a positive number")

    model = ScriptedModel([_tool_turn("c1", "bad"), _assistant(content="ok")])
    agent = Agent(model=model, tools=[bad], redact_errors=True)
    session = agent.start()

    await session.run("go")
    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    assert "use a positive number" in action.tool_results[0].content


async def test_non_parallel_safe_tools_run_in_order() -> None:
    order: list[str] = []

    @tool(parallel_safe=False)
    async def first() -> str:
        """First tool."""
        order.append("first-start")
        await asyncio.sleep(0.02)
        order.append("first-end")
        return "1"

    @tool
    async def second() -> str:
        """Second tool."""
        order.append("second")
        return "2"

    model = ScriptedModel(
        [
            _assistant(
                tool_calls=[
                    ToolCall(id="c1", name="first", arguments={}),
                    ToolCall(id="c2", name="second", arguments={}),
                ]
            ),
            _assistant(content="done"),
        ]
    )
    agent = Agent(model=model, tools=[first, second])
    session = agent.start()

    await session.run("go")
    # first is not parallel_safe, so the step runs sequentially: it finishes
    # before second starts.
    assert order == ["first-start", "first-end", "second"]


# --------------------------------------------------------------------------- #
# Memory hardening: cumulative usage and the context-window hook
# --------------------------------------------------------------------------- #
async def test_final_event_reports_cumulative_usage() -> None:
    model = ScriptedModel([_tool_turn("c1", "add", a=1, b=1), _assistant(content="2")])
    agent = Agent(model=model, tools=[add])

    events = [event async for event in agent.run("go", stream=True)]
    final = events[-1]
    assert isinstance(final, FinalEvent)
    # Each ScriptedModel turn reports Usage(1, 1); two turns sum to Usage(2, 2).
    assert final.usage == Usage(2, 2)


async def test_context_manager_trims_the_prompt() -> None:
    def keep_only_system(messages: list[ChatMessage]) -> list[ChatMessage]:
        return messages[:1]

    model = ScriptedModel([_assistant(content="ok")])
    agent = Agent(model=model, context_manager=keep_only_system)

    assert await agent.run("hello") == "ok"
    # The model saw only the trimmed message list.
    assert len(model.calls[0]) == 1


# --------------------------------------------------------------------------- #
# Terminal run status and print_events
# --------------------------------------------------------------------------- #
async def test_completed_run_reports_status() -> None:
    model = ScriptedModel([_assistant(content="done")])
    agent = Agent(model=model)

    events = [event async for event in agent.run("go", stream=True)]
    final = events[-1]
    assert isinstance(final, FinalEvent)
    assert final.status == "completed"


async def test_interrupted_run_reports_status() -> None:
    model = ScriptedModel([_tool_turn("c1", "add", a=1, b=1), _assistant(content="x")])
    agent = Agent(model=model, tools=[add])
    session = agent.start()

    fired: list[bool] = []

    def stop(step: object) -> None:
        if not fired:
            fired.append(True)
            session.interrupt()

    session.step_callbacks.append(stop)

    events = [event async for event in session.run("loop", stream=True)]
    final = events[-1]
    assert isinstance(final, FinalEvent)
    assert final.status == "interrupted"


async def test_max_steps_run_reports_status() -> None:
    looping = [_tool_turn(f"c{i}", "add", a=1, b=1) for i in range(2)]
    model = ScriptedModel([*looping, _assistant(content="forced")])
    agent = Agent(model=model, tools=[add], max_steps=2)

    events = [event async for event in agent.run("loop", stream=True)]
    final = events[-1]
    assert isinstance(final, FinalEvent)
    assert final.answer == "forced"
    assert final.status == "max_steps"


async def test_print_events_writes_to_file_and_returns_answer() -> None:
    model = ScriptedModel([_tool_turn("c1", "add", a=2, b=3), _assistant(content="5")])
    agent = Agent(model=model, tools=[add])

    buffer = io.StringIO()
    answer = await print_events(agent.run("add", stream=True), file=buffer)

    assert answer == "5"
    output = buffer.getvalue()
    assert "add" in output  # the tool call was rendered
    assert "5" in output  # the final answer was rendered


# --------------------------------------------------------------------------- #
# Edge cases: reserved load_skill name and tiny truncation limits
# --------------------------------------------------------------------------- #
def test_load_skill_name_is_reserved_when_skills_are_configured() -> None:
    @tool
    def load_skill(name: str) -> str:
        """A tool that collides with the built-in skill loader.

        Args:
            name: Unused.
        """
        return name

    with pytest.raises(ValueError, match="reserved"):
        Agent(model=ScriptedModel([]), tools=[load_skill], skills=[_REVIEWER_SKILL])


def test_truncate_middle_handles_tiny_limits() -> None:
    assert _truncate_middle("abc", 10) == "abc"  # under the limit, unchanged
    assert _truncate_middle("abcdef", 0) == ""
    assert _truncate_middle("abcdef", 1) == "a"

    windowed = _truncate_middle("x" * 100, 20)
    assert "omitted" in windowed
    assert len(windowed) < 100


# --------------------------------------------------------------------------- #
# Dynamic tools: add_tools, remove_tools and registration checks
# --------------------------------------------------------------------------- #
class _SchemaTool(Tool):
    """A tool built from a raw schema that answers with a fixed label."""

    def __init__(self, name: str, label: str = "", parameters: Any = None) -> None:
        self.name = name
        self.description = f"The {name} tool."
        self.parameters = parameters if parameters is not None else {"type": "object"}
        self._label = label or name

    async def forward(self, **kwargs: Any) -> Any:
        return self._label


def _tool_names(definitions: list[ToolDefinition]) -> list[str]:
    return [definition.name for definition in definitions]


def _results(session: AgentSession) -> list[Any]:
    return [
        result
        for step in session.memory.steps
        if isinstance(step, ActionStep)
        for result in step.tool_results
    ]


def test_agent_rejects_invalid_subclass_tool() -> None:
    with pytest.raises(ValueError, match="'issues.create' is not a valid tool name"):
        Agent(model=ScriptedModel([]), tools=[_SchemaTool("issues.create")])


def test_agent_rejects_non_object_schema_tool() -> None:
    with pytest.raises(ValueError, match='JSON Schema object with "type": "object"'):
        Agent(
            model=ScriptedModel([]),
            tools=[_SchemaTool("t", parameters={"type": "string"})],
        )


async def test_add_tools_exposes_tool_on_next_call() -> None:
    model = ScriptedModel([_tool_turn("c1", "a"), _assistant(content="done")])
    session = Agent(model=model).start()

    session.add_tools([_SchemaTool("a", label="from a")])

    assert await session.run("go") == "done"
    assert _tool_names(model.tools[0]) == ["final_answer", "a"]
    assert _results(session)[0].content == "from a"


async def test_add_tools_mid_run_visible_next_step() -> None:
    model = ScriptedModel(
        [_tool_turn("c1", "b"), _tool_turn("c2", "a"), _assistant(content="done")]
    )
    agent = Agent(model=model, tools=[_SchemaTool("b")])
    session = agent.start()
    added: list[bool] = []

    def add_after_first_step(step: Any) -> None:
        if not added:
            session.add_tools([_SchemaTool("a")])
            added.append(True)

    session.step_callbacks.append(add_after_first_step)

    assert await session.run("go") == "done"
    assert "a" not in _tool_names(model.tools[0])
    assert "a" in _tool_names(model.tools[1])
    assert _results(session)[1].is_error is False


def test_add_tools_duplicate_in_batch_is_atomic() -> None:
    session = Agent(model=ScriptedModel([])).start()
    before = list(session._tool_definitions)

    with pytest.raises(ValueError, match="Duplicate tool name: 'a'"):
        session.add_tools([_SchemaTool("a"), _SchemaTool("a")])

    assert "a" not in session.tools
    assert session._tool_definitions == before


def test_add_tools_existing_name_without_replace() -> None:
    session = Agent(model=ScriptedModel([])).start()
    original = _SchemaTool("a")
    session.add_tools([original])

    with pytest.raises(ValueError, match="Duplicate tool name: 'a'"):
        session.add_tools([_SchemaTool("a")])

    assert session.tools["a"] is original


async def test_add_tools_replace_keeps_position() -> None:
    model = ScriptedModel([_tool_turn("c1", "a"), _assistant(content="done")])
    session = Agent(model=model, tools=[_SchemaTool("x")]).start()
    session.add_tools([_SchemaTool("a", label="v1"), _SchemaTool("y")])
    position = _tool_names(session._tool_definitions).index("a")

    replacement = _SchemaTool("a", label="v2")
    session.add_tools([replacement], replace=True)

    assert _tool_names(session._tool_definitions).index("a") == position
    assert session.tools["a"] is replacement
    await session.run("go")
    assert _results(session)[0].content == "v2"


def test_add_tools_invalid_tool_is_atomic() -> None:
    session = Agent(model=ScriptedModel([])).start()

    with pytest.raises(ValueError, match="not a valid tool name"):
        session.add_tools([_SchemaTool("good"), _SchemaTool("bad.name")])

    assert "good" not in session.tools


def test_add_tools_accepts_generator_and_empty() -> None:
    session = Agent(model=ScriptedModel([])).start()

    session.add_tools(t for t in [_SchemaTool("a")])
    session.add_tools([])

    assert list(session.tools).count("a") == 1
    assert session._tool_definitions == [t.definition for t in session.tools.values()]


def test_add_tools_is_session_local() -> None:
    agent = Agent(model=ScriptedModel([]))
    s1 = agent.start()
    s2 = agent.start()

    s1.add_tools([_SchemaTool("a")])

    assert "a" in s1.tools
    assert "a" not in s2.tools
    assert "a" not in agent.start().tools


async def test_remove_tools_removes_tool_and_schema() -> None:
    model = ScriptedModel([_assistant(content="done")])
    session = Agent(model=model).start()
    session.add_tools([_SchemaTool("a"), _SchemaTool("b")])

    session.remove_tools(["a"])

    assert "a" not in session.tools
    await session.run("go")
    names = _tool_names(model.tools[0])
    assert "a" not in names
    assert "b" in names


def test_remove_tools_unknown_is_atomic() -> None:
    session = Agent(model=ScriptedModel([])).start()
    session.add_tools([_SchemaTool("a")])

    with pytest.raises(ValueError, match="Unknown tool: 'missing'"):
        session.remove_tools(["a", "missing"])

    assert "a" in session.tools


def test_remove_tools_dedupes_and_empty() -> None:
    session = Agent(model=ScriptedModel([])).start()
    session.add_tools([_SchemaTool("a")])

    session.remove_tools(["a", "a"])
    session.remove_tools([])

    assert "a" not in session.tools


def test_reserved_names_cannot_be_removed() -> None:
    session = Agent(model=ScriptedModel([]), skills=[_REVIEWER_SKILL]).start()

    with pytest.raises(ValueError, match="'final_answer' is a reserved tool name."):
        session.remove_tools(["final_answer"])
    with pytest.raises(ValueError, match="'load_skill' is a reserved tool name."):
        session.remove_tools(["load_skill"])

    assert {"final_answer", "load_skill"} <= set(session.tools)


def test_reserved_names_cannot_be_replaced() -> None:
    session = Agent(model=ScriptedModel([]), skills=[_REVIEWER_SKILL]).start()
    final_answer = session.tools["final_answer"]
    load_skill = session.tools["load_skill"]

    for name in ("final_answer", "load_skill"):
        with pytest.raises(ValueError, match=f"'{name}' is a reserved tool name."):
            session.add_tools([_SchemaTool(name)], replace=True)

    assert session.tools["final_answer"] is final_answer
    assert session.tools["load_skill"] is load_skill


def test_load_skill_not_reserved_without_skills() -> None:
    session = Agent(model=ScriptedModel([])).start()

    session.add_tools([_SchemaTool("load_skill")])

    assert "load_skill" in session.tools


async def test_removed_tool_call_returns_unknown_tool() -> None:
    model = ScriptedModel([_tool_turn("c1", "a"), _assistant(content="recovered")])
    session = Agent(model=model).start()
    session.add_tools([_SchemaTool("a")])
    session.remove_tools(["a"])

    events = [event async for event in session.run("go", stream=True)]

    result = _results(session)[0]
    assert result.is_error is True
    assert result.error_kind == "unknown_tool"
    assert result.content.startswith("Unknown tool 'a'")
    final = events[-1]
    assert isinstance(final, FinalEvent)
    assert final.status == "completed"
    assert final.answer == "recovered"


async def test_removed_mid_run_tool_returns_unknown_tool() -> None:
    model = ScriptedModel(
        [_tool_turn("c1", "a"), _tool_turn("c2", "a"), _assistant(content="done")]
    )
    session = Agent(model=model).start()
    session.add_tools([_SchemaTool("a")])

    def remove_after_first_step(step: Any) -> None:
        if "a" in session.tools:
            session.remove_tools(["a"])

    session.step_callbacks.append(remove_after_first_step)

    assert await session.run("go") == "done"
    first, second = _results(session)
    assert first.is_error is False
    assert second.error_kind == "unknown_tool"


async def test_load_skill_twice_is_idempotent() -> None:
    model = ScriptedModel(
        [
            _load_skill_turn("c1", "calc"),
            _load_skill_turn("c2", "calc"),
            _assistant(content="done"),
        ]
    )
    session = Agent(model=model, skills=[_CALC_SKILL]).start()

    assert await session.run("load twice") == "done"
    assert all(result.is_error is False for result in _results(session))
    assert _tool_names(model.tools[2]).count("multiply") == 1


async def test_load_skill_conflict_is_an_observation() -> None:
    @tool
    def multiply(a: int, b: int) -> int:
        """A different tool that shares the skill tool's name.

        Args:
            a: The first number.
            b: The second number.
        """
        return 0

    model = ScriptedModel([_load_skill_turn("c1", "calc"), _assistant(content="done")])
    session = Agent(model=model, tools=[multiply], skills=[_CALC_SKILL]).start()

    assert await session.run("load calc") == "done"
    result = _results(session)[0]
    assert result.is_error is True
    assert result.error_kind == "validation"
    assert (
        "Skill 'calc' declares tool 'multiply', which conflicts with an existing tool."
        in result.content
    )
    assert session.tools["multiply"] is multiply


async def test_memory_with_removed_tool_restores_and_runs() -> None:
    first_model = ScriptedModel([_tool_turn("c1", "b"), _assistant(content="first")])
    first = Agent(model=first_model, tools=[_SchemaTool("b")]).start()
    await first.run("use b")
    saved = first.memory.dump_json()

    second_model = ScriptedModel([_assistant(content="second")])
    second = Agent(model=second_model).start()
    second.memory = Memory.load_json(saved)

    rendered = second.memory.to_messages("instructions")
    assert any(call.name == "b" for message in rendered for call in message.tool_calls)
    assert await second.run("again", reset=False) == "second"


def test_no_new_runtime_dependencies() -> None:
    import tomllib

    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    names = sorted(
        dep.split(">")[0].split("=")[0].split("<")[0].strip()
        for dep in project["project"]["dependencies"]
    )
    assert names == ["pyyaml"]

    _assert_no_vendor_imports(root / "src" / "agentling", _SRC_VENDOR_IMPORT)


def test_tests_import_no_vendor_sdk() -> None:
    _assert_no_vendor_imports(Path(__file__).resolve().parent, _TEST_VENDOR_IMPORT)


def test_vendor_import_guard_catches_every_import_form(tmp_path: Path) -> None:
    for line in ("import openai", "from openai import X", "import anthropic as a"):
        (tmp_path / "offender.py").write_text(line + "\n", encoding="utf-8")
        with pytest.raises(AssertionError, match="offender.py:1"):
            _assert_no_vendor_imports(tmp_path, _TEST_VENDOR_IMPORT)


_SRC_VENDOR_IMPORT = re.compile(r"^\s*(?:from|import)\s+(?:openai|anthropic|mcp)\b")
_TEST_VENDOR_IMPORT = re.compile(r"^\s*(?:from|import)\s+(?:openai|anthropic)\b")


def _assert_no_vendor_imports(folder: Path, pattern: re.Pattern[str]) -> None:
    for source in folder.rglob("*.py"):
        lines = source.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, 1):
            assert not pattern.match(line), f"{source}:{number}: {line}"
