import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from agentling.agent import Agent
from agentling.errors import MemoryLoadError
from agentling.memory import (
    ActionStep,
    FinalStep,
    Memory,
    TaskStep,
    ToolResult,
)
from agentling.models import ChatMessage, Delta, ToolCall, ToolDefinition, Usage


def _action_step() -> ActionStep:
    """A representative ActionStep: one tool call, one successful result."""
    return ActionStep(
        model_message=ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="c1", name="get_weather", arguments={"city": "Paris"})
            ],
            usage=Usage(10, 5),
        ),
        tool_results=[
            ToolResult(tool_call_id="c1", name="get_weather", content="Sunny")
        ],
        usage=Usage(10, 5),
        duration=1.2,
    )


# --------------------------------------------------------------------------- #
# Per-step rendering
# --------------------------------------------------------------------------- #
def test_task_step_renders_user_message() -> None:
    msgs = TaskStep(task="hi").to_messages()
    assert [m.role for m in msgs] == ["user"]
    assert msgs[0].content == "hi"


def test_action_step_renders_assistant_then_tool() -> None:
    msgs = _action_step().to_messages()
    assert [m.role for m in msgs] == ["assistant", "tool"]
    assert msgs[1].content == "Sunny"
    assert msgs[1].tool_call_id == "c1"


def test_action_step_renders_error_observation() -> None:
    step = ActionStep(
        model_message=ChatMessage(
            role="assistant",
            tool_calls=[ToolCall(id="c1", name="f", arguments={})],
        ),
        tool_results=[
            ToolResult(
                tool_call_id="c1",
                name="f",
                content="boom",
                is_error=True,
                error_kind="validation",
            )
        ],
    )
    tool_msg = step.to_messages()[1]
    assert tool_msg.content == "Error from 'f': boom. Fix the arguments and try again."


def test_action_step_execution_error_uses_a_different_hint() -> None:
    step = ActionStep(
        model_message=ChatMessage(role="assistant"),
        tool_results=[
            ToolResult(
                tool_call_id="c1",
                name="f",
                content="network down",
                is_error=True,
                error_kind="execution",
            )
        ],
    )
    tool_msg = step.to_messages()[1]
    assert "different approach" in tool_msg.content
    assert "Fix the arguments" not in tool_msg.content


def test_final_step_renders_assistant_message() -> None:
    msgs = FinalStep(answer="done").to_messages()
    assert [m.role for m in msgs] == ["assistant"]
    assert msgs[0].content == "done"


# --------------------------------------------------------------------------- #
# Memory.to_messages
# --------------------------------------------------------------------------- #
def test_to_messages_full_shape() -> None:
    mem = Memory()
    mem.add(TaskStep(task="weather?"))
    mem.add(_action_step())
    mem.add(FinalStep(answer="Sunny."))

    msgs = mem.to_messages("You are helpful.")
    assert [m.role for m in msgs] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert msgs[0].content == "You are helpful."


def test_to_messages_empty_is_just_the_system_prompt() -> None:
    msgs = Memory().to_messages("sys")
    assert [m.role for m in msgs] == ["system"]


# --------------------------------------------------------------------------- #
# add / reset (multi-turn support)
# --------------------------------------------------------------------------- #
def test_add_and_reset() -> None:
    mem = Memory()
    mem.add(TaskStep(task="a"))
    mem.add(TaskStep(task="b"))
    assert len(mem.steps) == 2

    mem.reset()
    assert mem.steps == []


# --------------------------------------------------------------------------- #
# JSON round-trip (persistence / replay)
# --------------------------------------------------------------------------- #
def test_json_round_trip_preserves_all_steps() -> None:
    mem = Memory()
    mem.add(TaskStep(task="weather?"))
    mem.add(_action_step())
    mem.add(FinalStep(answer="Sunny."))

    restored = Memory.load_json(mem.dump_json())
    assert restored.steps == mem.steps


def test_dump_json_tags_each_step_with_its_kind() -> None:
    mem = Memory()
    mem.add(TaskStep(task="hi"))
    mem.add(_action_step())
    mem.add(FinalStep(answer="bye"))

    data = json.loads(mem.dump_json())
    assert [entry["type"] for entry in data["steps"]] == ["task", "action", "final"]


def test_round_trip_preserves_error_flag() -> None:
    mem = Memory()
    mem.add(
        ActionStep(
            model_message=ChatMessage(
                role="assistant",
                tool_calls=[ToolCall(id="c1", name="f", arguments={})],
            ),
            tool_results=[
                ToolResult(tool_call_id="c1", name="f", content="boom", is_error=True)
            ],
        )
    )

    action = Memory.load_json(mem.dump_json()).steps[0]
    assert isinstance(action, ActionStep)
    assert action.tool_results[0].is_error is True


def test_from_dict_rejects_unknown_step_type() -> None:
    with pytest.raises(MemoryLoadError, match="Unknown step type"):
        Memory.from_dict({"version": 1, "steps": [{"type": "bogus", "data": {}}]})


def test_from_dict_rejects_unsupported_version() -> None:
    with pytest.raises(MemoryLoadError, match="version"):
        Memory.from_dict({"version": 999, "steps": []})


def test_from_dict_rejects_non_list_steps() -> None:
    with pytest.raises(MemoryLoadError, match="steps"):
        Memory.from_dict({"version": 1, "steps": "not a list"})


# --------------------------------------------------------------------------- #
# Format v2 and loading older saves
# --------------------------------------------------------------------------- #
_FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def memory_v1() -> dict[str, Any]:
    """A memory saved by agentling 0.1.1, the format Aldwin has on disk."""
    loaded: dict[str, Any] = json.loads(
        (_FIXTURES / "memory_v1.json").read_text(encoding="utf-8")
    )
    return loaded


def test_round_trip_keeps_the_new_fields() -> None:
    memory = Memory(
        steps=[
            ActionStep(
                model_message=ChatMessage(
                    role="assistant",
                    tool_calls=[ToolCall(id="c1", name="f", arguments={})],
                    usage=Usage(1, 2, 3, 4),
                    reasoning="r",
                    finish_reason="tool_calls",
                    provider_data={"x": {"sig": "abc"}},
                ),
                usage=Usage(1, 2, 3, 4),
            )
        ]
    )

    raw = memory.dump_json()

    assert json.loads(raw)["version"] == 2
    assert Memory.load_json(raw) == memory


def test_v1_fixture_loads_with_new_fields_at_defaults(
    memory_v1: dict[str, Any],
) -> None:
    memory = Memory.from_dict(memory_v1)

    task, action, final = memory.steps
    assert task == TaskStep(task="What's the weather in Paris?")
    assert isinstance(action, ActionStep)
    assert action.model_message.tool_calls[0].id == "call_1"
    assert action.model_message.reasoning is None
    assert action.model_message.finish_reason is None
    assert action.model_message.provider_data is None
    assert action.usage == Usage(12, 7)
    assert final == FinalStep(answer="It is sunny in Paris.")


async def test_v1_session_resumes_and_saves_as_v2(memory_v1: dict[str, Any]) -> None:
    class _TextModel:
        async def stream(
            self, messages: Sequence[ChatMessage], tools: Sequence[ToolDefinition] = ()
        ) -> AsyncIterator[Delta]:
            yield Delta(content="Still sunny.")

    session = Agent(model=_TextModel()).start()
    session.memory = Memory.from_dict(memory_v1)

    assert await session.run("And now?", reset=False) == "Still sunny."

    saved = json.loads(session.memory.dump_json())
    assert saved["version"] == 2
    assert [step["type"] for step in saved["steps"]] == [
        "task",
        "action",
        "final",
        "task",
        "final",
    ]


def test_unknown_keys_are_ignored() -> None:
    memory = Memory.from_dict(
        {
            "version": 2,
            "steps": [{"type": "task", "data": {"task": "hi", "future_field": 1}}],
        }
    )
    assert memory.steps == [TaskStep(task="hi")]


def test_missing_required_field_raises() -> None:
    message: dict[str, Any] = {
        "content": "",
        "tool_calls": [],
        "tool_call_id": None,
        "usage": None,
    }
    with pytest.raises(MemoryLoadError):
        Memory.from_dict(
            {
                "version": 2,
                "steps": [{"type": "action", "data": {"model_message": message}}],
            }
        )


@pytest.mark.parametrize(
    "payload",
    [{"version": 3}, {}, {"version": "2"}, {"version": True}, {"version": 1.0}],
)
def test_bad_versions_raise(payload: dict[str, Any]) -> None:
    with pytest.raises(MemoryLoadError, match="version"):
        Memory.from_dict({**payload, "steps": []})


def test_malformed_step_data_raises_memory_load_error() -> None:
    with pytest.raises(MemoryLoadError):
        Memory.from_dict({"version": 2, "steps": [{"type": "task", "data": "x"}]})
