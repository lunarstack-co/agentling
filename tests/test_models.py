import dataclasses
import re
import typing
from pathlib import Path

import pytest

from agentling.errors import ModelOutputError
from agentling.models import (
    ChatMessage,
    Delta,
    FinishReason,
    Model,
    ToolCall,
    ToolCallDelta,
    ToolDefinition,
    Usage,
    agglomerate_deltas,
    deltas_from_message,
    parse_tool_arguments,
)

_SYNTHESISED_ID = re.compile(r"^agentling_call_(\d+)_[0-9a-f]{8}$")


# --------------------------------------------------------------------------- #
# Neutral types
# --------------------------------------------------------------------------- #
def test_usage_total_tokens() -> None:
    assert Usage(input_tokens=3, output_tokens=4).total_tokens == 7


def test_usage_adds_every_field() -> None:
    assert Usage(1, 2, 3, 4) + Usage(10, 20, 30, 40) == Usage(11, 22, 33, 44)
    assert Usage(5, 7).total_tokens == 12


def test_usage_breakdowns_default_to_zero() -> None:
    usage = Usage(1, 2)
    assert usage.cached_input_tokens == 0
    assert usage.reasoning_tokens == 0


def test_tool_definition_is_frozen() -> None:
    definition = ToolDefinition("a", "d", {"type": "object"})

    assert (definition.name, definition.description) == ("a", "d")
    assert definition.parameters == {"type": "object"}
    with pytest.raises(dataclasses.FrozenInstanceError):
        definition.name = "b"  # type: ignore[misc]


@pytest.mark.parametrize("value", [Delta(), ChatMessage(role="assistant")])
def test_new_fields_default_to_none(value: Delta | ChatMessage) -> None:
    assert value.reasoning is None
    assert value.finish_reason is None
    assert value.provider_data is None


def test_finish_reason_values() -> None:
    assert typing.get_args(FinishReason) == (
        "stop",
        "tool_calls",
        "length",
        "content_filter",
        "error",
        "unknown",
    )


@pytest.mark.parametrize(
    "rule",
    [
        "Yield no `Delta` until",
        "`usage` at least once",
        "`finish_reason` exactly once",
        "only `ModelError`",
        "Never mutate",
        "system message",
        "`provider_data` namespace",
    ],
)
def test_model_docstring_states_the_adapter_contract(rule: str) -> None:
    assert Model.__doc__ is not None
    assert rule in " ".join(Model.__doc__.split())


# --------------------------------------------------------------------------- #
# parse_tool_arguments
# --------------------------------------------------------------------------- #
def test_parse_tool_arguments_valid() -> None:
    assert parse_tool_arguments('{"a": 1}') == {"a": 1}


def test_parse_tool_arguments_empty_defaults_to_object() -> None:
    assert parse_tool_arguments("") == {}


def test_parse_tool_arguments_invalid_json_raises() -> None:
    with pytest.raises(ModelOutputError, match="Invalid tool call arguments"):
        parse_tool_arguments("{bad")


def test_parse_tool_arguments_non_object_raises() -> None:
    with pytest.raises(ModelOutputError, match="must be a JSON object"):
        parse_tool_arguments("[1, 2]")


# --------------------------------------------------------------------------- #
# agglomerate_deltas: reassemble a stream into one ChatMessage
# --------------------------------------------------------------------------- #
def test_agglomerate_concatenates_content() -> None:
    msg = agglomerate_deltas([Delta(content="Hel"), Delta(content="lo"), Delta()])
    assert msg.role == "assistant"
    assert msg.content == "Hello"
    assert msg.tool_calls == []


def test_agglomerate_merges_tool_call_fragments() -> None:
    deltas = [
        Delta(
            tool_calls=[ToolCallDelta(index=0, id="c1", name="wx", arguments='{"ci')]
        ),
        Delta(tool_calls=[ToolCallDelta(index=0, arguments='ty": "Paris"}')]),
    ]
    msg = agglomerate_deltas(deltas)
    assert msg.tool_calls == [ToolCall(id="c1", name="wx", arguments={"city": "Paris"})]


def test_agglomerate_groups_parallel_calls_by_index() -> None:
    deltas = [
        Delta(tool_calls=[ToolCallDelta(index=0, id="a", name="f", arguments="{}")]),
        Delta(tool_calls=[ToolCallDelta(index=1, id="b", name="g", arguments="{}")]),
    ]
    msg = agglomerate_deltas(deltas)
    assert [tc.id for tc in msg.tool_calls] == ["a", "b"]
    assert [tc.name for tc in msg.tool_calls] == ["f", "g"]


def test_agglomerate_captures_usage() -> None:
    msg = agglomerate_deltas([Delta(content="x"), Delta(usage=Usage(5, 2))])
    assert msg.usage == Usage(5, 2)


def test_agglomerate_rejects_invalid_tool_json() -> None:
    deltas = [
        Delta(tool_calls=[ToolCallDelta(index=0, id="c1", name="f", arguments="{bad")])
    ]
    with pytest.raises(ModelOutputError, match="Invalid tool call arguments"):
        agglomerate_deltas(deltas)


def test_agglomerate_rejects_non_object_tool_args() -> None:
    deltas = [
        Delta(
            tool_calls=[ToolCallDelta(index=0, id="c1", name="f", arguments="[1, 2]")]
        )
    ]
    with pytest.raises(ModelOutputError, match="must be a JSON object"):
        agglomerate_deltas(deltas)


def test_agglomerate_rejects_tool_call_without_name() -> None:
    deltas = [Delta(tool_calls=[ToolCallDelta(index=0, id="c1", arguments="{}")])]
    with pytest.raises(ModelOutputError, match="missing a name"):
        agglomerate_deltas(deltas)


def test_agglomerate_synthesises_a_missing_id() -> None:
    msg = agglomerate_deltas([Delta(tool_calls=[ToolCallDelta(0, None, "f", "{}")])])

    assert len(msg.tool_calls) == 1
    match = _SYNTHESISED_ID.match(msg.tool_calls[0].id)
    assert match is not None
    assert match.group(1) == "0"


def test_agglomerate_synthesises_distinct_ids_for_empty_ids() -> None:
    msg = agglomerate_deltas(
        [
            Delta(tool_calls=[ToolCallDelta(0, "", "f", "{}")]),
            Delta(tool_calls=[ToolCallDelta(1, "", "g", "{}")]),
        ]
    )

    first, second = (call.id for call in msg.tool_calls)
    assert first != second
    assert first.startswith("agentling_call_0_")
    assert second.startswith("agentling_call_1_")
    assert all(_SYNTHESISED_ID.match(call.id) for call in msg.tool_calls)


def test_agglomerate_gives_one_id_per_index_across_fragments() -> None:
    msg = agglomerate_deltas(
        [
            Delta(tool_calls=[ToolCallDelta(0, name="f", arguments='{"a')]),
            Delta(tool_calls=[ToolCallDelta(0, arguments='":')]),
            Delta(tool_calls=[ToolCallDelta(0, arguments="1}")]),
        ]
    )

    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].arguments == {"a": 1}
    assert _SYNTHESISED_ID.match(msg.tool_calls[0].id)


def test_agglomerate_keeps_a_real_id() -> None:
    msg = agglomerate_deltas(
        [Delta(tool_calls=[ToolCallDelta(0, "call_x", "f", "{}")])]
    )
    assert msg.tool_calls[0].id == "call_x"


def test_agglomerate_keeps_a_real_id_and_name_over_later_empty_ones() -> None:
    msg = agglomerate_deltas(
        [
            Delta(tool_calls=[ToolCallDelta(0, "call_real", "f", '{"a')]),
            Delta(tool_calls=[ToolCallDelta(0, "", "", '": 1}')]),
        ]
    )
    assert msg.tool_calls == [ToolCall(id="call_real", name="f", arguments={"a": 1})]


def test_agglomerate_concatenates_reasoning() -> None:
    msg = agglomerate_deltas([Delta(reasoning="a"), Delta(reasoning="b")])
    assert msg.reasoning == "ab"


def test_agglomerate_leaves_reasoning_none_when_absent() -> None:
    assert agglomerate_deltas([Delta(content="x")]).reasoning is None


def test_agglomerate_last_finish_reason_wins() -> None:
    msg = agglomerate_deltas(
        [Delta(), Delta(finish_reason="length"), Delta(finish_reason=None)]
    )
    assert msg.finish_reason == "length"


def test_agglomerate_provider_data_last_wins_unmerged() -> None:
    msg = agglomerate_deltas(
        [
            Delta(provider_data={"a": 1}),
            Delta(provider_data={"b": 2}),
            Delta(provider_data=None),
        ]
    )
    assert msg.provider_data == {"b": 2}


# --------------------------------------------------------------------------- #
# deltas_from_message
# --------------------------------------------------------------------------- #
async def test_deltas_from_message_round_trips() -> None:
    message = ChatMessage(
        role="assistant",
        content="thinking done",
        tool_calls=[
            ToolCall(id="c1", name="f", arguments={"x": 1}),
            ToolCall(id="c2", name="g", arguments={}),
        ],
        usage=Usage(3, 4),
        reasoning="because",
        finish_reason="tool_calls",
        provider_data={"vendor": {"sig": "s"}},
    )

    deltas = [delta async for delta in deltas_from_message(message)]

    assert len(deltas) == 1
    assert agglomerate_deltas(deltas) == message


# --------------------------------------------------------------------------- #
# Non-functional budgets
# --------------------------------------------------------------------------- #
def test_models_module_stays_small() -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "agentling" / "models.py"
    assert len(source.read_text(encoding="utf-8").splitlines()) <= 250
