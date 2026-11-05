import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

import examples.cli_advanced_observer as observer
import examples.cli_dynamic_tools as dynamic_tools
import examples.cli_failure_recovery as failure_recovery
import examples.cli_math_tutor as math_tutor
import examples.cli_memory_chat as memory_chat
import examples.cli_notes_agent as notes_agent
import examples.cli_repo_assistant as repo_assistant
import examples.custom_model_generate_only as generate_only
import examples.providers.record as record
from agentling import (
    ActionStep,
    ChatMessage,
    Memory,
    TaskStep,
    ToolCall,
    ToolCallError,
    ToolResultEvent,
    Usage,
)
from agentling.testing import (
    ConformanceReport,
    ScriptedModel,
    check_model,
    read_fixture,
)
from examples.providers.anthropic_messages import AnthropicMessagesModel
from examples.providers.gemini_native import GeminiModel
from examples.providers.ollama_native import OllamaModel
from examples.providers.openai_chat import OpenAIChatModel


def _assistant(
    content: str = "", tool_calls: list[ToolCall] | None = None
) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls or [],
        usage=Usage(1, 1),
    )


def _call(call_id: str, name: str, /, **arguments: Any) -> ToolCall:
    # name is positional-only so a tool argument literally called "name"
    # (as in add_note) does not collide with this parameter.
    return ToolCall(id=call_id, name=name, arguments=arguments)


# Imports and shape
def test_offline_examples_expose_build_agent_and_main() -> None:
    for module in (dynamic_tools, failure_recovery, memory_chat, observer):
        assert callable(module.build_agent)
        assert callable(module.main)


# Failure recovery (offline)
async def test_failure_recovery_recovers_from_tool_errors() -> None:
    session = failure_recovery.build_agent().start()
    answer = await session.run("compute")

    assert "5" in answer
    action_steps = [s for s in session.memory.steps if isinstance(s, ActionStep)]
    assert action_steps
    errors = [r for s in action_steps for r in s.tool_results if r.is_error]
    assert errors  # the tool failure came back as an observation


# Dynamic tools (offline)
async def test_dynamic_tools_example_adds_and_removes() -> None:
    session = dynamic_tools.build_agent().start()
    first, second = await dynamic_tools.run_demo(session)

    assert first == "It's 18°C with light rain in Lagos."
    assert second == "The weather tool is gone, so I can't check again."
    results = [
        r
        for s in session.memory.steps
        if isinstance(s, ActionStep)
        for r in s.tool_results
    ]
    assert results[0].name == "get_weather"
    assert results[0].is_error is False
    assert results[1].name == "get_weather"
    assert results[1].error_kind == "unknown_tool"


# Memory replay (offline)
async def test_memory_replay_dumps_reloads_and_continues() -> None:
    agent = memory_chat.build_agent()

    first = agent.start()
    await first.run("Hi, my name is Sam.")
    saved = first.memory.dump_json()
    assert saved

    second = agent.start()
    second.memory = Memory.load_json(saved)
    assert second.memory.steps

    await second.run("What is my name?", reset=False)
    task_steps = [s for s in second.memory.steps if isinstance(s, TaskStep)]
    assert len(task_steps) > 1  # continuation built on the restored memory


# Event observer (offline)
async def test_observer_handles_every_event_type() -> None:
    counts = await observer.observe(observer.build_agent())

    for event_type in (
        "TextDelta",
        "ToolCallEvent",
        "ToolResultEvent",
        "StepEvent",
        "FinalEvent",
    ):
        assert counts.get(event_type, 0) >= 1


# API-backed examples (driven offline with an injected fake model)
def test_api_examples_expose_build_agent_and_main() -> None:
    for module in (math_tutor, repo_assistant, notes_agent):
        assert callable(module.build_agent)
        assert callable(module.main)


async def test_math_tutor_uses_its_tools() -> None:
    model = ScriptedModel(
        [
            _assistant(tool_calls=[_call("c1", "multiply", a=6, b=7)]),
            _assistant(tool_calls=[_call("c2", "add", a=42, b=3)]),
            _assistant(content="6 times 7 plus 3 is 45."),
        ]
    )

    answer = await math_tutor.build_agent(model=model).run("6*7+3?")
    assert "45" in answer


async def test_repo_assistant_reads_within_root_and_streams(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hi there", encoding="utf-8")
    model = ScriptedModel(
        [
            _assistant(tool_calls=[_call("c1", "read_file", path="hello.txt")]),
            _assistant(content="The file says hi there."),
        ]
    )
    agent = repo_assistant.build_agent(model=model, root=str(tmp_path))

    events = [event async for event in agent.run("read hello.txt", stream=True)]
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert results and results[0].result.content == "hi there"


def test_repo_assistant_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ToolCallError):
        repo_assistant._resolve_within(tmp_path, "../secret.txt")


def test_repo_assistant_sets_output_cap() -> None:
    agent = repo_assistant.build_agent(model=ScriptedModel([]))
    assert agent.max_tool_output_chars == 2000


async def test_notes_agent_adds_and_searches(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    model = ScriptedModel(
        [
            _assistant(
                tool_calls=[_call("c1", "add_note", name="todo", text="buy milk")]
            ),
            _assistant(tool_calls=[_call("c2", "search_notes", query="milk")]),
            _assistant(content="Found it in your notes."),
        ]
    )
    session = notes_agent.build_agent(model=model, notes_dir=str(notes_dir)).start()
    await session.run("add and search a note")

    assert (notes_dir / "todo.txt").exists()
    search_step = session.memory.steps[2]
    assert isinstance(search_step, ActionStep)
    assert "todo" in search_step.tool_results[0].content


def test_notes_agent_rejects_unsafe_paths(tmp_path: Path) -> None:
    with pytest.raises(ToolCallError):
        notes_agent._resolve_within(tmp_path, "../escape.txt")


def test_notes_agent_configures_timeout_and_redaction() -> None:
    agent = notes_agent.build_agent(model=ScriptedModel([]), notes_dir="unused")
    assert agent.tool_timeout == 10.0
    assert agent.redact_errors is True


# Reference adapters against their recorded fixtures
_FIXTURES = Path(__file__).resolve().parents[1] / "examples" / "providers" / "fixtures"


def _openai(transport: httpx.AsyncBaseTransport | None) -> OpenAIChatModel:
    return OpenAIChatModel("gpt-4o-mini", api_key="k", transport=transport)


async def test_openai_fixture_passes_check_model() -> None:
    report = await check_model(_openai, recorded=_FIXTURES / "openai_chat.jsonl")

    assert report.ok, str(report)
    assert {r.status for r in report.rules} == {"passed"}
    assert len(read_fixture(_FIXTURES / "openai_chat.jsonl")) == 3


async def test_lmstudio_fixture_synthesises_ids() -> None:
    fixture = _FIXTURES / "openai_chat_lmstudio.jsonl"

    report = await check_model(_openai, recorded=fixture)

    assert report.ok, str(report)
    tool_turn = report.turns[1]
    assert tool_turn is not None
    (call,) = tool_turn.tool_calls
    assert re.fullmatch(r"agentling_call_0_[0-9a-f]{8}", call.id)
    assert (call.name, call.arguments) == ("echo", {"text": "ping"})


def _anthropic(transport: httpx.AsyncBaseTransport | None) -> AnthropicMessagesModel:
    return AnthropicMessagesModel("claude-sonnet-5", api_key="k", transport=transport)


def _ollama(transport: httpx.AsyncBaseTransport | None) -> OllamaModel:
    return OllamaModel("qwen3:8b", transport=transport)


def _gemini(transport: httpx.AsyncBaseTransport | None) -> GeminiModel:
    return GeminiModel("gemini-2.5-flash", api_key="k", transport=transport)


@pytest.mark.parametrize(
    ("factory", "fixture", "error_status"),
    [
        (_anthropic, "anthropic_messages.jsonl", 529),
        (_ollama, "ollama_native.jsonl", 404),
        (_gemini, "gemini_native.jsonl", 429),
    ],
)
async def test_reference_adapter_fixture_passes_check_model(
    factory: Any, fixture: str, error_status: int
) -> None:
    report = await check_model(factory, recorded=_FIXTURES / fixture)

    assert report.ok, str(report)
    assert {r.status for r in report.rules} == {"passed"}
    assert read_fixture(_FIXTURES / fixture)[2].status == error_status


async def test_anthropic_fixture_keeps_both_thinking_blocks() -> None:
    report = await check_model(
        _anthropic, recorded=_FIXTURES / "anthropic_messages.jsonl"
    )

    tool_turn = report.turns[1]
    assert tool_turn is not None and tool_turn.provider_data is not None
    blocks = tool_turn.provider_data["anthropic"]["blocks"]
    assert [b["type"] for b in blocks] == ["thinking", "redacted_thinking"]


# record.py and the generate-only example
def _replies(*bodies: str) -> httpx.MockTransport:
    remaining = list(bodies)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=remaining.pop(0).encode())

    return httpx.MockTransport(handler)


def _sse_body(path: Path, line: int) -> str:
    return "\n".join(read_fixture(path)[line].body_lines)


async def test_record_writes_a_redacted_fixture(tmp_path: Path) -> None:
    source = _FIXTURES / "openai_chat.jsonl"
    out = tmp_path / "f.jsonl"

    report = await record.record(
        "openai_chat",
        model="m",
        base_url="http://test/v1",
        api_key="sk-test-123",
        out=out,
        inner=_replies(_sse_body(source, 0), _sse_body(source, 1)),
    )

    assert report.ok, str(report)
    exchanges = read_fixture(out)
    assert len(exchanges) == 2
    assert {e.request["headers"]["authorization"] for e in exchanges} == {"‹redacted›"}
    assert "sk-test-123" not in out.read_text(encoding="utf-8")
    replay = await check_model(_openai, recorded=out)
    assert replay.ok, str(replay)


def test_record_help_lists_the_adapters() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "examples.providers.record", "--help"],
        capture_output=True,
        text=True,
        cwd=_FIXTURES.parents[2],
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for word in (
        "openai_chat",
        "anthropic_messages",
        "ollama_native",
        "gemini_native",
        "--base-url",
        "--model",
        "--api-key-env",
        "--out",
    ):
        assert word in result.stdout


@pytest.mark.parametrize("passing", [False, True])
def test_record_main_exit_code(
    passing: bool, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    turns = _report_turns(passing)

    async def fake_record(*args: Any, **kwargs: Any) -> ConformanceReport:
        return await check_model(lambda _: ScriptedModel(turns), live=True)

    monkeypatch.setattr(record, "record", fake_record)

    code = record.main(["openai_chat", "--model", "m", "--out", "x.jsonl"])

    assert code == (0 if passing else 1)
    output = capsys.readouterr().out
    assert "rule 1 (text reassembles)" in output
    assert ("failed:" in output) is not passing


def _report_turns(passing: bool) -> list[ChatMessage]:
    text = ChatMessage("assistant", "pong", usage=Usage(1, 1), finish_reason="stop")
    tool_turn = ChatMessage(
        "assistant",
        tool_calls=[ToolCall("c1", "echo", {"text": "ping"})],
        usage=Usage(1, 1),
        finish_reason="tool_calls",
    )
    return [text, tool_turn if passing else text]


async def test_generate_only_example() -> None:
    assert await generate_only.build_agent().run("What is 2 + 3?") == "The answer is 5."

    report = await check_model(
        lambda _: generate_only.GenerateOnlyModel(generate_only.CannedBackend()),
        live=True,
    )

    assert report.ok, str(report)


@pytest.mark.parametrize(
    ("module", "last_line"),
    [
        (observer, "Event counts:"),
        (dynamic_tools, "[unknown_tool] get_weather"),
        (failure_recovery, "[ok] divide: 5.0"),
        (memory_chat, "[completed] Your name is Sam."),
    ],
)
async def test_offline_example_main_runs(
    module: Any, last_line: str, capsys: pytest.CaptureFixture[str]
) -> None:
    await module.main()

    assert last_line in capsys.readouterr().out.strip().splitlines()[-1]
