"""agentling.testing: ScriptedModel, check_model and the fixture recorder."""

import ast
import json
import subprocess
import sys
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path

import httpx
import pytest

from agentling import (
    Agent,
    ChatMessage,
    Delta,
    ModelAuthError,
    ModelRateLimitError,
    ToolCall,
    ToolCallDelta,
    ToolDefinition,
    Usage,
    agglomerate_deltas,
)
from agentling.testing import (
    ConformanceReport,
    Exchange,
    RecordingTransport,
    ScriptedModel,
    ScriptedTurn,
    check_model,
    read_fixture,
    write_fixture,
)
from examples.providers.openai_chat import OpenAIChatModel

REPO = Path(__file__).resolve().parents[1]


async def _collect(model: ScriptedModel) -> list[Delta]:
    return [delta async for delta in model.stream([ChatMessage("user", "q")])]


# ScriptedModel
async def test_scripted_model_answers_with_a_chat_message() -> None:
    model = ScriptedModel([ChatMessage(role="assistant", content="hi")])

    assert await Agent(model=model).run("q") == "hi"
    assert len(model.calls) == 1
    assert [m.role for m in model.calls[0]] == ["system", "user"]


async def test_scripted_model_error_then_text_succeeds_after_one_retry() -> None:
    model = ScriptedModel(
        [
            ModelRateLimitError(retry_after=0),
            ChatMessage(role="assistant", content="ok"),
        ]
    )

    assert await Agent(model=model).run("q") == "ok"
    assert len(model.calls) == 2


async def test_chat_message_turn_streams_in_pieces() -> None:
    message = ChatMessage(
        role="assistant",
        content="a",
        reasoning="r",
        tool_calls=[ToolCall("c1", "add", {"a": 1}), ToolCall("c2", "mul", {"b": 2})],
        usage=Usage(3, 4),
        finish_reason="tool_calls",
        provider_data={"x": {"k": 1}},
    )

    deltas = await _collect(ScriptedModel([message]))

    assert deltas == [
        Delta(content="a", reasoning="r"),
        Delta(tool_calls=[ToolCallDelta(0, "c1", "add", '{"a": 1}')]),
        Delta(tool_calls=[ToolCallDelta(1, "c2", "mul", '{"b": 2}')]),
        Delta(
            usage=Usage(3, 4),
            finish_reason="tool_calls",
            provider_data={"x": {"k": 1}},
        ),
    ]
    assert agglomerate_deltas(deltas) == message


async def test_plain_chat_message_yields_one_delta() -> None:
    deltas = await _collect(ScriptedModel([ChatMessage("assistant", "hi")]))

    assert deltas == [Delta(content="hi")]


async def test_delta_sequence_turn_is_yielded_as_given() -> None:
    turn = [Delta(content="x"), Delta(finish_reason="stop")]

    deltas = await _collect(ScriptedModel([turn]))

    assert len(deltas) == 2
    assert deltas[0] is turn[0]
    assert deltas[1] is turn[1]


async def test_scripted_exception_raises_before_the_first_delta() -> None:
    model = ScriptedModel([ValueError("boom")])

    with pytest.raises(ValueError, match="^boom$"):
        await anext(aiter(model.stream([])))
    assert len(model.calls) == 1


async def test_scripted_model_out_of_turns() -> None:
    with pytest.raises(AssertionError) as caught:
        await _collect(ScriptedModel([]))

    assert str(caught.value) == "ScriptedModel has no more turns"


async def test_scripted_model_records_copies() -> None:
    definition = ToolDefinition("t", "d", {"type": "object"})
    model = ScriptedModel([ChatMessage("assistant", "x")])
    messages = [ChatMessage("user", "q")]

    [_ async for _ in model.stream(messages, [definition])]
    messages.append(ChatMessage("user", "later"))

    assert len(model.calls[0]) == 1
    assert model.tools == [[definition]]


# check_model: live mode and the rules
def _conforming_turns() -> list[ChatMessage]:
    return [
        ChatMessage("assistant", "pong", usage=Usage(1, 1), finish_reason="stop"),
        ChatMessage(
            "assistant",
            tool_calls=[ToolCall("c1", "echo", {"text": "ping"})],
            usage=Usage(1, 1),
            finish_reason="tool_calls",
        ),
    ]


def _failed(report: ConformanceReport) -> dict[int, str]:
    return {r.number: r.reason for r in report.rules if r.status == "failed"}


async def test_check_model_passes_a_conforming_model() -> None:
    seen: list[object] = []

    def factory(transport: httpx.AsyncBaseTransport | None) -> ScriptedModel:
        seen.append(transport)
        return ScriptedModel(_conforming_turns())

    report = await check_model(factory, live=True)

    assert seen == [None]
    assert report.ok is True
    assert [r.number for r in report.rules] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert {r.status for r in report.rules} == {"passed"}
    first, second = report.turns
    assert first is not None and first.content == "pong"
    assert second is not None
    assert second.tool_calls[0] == ToolCall("c1", "echo", {"text": "ping"})


async def test_check_model_sends_the_fixed_script() -> None:
    model = ScriptedModel(_conforming_turns())

    await check_model(lambda _: model, live=True)

    assert [(m.role, m.content) for m in model.calls[0]] == [
        ("system", "You are a test subject. Follow instructions exactly."),
        ("user", "Reply with exactly the word: pong"),
    ]
    assert model.tools[0] == []
    assert model.calls[1][1].content == (
        "Call the echo tool with text set to ping. Do not answer in prose."
    )
    (echo,) = model.tools[1]
    assert echo.name == "echo"
    assert echo.parameters["required"] == ["text"]
    assert echo.parameters["properties"]["text"]["type"] == "string"


_USAGE = Delta(usage=Usage(1, 1))


def _text_turn(*extra: Delta) -> list[Delta]:
    return [Delta(content="pong"), _USAGE, Delta(finish_reason="stop"), *extra]


def _tool_turn(arguments: str = '{"text": "ping"}') -> list[Delta]:
    return [
        Delta(tool_calls=[ToolCallDelta(0, "c1", "echo", arguments)]),
        _USAGE,
        Delta(finish_reason="tool_calls"),
    ]


class _MutatingModel:
    """Appends to its messages or tools on one turn, then behaves."""

    def __init__(self, target: str, turn: int) -> None:
        self.target = target
        self.turn = turn
        self.inner = ScriptedModel(_conforming_turns())

    async def stream(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolDefinition] = ()
    ) -> AsyncIterator[Delta]:
        if len(self.inner.calls) + 1 == self.turn:
            if self.target == "messages":
                messages.append(ChatMessage("user", "x"))  # type: ignore[attr-defined]
            else:
                tools.append(_ECHO_COPY)  # type: ignore[attr-defined]
        async for delta in self.inner.stream(messages, tools):
            yield delta


_ECHO_COPY = ToolDefinition("other", "x", {"type": "object"})


_BROKEN: dict[str, tuple[list[ScriptedTurn], dict[int, str]]] = {
    "a_no_finish_reason": (
        [[Delta(content="pong"), _USAGE], _tool_turn()],
        {2: "finish_reason was never yielded (turn 1)"},
    ),
    "b_two_finish_reasons": (
        [_text_turn(), [*_tool_turn(), Delta(finish_reason="stop")]],
        {2: "finish_reason was yielded 2 times (turn 2)"},
    ),
    "c_empty_content": (
        [[_USAGE, Delta(finish_reason="stop")], _tool_turn()],
        {1: "text turn reassembled to empty content (turn 1)"},
    ),
    "d_no_usage": (
        [[Delta(content="pong"), Delta(finish_reason="stop")], _tool_turn()],
        {3: "usage was never yielded (turn 1)"},
    ),
    "e_negative_usage": (
        [
            [Delta(content="pong"), Delta(usage=Usage(-1, 1), finish_reason="stop")],
            _tool_turn(),
        ],
        {3: "usage has a negative count: input_tokens=-1 (turn 1)"},
    ),
    "f_text_only_tool_turn": (
        [_text_turn(), _text_turn()],
        {4: "no tool call was reassembled (turn 2)"},
    ),
    "h_runtime_error": (
        [_text_turn(), RuntimeError("bug")],
        {
            2: "turn 2 raised RuntimeError: bug",
            3: "turn 2 raised RuntimeError: bug",
            4: "turn 2 raised RuntimeError: bug",
            5: "RuntimeError: bug (turn 2)",
        },
    ),
    "l_empty_delta": (
        [[Delta(), *_text_turn()], _tool_turn()],
        {8: "a Delta carried nothing (turn 1)"},
    ),
}


@pytest.mark.parametrize("case", sorted(_BROKEN))
async def test_check_model_flags_a_broken_rule(case: str) -> None:
    turns, expected = _BROKEN[case]

    report = await check_model(lambda _: ScriptedModel(turns), live=True)

    assert report.ok is False
    assert _failed(report) == expected


async def test_unparseable_tool_arguments_fail_rule_4() -> None:
    model = ScriptedModel([_text_turn(), _tool_turn('{"text": ')])

    report = await check_model(lambda _: model, live=True)

    assert set(_failed(report)) == {4}
    assert _failed(report)[4].startswith(
        "ModelOutputError: Invalid tool call arguments"
    )


@pytest.mark.parametrize(
    ("target", "turn", "reason"),
    [
        ("messages", 1, "messages were changed (turn 1)"),
        ("tools", 2, "tools were changed (turn 2)"),
    ],
)
async def test_mutating_inputs_fails_rule_6(
    target: str, turn: int, reason: str
) -> None:
    report = await check_model(lambda _: _MutatingModel(target, turn), live=True)

    assert _failed(report) == {6: reason}


async def test_unserialisable_provider_data_fails_rule_7() -> None:
    turn = [Delta(content="pong", provider_data={"x": {1, 2}}), *_text_turn()[1:]]

    report = await check_model(lambda _: ScriptedModel([turn, _tool_turn()]), live=True)

    assert set(_failed(report)) == {7}
    assert _failed(report)[7].startswith("provider_data is not JSON-serialisable")


async def test_model_error_fails_the_dependent_rules() -> None:
    model = ScriptedModel([ModelAuthError("bad key"), _tool_turn()])

    report = await check_model(lambda _: model, live=True)

    reason = "turn 1 raised ModelAuthError: bad key"
    assert _failed(report) == {1: reason, 2: reason, 3: reason}


async def test_expects_tool_calls_false_skips_rule_4() -> None:
    model = ScriptedModel([_text_turn(), _text_turn()])

    report = await check_model(lambda _: model, live=True, expects_tool_calls=False)

    rule_4 = report.rules[3]
    assert (rule_4.status, rule_4.reason) == ("skipped", "")
    assert report.ok is True
    assert [t.name for t in model.tools[1]] == ["echo"]


async def test_check_model_needs_exactly_one_mode(tmp_path: Path) -> None:
    calls: list[object] = []

    def factory(transport: httpx.AsyncBaseTransport | None) -> ScriptedModel:
        calls.append(transport)
        return ScriptedModel([])

    message = "^check_model needs exactly one of recorded= or live=True$"
    with pytest.raises(ValueError, match=message):
        await check_model(factory)
    with pytest.raises(ValueError, match=message):
        await check_model(factory, recorded=tmp_path / "f.jsonl", live=True)
    assert calls == []


async def _two_failures() -> ConformanceReport:
    turns = [[Delta(content="pong"), _USAGE], _text_turn()]
    return await check_model(lambda _: ScriptedModel(turns), live=True)


async def test_raise_for_failures_lists_every_failed_rule() -> None:
    report = await _two_failures()

    with pytest.raises(AssertionError) as caught:
        report.raise_for_failures()

    assert str(caught.value).splitlines() == [
        "2 conformance rules failed:",
        "rule 2 (finish_reason once): finish_reason was never yielded (turn 1)",
        "rule 4 (tool call reassembles): no tool call was reassembled (turn 2)",
    ]
    passing = await check_model(lambda _: ScriptedModel(_conforming_turns()), live=True)
    passing.raise_for_failures()


async def test_report_str_has_one_line_per_rule() -> None:
    lines = str(await _two_failures()).splitlines()

    assert len(lines) == 8
    assert lines[0] == "rule 1 (text reassembles): passed"
    assert lines[1] == (
        "rule 2 (finish_reason once): failed: finish_reason was never yielded (turn 1)"
    )


# Fixtures, recording and replay
def _exchange(body_lines: list[str], status: int = 200) -> Exchange:
    return Exchange(
        request={"method": "POST", "url": "http://test/v1", "headers": {}, "body": {}},
        status=status,
        headers={"content-type": "text/event-stream"},
        body_lines=body_lines,
    )


def _sse(*chunks: Mapping[str, object]) -> list[str]:
    lines: list[str] = []
    for chunk in chunks:
        lines += [f"data: {json.dumps(chunk)}", ""]
    return [*lines, "data: [DONE]", ""]


def _choice(
    delta: Mapping[str, object], finish: str | None = None
) -> dict[str, object]:
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


_USAGE_CHUNK = {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1}}
_TEXT_BODY = _sse(_choice({"content": "pong"}, "stop"), _USAGE_CHUNK)
_TOOL_BODY = _sse(
    _choice(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": "echo", "arguments": '{"text": "ping"}'},
                }
            ]
        },
        "tool_calls",
    ),
    _USAGE_CHUNK,
)


def _write(tmp_path: Path, exchanges: list[Exchange]) -> Path:
    path = tmp_path / "fixture.jsonl"
    write_fixture(path, exchanges)
    return path


def _openai_factory(transport: httpx.AsyncBaseTransport | None) -> OpenAIChatModel:
    return OpenAIChatModel("m", api_key="k", transport=transport)


def test_fixture_round_trip(tmp_path: Path) -> None:
    exchanges = [_exchange(_TEXT_BODY), _exchange(["{}"], status=429)]

    path = _write(tmp_path, exchanges)

    assert read_fixture(path) == exchanges
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        assert set(json.loads(line)) == {"request", "status", "headers", "body_lines"}


def _serve(body: str, seen: list[httpx.Request] | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, content=body.encode())

    return httpx.MockTransport(handler)


async def test_recording_transport_redacts_keys() -> None:
    recorder = RecordingTransport(inner=_serve("hello"))
    headers = {
        "Authorization": "Bearer sk-test-1",
        "X-Api-Key": "sk-ant-test-2",
        "api-key": "k3",
        "X-Goog-Api-Key": "google-test-key",
        "X-Trace": "t",
    }

    async with httpx.AsyncClient(transport=recorder) as client:
        response = await client.post("http://test/x", headers=headers, json={"a": 1})

    assert response.text == "hello"
    recorded = recorder.exchanges[0].request["headers"]
    for name in ("authorization", "x-api-key", "api-key", "x-goog-api-key"):
        assert recorded[name] == "‹redacted›"
    assert recorded["x-trace"] == "t"
    for secret in ("sk-test-1", "sk-ant-test-2", "k3", "google-test-key"):
        assert secret not in repr(recorder.exchanges)
    assert recorder.exchanges[0].request["body"] == {"a": 1}


async def test_recording_requests_identity_encoding() -> None:
    seen: list[httpx.Request] = []
    recorder = RecordingTransport(inner=_serve("a\nb\n", seen))

    async with httpx.AsyncClient(transport=recorder) as client:
        await client.get("http://test/x")

    assert seen[0].headers["accept-encoding"] == "identity"
    assert recorder.exchanges[0].body_lines == ["a", "b", ""]


async def test_recorded_mode_replays_through_check_model(tmp_path: Path) -> None:
    bodies = iter(["\n".join(_TEXT_BODY), "\n".join(_TOOL_BODY)])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=next(bodies).encode())

    recorder = RecordingTransport(inner=httpx.MockTransport(handler))
    live = await check_model(lambda _: _openai_factory(recorder), live=True)
    assert live.ok, str(live)
    path = _write(tmp_path, recorder.exchanges)
    transports: list[object] = []

    def factory(transport: httpx.AsyncBaseTransport | None) -> OpenAIChatModel:
        transports.append(transport)
        return _openai_factory(transport)

    report = await check_model(factory, recorded=path)

    assert report.ok, str(report)
    assert isinstance(transports[0], httpx.MockTransport)


class _LineProbe:
    """A model that turns every response line into a text Delta."""

    def __init__(self, transport: httpx.AsyncBaseTransport | None) -> None:
        self.transport = transport
        self.lines: list[str] = []

    async def stream(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolDefinition] = ()
    ) -> AsyncIterator[Delta]:
        async with httpx.AsyncClient(transport=self.transport) as client:
            async with client.stream("POST", "http://test/x") as response:
                async for line in response.aiter_lines():
                    self.lines.append(line)
        yield Delta(content="x", usage=Usage(1, 1), finish_reason="stop")


async def test_replay_streams_the_recorded_lines(tmp_path: Path) -> None:
    body = ["event: x", 'data: {"a": 1}', "", "data: [DONE]"]
    path = _write(tmp_path, [_exchange(body), _exchange(body)])
    probes: list[_LineProbe] = []

    def factory(transport: httpx.AsyncBaseTransport | None) -> _LineProbe:
        probes.append(_LineProbe(transport))
        return probes[-1]

    await check_model(factory, recorded=path, expects_tool_calls=False)

    assert probes[0].lines == body * 2


async def test_replay_ignores_recorded_encoding_headers(tmp_path: Path) -> None:
    headers = {
        "content-encoding": "gzip",
        "content-length": "999",
        "transfer-encoding": "chunked",
    }
    exchanges = [
        Exchange(request={}, status=200, headers=headers, body_lines=lines)
        for lines in (_TEXT_BODY, _TOOL_BODY)
    ]

    report = await check_model(_openai_factory, recorded=_write(tmp_path, exchanges))

    assert report.ok, str(report)


async def test_one_exchange_fixture_is_rejected(tmp_path: Path) -> None:
    calls: list[object] = []
    path = _write(tmp_path, [_exchange(_TEXT_BODY)])

    def factory(transport: httpx.AsyncBaseTransport | None) -> ScriptedModel:
        calls.append(transport)
        return ScriptedModel([])

    with pytest.raises(ValueError) as caught:
        await check_model(factory, recorded=path)

    assert str(caught.value) == (
        "fixture has 1 exchange; check_model needs at least 2 "
        "(a text turn and a tool turn)"
    )
    assert calls == []


class _TwoRequestsPerTurn:
    """Makes a second HTTP request on every turn after the first."""

    def __init__(self, transport: httpx.AsyncBaseTransport | None) -> None:
        self.transport = transport
        self.turns = 0

    async def stream(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolDefinition] = ()
    ) -> AsyncIterator[Delta]:
        self.turns += 1
        async with httpx.AsyncClient(transport=self.transport) as client:
            for _ in range(1 if self.turns == 1 else 2):
                await client.post("http://test/x")
        yield Delta(content="x", usage=Usage(1, 1), finish_reason="stop")


async def test_extra_request_beyond_the_fixture_fails_rule_5(tmp_path: Path) -> None:
    path = _write(tmp_path, [_exchange(["a"]), _exchange(["b"])])

    report = await check_model(_TwoRequestsPerTurn, recorded=path)

    assert "request 3 has no recorded exchange: the fixture has 2" in _failed(report)[5]


@pytest.mark.parametrize(
    ("third", "reason"),
    [
        (_exchange(['{"error": {"message": "slow"}}'], status=429), None),
        (_exchange(_TEXT_BODY), "turn 3 should raise a ModelError but raised nothing"),
    ],
)
async def test_third_exchange_must_raise_a_model_error(
    tmp_path: Path, third: Exchange, reason: str | None
) -> None:
    path = _write(tmp_path, [_exchange(_TEXT_BODY), _exchange(_TOOL_BODY), third])

    report = await check_model(_openai_factory, recorded=path)

    assert _failed(report).get(5) == reason


def test_read_fixture_rejects_bad_files(tmp_path: Path) -> None:
    good = json.dumps(
        {"request": {}, "status": 200, "headers": {}, "body_lines": ["x"]}
    )
    missing = json.dumps({"request": {}, "status": 200, "headers": {}})

    with pytest.raises(FileNotFoundError):
        read_fixture(tmp_path / "absent.jsonl")
    bad_json = tmp_path / "bad.jsonl"
    bad_json.write_text(f"{good}\nnot json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_fixture(bad_json)
    no_body = tmp_path / "no_body.jsonl"
    no_body.write_text(missing + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1 is missing body_lines"):
        read_fixture(no_body)


# httpx stays optional
def _without_httpx(code: str) -> subprocess.CompletedProcess[str]:
    script = "import sys\nsys.modules['httpx'] = None\n" + code
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )


def test_testing_imports_without_httpx() -> None:
    result = _without_httpx(
        "import asyncio\n"
        "from agentling import ChatMessage, ToolCall, Usage\n"
        "from agentling.testing import ScriptedModel, check_model\n"
        "turns = [\n"
        "    ChatMessage('assistant', 'pong', usage=Usage(1, 1),\n"
        "                finish_reason='stop'),\n"
        "    ChatMessage('assistant', tool_calls=[ToolCall('c1', 'echo', {})],\n"
        "                usage=Usage(1, 1), finish_reason='tool_calls'),\n"
        "]\n"
        "report = asyncio.run(check_model(lambda t: ScriptedModel(turns), live=True))\n"
        "print(report.ok)\n"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


def test_httpx_features_raise_import_error_without_httpx() -> None:
    result = _without_httpx(
        "import asyncio\n"
        "from agentling.testing import RecordingTransport, check_model\n"
        "for attempt in (\n"
        "    lambda: asyncio.run(check_model(lambda t: None, recorded='x.jsonl')),\n"
        "    RecordingTransport,\n"
        "):\n"
        "    try:\n"
        "        attempt()\n"
        "    except ImportError as exc:\n"
        "        print(exc)\n"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "check_model needs httpx: pip install httpx",
        "RecordingTransport needs httpx: pip install httpx",
    ]


_VENDOR_SDKS = {"openai", "anthropic", "ollama", "google", "boto3", "mistralai"}


def _imported_modules(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    return names


def test_testing_module_imports() -> None:
    for path in (REPO / "src" / "agentling").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not _imported_modules(tree) & _VENDOR_SDKS, path.name

    tree = ast.parse((REPO / "src/agentling/testing.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Try):
            handlers = {ast.unparse(h.type) for h in node.handlers if h.type}
            assert handlers == {"ImportError"}
            continue
        if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING":
            continue
        # Functions may import httpx lazily; only module-level imports count.
        if isinstance(node, ast.Import | ast.ImportFrom):
            assert "httpx" not in _imported_modules(node), ast.unparse(node)


def test_recording_transport_is_an_async_base_transport() -> None:
    transport: httpx.AsyncBaseTransport = RecordingTransport()

    assert isinstance(transport, httpx.AsyncBaseTransport)


def test_scripted_model_lives_in_agentling_testing() -> None:
    import agentling

    assert "ScriptedModel" not in agentling.__all__
    assert "check_model" not in agentling.__all__
    assert not hasattr(agentling, "ScriptedModel")


def test_no_copied_fake_models_remain() -> None:
    copies = [
        f"{path.relative_to(REPO)}: {name}"
        for folder in ("tests", "examples")
        for path in (REPO / folder).rglob("*.py")
        for name in ("class FakeModel", "class _ScriptedModel", "class ScriptedModel")
        # This file names the classes it looks for, so it is skipped.
        if path != Path(__file__) and name in path.read_text(encoding="utf-8")
    ]

    assert copies == []
    assert "class ScriptedModel" in (REPO / "src/agentling/testing.py").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("turn", ["oops", b"oops"])
async def test_a_string_turn_is_rejected(turn: str | bytes) -> None:
    model = ScriptedModel([turn])  # type: ignore[list-item]

    with pytest.raises(TypeError, match="a turn must be a ChatMessage"):
        await _collect(model)


async def test_recording_passes_a_body_the_server_still_encoded() -> None:
    # A server that ignores accept-encoding: identity and gzips anyway.
    import gzip

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            content=gzip.compress(b"plain text"),
        )

    recorder = RecordingTransport(inner=httpx.MockTransport(handler))

    async with httpx.AsyncClient(transport=recorder) as client:
        response = await client.get("http://test/x")

    assert response.text == "plain text"
    assert recorder.exchanges[0].body_lines == ["plain text"]
