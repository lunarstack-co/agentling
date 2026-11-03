"""Helpers for testing agents and model adapters.

ScriptedModel replays scripted turns, so an agent runs with no network and no
API key. check_model runs a model adapter through a fixed two-turn script and
reports which rule of the Model contract it broke. RecordingTransport,
write_fixture and read_fixture capture real HTTP exchanges so check_model can
replay them offline.

httpx is optional. ScriptedModel and check_model(live=True) work without it;
recorded mode and RecordingTransport need it.
"""

from __future__ import annotations

import copy
import json
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .errors import ModelError, ModelOutputError
from .models import (
    ChatMessage,
    Delta,
    Model,
    ToolCallDelta,
    ToolDefinition,
    agglomerate_deltas,
)

if TYPE_CHECKING:
    import httpx

# RecordingTransport subclasses httpx's transport so it type-checks wherever a
# transport is expected. Without httpx it falls back to a plain base and
# refuses to start.
try:
    from httpx import AsyncBaseTransport as _TransportBase
except ImportError:

    class _TransportBase:  # type: ignore[no-redef]
        pass


ScriptedTurn = ChatMessage | Sequence[Delta] | Exception

_REDACTED = "‹redacted›"
_SECRET_HEADERS = frozenset({"authorization", "x-api-key", "api-key"})
# Replayed bodies are plain text, so headers about the original encoding
# would make httpx decode them a second time.
_DROPPED_ON_REPLAY = frozenset(
    {"content-encoding", "content-length", "transfer-encoding"}
)


class ScriptedModel:
    """A model that replays one scripted turn per stream() call.

    A ChatMessage turn streams as pieces: its content and reasoning, one
    Delta per tool call, then its usage, finish_reason and provider_data.
    A sequence of Deltas is yielded as given. An exception is raised before
    the first Delta. Each call's messages and tools are recorded in `calls`
    and `tools`.
    """

    def __init__(self, turns: Sequence[ScriptedTurn]) -> None:
        self._turns = list(turns)
        self.calls: list[list[ChatMessage]] = []
        self.tools: list[list[ToolDefinition]] = []

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition] = (),
    ) -> AsyncIterator[Delta]:
        """Yield the next scripted turn."""

        self.calls.append(list(messages))
        self.tools.append(list(tools))
        index = len(self.calls) - 1
        if index >= len(self._turns):
            raise AssertionError("ScriptedModel has no more turns")
        turn = self._turns[index]
        if isinstance(turn, ChatMessage):
            for delta in _message_deltas(turn):
                yield delta
        elif isinstance(turn, Exception):
            raise turn
        elif isinstance(turn, (str, bytes)):
            raise TypeError(
                "a turn must be a ChatMessage, a sequence of Deltas, or an exception"
            )
        else:
            for delta in turn:
                yield delta


def _message_deltas(message: ChatMessage) -> list[Delta]:
    deltas: list[Delta] = []
    if message.content or message.reasoning:
        deltas.append(
            Delta(content=message.content or None, reasoning=message.reasoning)
        )
    for index, call in enumerate(message.tool_calls):
        fragment = ToolCallDelta(index, call.id, call.name, json.dumps(call.arguments))
        deltas.append(Delta(tool_calls=[fragment]))
    if (
        message.usage is not None
        or message.finish_reason is not None
        or message.provider_data is not None
    ):
        deltas.append(
            Delta(
                usage=message.usage,
                finish_reason=message.finish_reason,
                provider_data=message.provider_data,
            )
        )
    return deltas


# --------------------------------------------------------------------------- #
# Conformance report
# --------------------------------------------------------------------------- #

_RULE_NAMES = {
    1: "text reassembles",
    2: "finish_reason once",
    3: "usage present",
    4: "tool call reassembles",
    5: "only ModelError escapes",
    6: "inputs not mutated",
    7: "provider_data serialisable",
    8: "no empty Delta",
}


@dataclass(frozen=True)
class RuleResult:
    """The outcome of one contract rule."""

    number: int
    name: str
    status: Literal["passed", "failed", "skipped"]
    reason: str = ""

    def __str__(self) -> str:
        if self.status == "failed":
            return f"rule {self.number} ({self.name}): failed: {self.reason}"
        return f"rule {self.number} ({self.name}): {self.status}"


@dataclass(frozen=True)
class ConformanceReport:
    """What check_model found: one result per rule, and the turns it saw.

    `turns` holds the reassembled text turn and tool turn, or None for a turn
    that raised or couldn't be reassembled.
    """

    rules: tuple[RuleResult, ...]
    turns: tuple[ChatMessage | None, ...]

    @property
    def ok(self) -> bool:
        """True when no rule failed. Skipped rules don't count."""

        return all(rule.status != "failed" for rule in self.rules)

    def raise_for_failures(self) -> None:
        """Raise AssertionError listing every failed rule, if any failed."""

        failed = [rule for rule in self.rules if rule.status == "failed"]
        if not failed:
            return
        noun = "rule" if len(failed) == 1 else "rules"
        lines = [f"{len(failed)} conformance {noun} failed:"]
        lines += [f"rule {r.number} ({r.name}): {r.reason}" for r in failed]
        raise AssertionError("\n".join(lines))

    def __str__(self) -> str:
        return "\n".join(str(rule) for rule in self.rules)


# --------------------------------------------------------------------------- #
# check_model
# --------------------------------------------------------------------------- #

_SYSTEM = "You are a test subject. Follow instructions exactly."
_TEXT_PROMPT = "Reply with exactly the word: pong"
_TOOL_PROMPT = "Call the echo tool with text set to ping. Do not answer in prose."
_ECHO = ToolDefinition(
    name="echo",
    description="Echo the given text back.",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string", "description": "The text to echo."}},
        "required": ["text"],
        "additionalProperties": False,
    },
)


@dataclass
class _Turn:
    number: int
    deltas: list[Delta]
    error: Exception | None
    messages_changed: bool
    tools_changed: bool


async def check_model(
    factory: Callable[[httpx.AsyncBaseTransport | None], Model],
    *,
    recorded: str | Path | None = None,
    live: bool = False,
    expects_tool_calls: bool = True,
) -> ConformanceReport:
    """Check a model adapter against the Model contract.

    `factory` builds the adapter around an httpx transport. In recorded mode
    it gets a transport that replays the fixture at `recorded`; with
    `live=True` it gets None and talks to the real backend (or, for a model
    that needs no network, ignores the argument). Exactly one mode is needed.

    The script is a text turn, then a turn offering an `echo` tool. A fixture
    with a third exchange adds a text turn that must raise a ModelError.
    """

    if (recorded is not None) == live:
        raise ValueError("check_model needs exactly one of recorded= or live=True")

    expect_failure = False
    if recorded is not None:
        _require_httpx("check_model")
        exchanges = read_fixture(recorded)
        if len(exchanges) < 2:
            noun = "exchange" if len(exchanges) == 1 else "exchanges"
            raise ValueError(
                f"fixture has {len(exchanges)} {noun}; check_model needs at least 2 "
                "(a text turn and a tool turn)"
            )
        expect_failure = len(exchanges) >= 3
        model = factory(_replay_transport(exchanges))
    else:
        model = factory(None)

    text = [ChatMessage(role="system", content=_SYSTEM), _user(_TEXT_PROMPT)]
    tool = [ChatMessage(role="system", content=_SYSTEM), _user(_TOOL_PROMPT)]
    turns = [
        await _run_turn(model, 1, text, []),
        await _run_turn(model, 2, tool, [_ECHO]),
    ]
    if expect_failure:
        turns.append(await _run_turn(model, 3, text, []))

    first, second = turns[0], turns[1]
    rules = (
        _rule_text(first),
        _rule_finish_reason([first, second]),
        _rule_usage([first, second]),
        _rule_tool_call(second, expects_tool_calls),
        _rule_errors(turns, expect_failure),
        _rule_inputs(turns),
        _rule_provider_data(turns),
        _rule_empty_deltas(turns),
    )
    return ConformanceReport(
        rules=rules, turns=(_reassemble(first), _reassemble(second))
    )


def _user(content: str) -> ChatMessage:
    return ChatMessage(role="user", content=content)


async def _run_turn(
    model: Model, number: int, messages: list[ChatMessage], tools: list[ToolDefinition]
) -> _Turn:
    messages_before = copy.deepcopy(messages)
    tools_before = copy.deepcopy(tools)
    deltas: list[Delta] = []
    error: Exception | None = None
    try:
        async for delta in model.stream(messages, tools):
            deltas.append(delta)
    except Exception as exc:
        error = exc
    return _Turn(
        number=number,
        deltas=deltas,
        error=error,
        messages_changed=messages != messages_before,
        tools_changed=tools != tools_before,
    )


def _reassemble(turn: _Turn) -> ChatMessage | None:
    if turn.error is not None:
        return None
    try:
        return agglomerate_deltas(turn.deltas)
    except ModelOutputError:
        return None


def _passed(number: int) -> RuleResult:
    return RuleResult(number, _RULE_NAMES[number], "passed")


def _failed(number: int, reason: str) -> RuleResult:
    return RuleResult(number, _RULE_NAMES[number], "failed", reason)


def _raised(turn: _Turn) -> str:
    assert turn.error is not None
    return f"turn {turn.number} raised {type(turn.error).__name__}: {turn.error}"


def _rule_text(turn: _Turn) -> RuleResult:
    if turn.error is not None:
        return _failed(1, _raised(turn))
    try:
        message = agglomerate_deltas(turn.deltas)
    except ModelOutputError as exc:
        return _failed(1, f"ModelOutputError: {exc} (turn 1)")
    if not message.content:
        return _failed(1, "text turn reassembled to empty content (turn 1)")
    return _passed(1)


def _rule_finish_reason(turns: list[_Turn]) -> RuleResult:
    for turn in turns:
        if turn.error is not None:
            return _failed(2, _raised(turn))
        count = sum(delta.finish_reason is not None for delta in turn.deltas)
        if count == 0:
            return _failed(2, f"finish_reason was never yielded (turn {turn.number})")
        if count > 1:
            return _failed(
                2, f"finish_reason was yielded {count} times (turn {turn.number})"
            )
    return _passed(2)


def _rule_usage(turns: list[_Turn]) -> RuleResult:
    for turn in turns:
        if turn.error is not None:
            return _failed(3, _raised(turn))
        usages = [delta.usage for delta in turn.deltas if delta.usage is not None]
        if not usages:
            return _failed(3, f"usage was never yielded (turn {turn.number})")
        for usage in usages:
            for name in (
                "input_tokens",
                "output_tokens",
                "cached_input_tokens",
                "reasoning_tokens",
            ):
                value = getattr(usage, name)
                if type(value) is not int:
                    return _failed(
                        3,
                        f"usage has a non-integer count: {name}={value!r} "
                        f"(turn {turn.number})",
                    )
                if value < 0:
                    return _failed(
                        3,
                        f"usage has a negative count: {name}={value} "
                        f"(turn {turn.number})",
                    )
    return _passed(3)


def _rule_tool_call(turn: _Turn, expected: bool) -> RuleResult:
    if not expected:
        return RuleResult(4, _RULE_NAMES[4], "skipped")
    if turn.error is not None:
        return _failed(4, _raised(turn))
    try:
        message = agglomerate_deltas(turn.deltas)
    except ModelOutputError as exc:
        return _failed(4, f"ModelOutputError: {exc} (turn 2)")
    if not message.tool_calls:
        return _failed(4, "no tool call was reassembled (turn 2)")
    return _passed(4)


def _rule_errors(turns: list[_Turn], expect_failure: bool) -> RuleResult:
    for turn in turns:
        if turn.error is not None and not isinstance(turn.error, ModelError):
            return _failed(
                5, f"{type(turn.error).__name__}: {turn.error} (turn {turn.number})"
            )
    if expect_failure and turns[2].error is None:
        return _failed(5, "turn 3 should raise a ModelError but raised nothing")
    return _passed(5)


def _rule_inputs(turns: list[_Turn]) -> RuleResult:
    for turn in turns:
        if turn.messages_changed:
            return _failed(6, f"messages were changed (turn {turn.number})")
        if turn.tools_changed:
            return _failed(6, f"tools were changed (turn {turn.number})")
    return _passed(6)


def _rule_provider_data(turns: list[_Turn]) -> RuleResult:
    for turn in turns:
        for delta in turn.deltas:
            if delta.provider_data is None:
                continue
            try:
                json.dumps(delta.provider_data)
            except (TypeError, ValueError) as exc:
                return _failed(
                    7,
                    f"provider_data is not JSON-serialisable: {exc} "
                    f"(turn {turn.number})",
                )
    return _passed(7)


def _rule_empty_deltas(turns: list[_Turn]) -> RuleResult:
    for turn in turns:
        for delta in turn.deltas:
            if not (
                delta.content
                or delta.reasoning
                or delta.tool_calls
                or delta.usage is not None
                or delta.finish_reason is not None
            ):
                return _failed(8, f"a Delta carried nothing (turn {turn.number})")
    return _passed(8)


# --------------------------------------------------------------------------- #
# Fixtures and recording
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Exchange:
    """One recorded HTTP request and its response.

    `request` holds the method, url, headers (keys redacted) and body (parsed
    JSON when possible). `body_lines` is the response body split on newlines;
    joining them with newlines gives the body back exactly.
    """

    request: dict[str, Any]
    status: int
    headers: dict[str, str]
    body_lines: list[str]


def write_fixture(path: str | Path, exchanges: Iterable[Exchange]) -> None:
    """Write exchanges as a fixture file: one JSON object per line."""

    lines = [
        json.dumps(
            {
                "request": exchange.request,
                "status": exchange.status,
                "headers": exchange.headers,
                "body_lines": exchange.body_lines,
            },
            ensure_ascii=False,
        )
        for exchange in exchanges
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_fixture(path: str | Path) -> list[Exchange]:
    """Read a fixture file written by write_fixture.

    Raises ValueError naming the line when a line isn't a valid exchange.
    """

    exchanges: list[Exchange] = []
    text = Path(path).read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: line {number} is not valid JSON") from exc
        exchanges.append(_exchange(raw, path, number))
    return exchanges


def _exchange(raw: object, path: str | Path, number: int) -> Exchange:
    where = f"{path}: line {number}"
    if not isinstance(raw, dict):
        raise ValueError(f"{where} is not a JSON object")
    for key, kind in (
        ("request", dict),
        ("status", int),
        ("headers", dict),
        ("body_lines", list),
    ):
        if key not in raw:
            raise ValueError(f"{where} is missing {key}")
        if not isinstance(raw[key], kind):
            raise ValueError(f"{where} has a {key} that isn't a {kind.__name__}")
    if not all(isinstance(item, str) for item in raw["body_lines"]):
        raise ValueError(f"{where} has body_lines that aren't all strings")
    return Exchange(
        request=raw["request"],
        status=raw["status"],
        headers={str(k): str(v) for k, v in raw["headers"].items()},
        body_lines=list(raw["body_lines"]),
    )


class RecordingTransport(_TransportBase):
    """An httpx transport that records every exchange it carries.

    Wraps `inner` (a real network transport by default). Each request asks
    for an unencoded body so the recording holds the bytes the adapter
    parsed. API keys in the authorization, x-api-key and api-key headers are
    replaced before anything is stored. Request bodies are recorded as sent,
    so a key a backend takes in the body is kept: check a fixture before
    committing it.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport | None = None) -> None:
        _require_httpx("RecordingTransport")
        import httpx

        self._inner = inner if inner is not None else httpx.AsyncHTTPTransport()
        self.exchanges: list[Exchange] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        import httpx

        request.headers["accept-encoding"] = "identity"
        response = await self._inner.handle_async_request(request)
        try:
            body = await response.aread()
        finally:
            await response.aclose()
        self.exchanges.append(
            Exchange(
                request={
                    "method": request.method,
                    "url": str(request.url),
                    "headers": _redacted(request.headers.items()),
                    "body": _request_body(request.content),
                },
                status=response.status_code,
                headers={k.lower(): v for k, v in response.headers.items()},
                body_lines=body.decode("utf-8", errors="replace").split("\n"),
            )
        )
        # The body is already read and decoded, so headers describing the
        # wire encoding would make the caller decode it a second time.
        passthrough = {
            key: value
            for key, value in response.headers.items()
            if key.lower() not in _DROPPED_ON_REPLAY
        }
        return httpx.Response(
            response.status_code,
            headers=passthrough,
            content=body,
            request=request,
        )

    async def aclose(self) -> None:
        await self._inner.aclose()


def _redacted(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    return {
        key.lower(): _REDACTED if key.lower() in _SECRET_HEADERS else value
        for key, value in headers
    }


def _request_body(content: bytes) -> Any:
    text = content.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _replay_transport(exchanges: list[Exchange]) -> httpx.AsyncBaseTransport:
    import httpx

    remaining = list(exchanges)
    count = len(exchanges)
    served = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal served
        served += 1
        if not remaining:
            raise AssertionError(
                f"request {served} has no recorded exchange: the fixture has {count}"
            )
        exchange = remaining.pop(0)
        headers = {
            key: value
            for key, value in exchange.headers.items()
            if key.lower() not in _DROPPED_ON_REPLAY
        }
        return httpx.Response(
            exchange.status, headers=headers, content=_lines(exchange.body_lines)
        )

    return httpx.MockTransport(handler)


async def _lines(lines: list[str]) -> AsyncIterator[bytes]:
    # One chunk per recorded line, so the adapter sees the framing it saw live.
    last = len(lines) - 1
    for index, line in enumerate(lines):
        yield (line if index == last else line + "\n").encode("utf-8")


def _require_httpx(feature: str) -> None:
    try:
        import httpx  # noqa: F401
    except ImportError as exc:
        raise ImportError(f"{feature} needs httpx: pip install httpx") from exc
