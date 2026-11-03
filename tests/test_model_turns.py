"""How the loop drives a model turn: retries, timeouts, the forced final answer,
provider_data, usage, finish_reason and the model check."""

import asyncio
import re
import statistics
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

from agentling.agent import Agent, AgentSession, _check_provider_data
from agentling.errors import (
    ModelAuthError,
    ModelError,
    ModelOutputError,
    ModelOutputTruncatedError,
    ModelRateLimitError,
    ModelServerError,
    ModelTimeoutError,
)
from agentling.events import Event, FinalEvent, TextDelta
from agentling.memory import ActionStep, FinalStep, TaskStep
from agentling.models import (
    ChatMessage,
    Delta,
    ToolCallDelta,
    ToolDefinition,
    Usage,
    agglomerate_deltas,
)
from agentling.retry import RetryPolicy
from agentling.tools import tool

ScriptEntry = list[Delta | BaseException] | BaseException

FAST = RetryPolicy(base_delay=0, jitter=0)
_SYNTHESISED_ID = re.compile(r"^agentling_call_0_[0-9a-f]{8}$")


class StreamFake:
    """A scripted model: one script entry per stream() call.

    An entry is a list of Deltas (any exception in it is raised at that
    point) or a single exception raised before the first Delta. Records what
    each call received and whether its generator was closed.
    """

    def __init__(
        self,
        script: Sequence[ScriptEntry],
        *,
        delay_before_first: float = 0.0,
        delay_between: float = 0.0,
        log: list[str] | None = None,
    ) -> None:
        self.script = list(script)
        self.delay_before_first = delay_before_first
        self.delay_between = delay_between
        self.log = log if log is not None else []
        self.calls: list[list[ChatMessage]] = []
        self.tools: list[list[ToolDefinition]] = []
        self.finally_ran: list[bool] = []

    async def stream(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolDefinition] = ()
    ) -> AsyncIterator[Delta]:
        attempt = len(self.calls)
        self.calls.append(list(messages))
        self.tools.append(list(tools))
        self.finally_ran.append(False)
        self.log.append("stream")
        entry = self.script[attempt]
        items = entry if isinstance(entry, list) else [entry]
        try:
            if self.delay_before_first:
                await asyncio.sleep(self.delay_before_first)
            for position, item in enumerate(items):
                if isinstance(item, BaseException):
                    raise item
                if position and self.delay_between:
                    await asyncio.sleep(self.delay_between)
                yield item
        finally:
            self.finally_ran[attempt] = True
            self.log.append("finally")


def _text(content: str, **fields: Any) -> list[Delta | BaseException]:
    return [Delta(content=content, **fields)]


def _call(
    name: str, arguments: str = "{}", call_id: str | None = "c1", **fields: Any
) -> list[Delta | BaseException]:
    return [
        Delta(tool_calls=[ToolCallDelta(0, call_id, name, arguments)], **fields),
    ]


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b


async def _collect(agent: Agent, task: str = "go") -> list[Event]:
    return [event async for event in agent.run(task, stream=True)]


async def _final(agent: Agent, task: str = "go") -> FinalEvent:
    final = (await _collect(agent, task))[-1]
    assert isinstance(final, FinalEvent)
    return final


class _Recorder:
    """An on_retry callback that records its arguments."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, ModelError, float]] = []

    def __call__(
        self, attempt: int, max_retries: int, exc: ModelError, delay: float
    ) -> None:
        self.calls.append((attempt, max_retries, exc, delay))


# --------------------------------------------------------------------------- #
# The stream-only protocol and tool definitions
# --------------------------------------------------------------------------- #
class OnlyStreamModel:
    async def stream(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolDefinition] = ()
    ) -> AsyncIterator[Delta]:
        yield Delta(content="hi", finish_reason="stop")


async def test_stream_only_model_runs() -> None:
    assert await Agent(model=OnlyStreamModel()).run("x") == "hi"


async def test_loop_passes_tool_definitions() -> None:
    fake = StreamFake([_text("done")])
    await Agent(model=fake, tools=[add]).run("go")

    assert all(isinstance(d, ToolDefinition) for d in fake.tools[0])
    assert [d.name for d in fake.tools[0]] == ["add", "final_answer"]


async def test_synthesised_id_round_trips() -> None:
    fake = StreamFake([_call("add", '{"a": 1, "b": 2}', call_id=None), _text("done")])
    session = Agent(model=fake, tools=[add]).start()

    assert await session.run("go") == "done"

    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    call_id = action.tool_results[0].tool_call_id
    assert _SYNTHESISED_ID.match(call_id)
    second_request = fake.calls[1]
    assert any(
        call.id == call_id
        for message in second_request
        if message.role == "assistant"
        for call in message.tool_calls
    )
    assert any(
        message.role == "tool" and message.tool_call_id == call_id
        for message in second_request
    )


# --------------------------------------------------------------------------- #
# Retry
# --------------------------------------------------------------------------- #
async def test_rate_limit_is_retried_then_succeeds() -> None:
    error = ModelRateLimitError(retry_after=0.01)
    fake = StreamFake([error, _text("ok")])
    recorder = _Recorder()

    answer = await Agent(model=fake, retry=RetryPolicy(on_retry=recorder)).run("x")

    assert answer == "ok"
    assert len(fake.calls) == 2
    assert len(recorder.calls) == 1
    attempt, max_retries, exc, delay = recorder.calls[0]
    assert (attempt, max_retries) == (1, 2)
    assert exc is error
    assert 0.008 <= delay <= 0.012


async def test_async_on_retry_is_awaited_before_the_next_attempt() -> None:
    log: list[str] = []

    async def on_retry(
        attempt: int, max_retries: int, exc: ModelError, delay: float
    ) -> None:
        await asyncio.sleep(0)
        log.append("on_retry")

    fake = StreamFake([ModelServerError(), _text("ok")], log=log)
    policy = RetryPolicy(base_delay=0, jitter=0, on_retry=on_retry)

    assert await Agent(model=fake, retry=policy).run("x") == "ok"
    assert log == ["stream", "finally", "on_retry", "stream", "finally"]


async def test_auth_error_is_not_retried() -> None:
    fake = StreamFake([ModelAuthError("bad key"), _text("never")])

    with pytest.raises(ModelAuthError, match="bad key"):
        await Agent(model=fake, retry=FAST).run("x")
    assert len(fake.calls) == 1


async def test_no_retry_after_a_delta() -> None:
    fake = StreamFake([[Delta(content="a"), ModelServerError()], _text("never")])
    events: list[Event] = []

    with pytest.raises(ModelServerError):
        async for event in Agent(model=fake, retry=FAST).run("x", stream=True):
            events.append(event)

    assert events == [TextDelta(text="a")]
    assert len(fake.calls) == 1


async def test_retries_exhausted_raises_the_last_error() -> None:
    errors = [ModelServerError(str(n)) for n in range(3)]
    fake = StreamFake(list(errors))
    recorder = _Recorder()
    policy = RetryPolicy(base_delay=0, jitter=0, on_retry=recorder)

    with pytest.raises(ModelServerError) as caught:
        await Agent(model=fake, retry=policy).run("x")

    assert caught.value is errors[2]
    assert len(fake.calls) == 3
    assert [call[0] for call in recorder.calls] == [1, 2]


async def test_instance_can_opt_out_of_retry() -> None:
    fake = StreamFake([ModelServerError(retryable=False), _text("never")])

    with pytest.raises(ModelServerError):
        await Agent(model=fake, retry=FAST).run("x")
    assert len(fake.calls) == 1


async def test_non_model_error_propagates_unchanged() -> None:
    fake = StreamFake([RuntimeError("bug"), _text("never")])

    with pytest.raises(RuntimeError, match="bug"):
        await Agent(model=fake, retry=FAST).run("x")
    assert len(fake.calls) == 1


async def test_on_retry_raising_stops_the_run() -> None:
    def on_retry(attempt: int, max_retries: int, exc: ModelError, delay: float) -> None:
        raise ValueError("stop retrying")

    fake = StreamFake([ModelServerError(), _text("never")])
    policy = RetryPolicy(base_delay=0, jitter=0, on_retry=on_retry)

    with pytest.raises(ValueError, match="stop retrying"):
        await Agent(model=fake, retry=policy).run("x")
    assert len(fake.calls) == 1


async def test_retry_none_disables_retries() -> None:
    fake = StreamFake([ModelServerError(), _text("never")])

    with pytest.raises(ModelServerError):
        await Agent(model=fake, retry=None).run("x")
    assert len(fake.calls) == 1


async def test_cancel_during_retry_sleep() -> None:
    fake = StreamFake([ModelServerError(), _text("never")])
    agent = Agent(model=fake, retry=RetryPolicy(base_delay=10, jitter=0))

    task = asyncio.ensure_future(agent.run("x"))
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(fake.calls) == 1


async def test_failed_attempt_is_closed_before_the_next() -> None:
    log: list[str] = []
    fake = StreamFake([ModelServerError(), _text("ok")], log=log)

    assert await Agent(model=fake, retry=FAST).run("x") == "ok"
    assert fake.finally_ran[0] is True
    assert log.index("finally") < log.index("stream", 1)


class _InterruptingModel:
    """Interrupts its session mid-stream and records whether it was closed."""

    def __init__(self) -> None:
        self.session: AgentSession | None = None
        self.closed = False

    async def stream(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolDefinition] = ()
    ) -> AsyncIterator[Delta]:
        try:
            yield Delta(content="a")
            assert self.session is not None
            self.session.interrupt()
            yield Delta(content="b")
            yield Delta(content="never consumed")
        finally:
            self.closed = True


async def test_interrupt_closes_the_abandoned_stream() -> None:
    model = _InterruptingModel()
    session = Agent(model=model).start()
    model.session = session

    assert await session.run("x") == "Run interrupted."
    assert model.closed is True


async def test_max_retries_zero_means_a_single_attempt() -> None:
    fake = StreamFake([ModelServerError(), _text("never")])

    with pytest.raises(ModelServerError):
        await Agent(model=fake, retry=RetryPolicy(max_retries=0)).run("x")
    assert len(fake.calls) == 1


async def test_interrupt_during_a_retry_wait_stops_without_another_attempt() -> None:
    fake = StreamFake([ModelServerError(), _text("after")])
    session = Agent(model=fake, retry=RetryPolicy(base_delay=10, jitter=0)).start()

    run = asyncio.ensure_future(session.run("x"))
    await asyncio.sleep(0.05)
    session.interrupt()

    assert await asyncio.wait_for(run, 1) == "Run interrupted."
    assert len(fake.calls) == 1
    # The interrupt was consumed, so the next run goes ahead.
    assert await session.run("again") == "after"


# --------------------------------------------------------------------------- #
# Per-chunk timeout
# --------------------------------------------------------------------------- #
async def test_slow_consumer_is_not_charged_to_model_timeout() -> None:
    fake = StreamFake([[Delta(content=c) for c in "abcdef"]], delay_between=0.01)
    agent = Agent(model=fake, model_timeout=0.1)
    text = []

    async for event in agent.run("x", stream=True):
        if isinstance(event, TextDelta):
            text.append(event.text)
        await asyncio.sleep(0.05)

    assert "".join(text) == "abcdef"


async def test_first_chunk_stall_is_retried() -> None:
    fake = StreamFake([_text("late"), _text("late")], delay_before_first=1.0)
    policy = RetryPolicy(max_retries=1, base_delay=0, jitter=0)

    with pytest.raises(ModelTimeoutError):
        await Agent(model=fake, model_timeout=0.05, retry=policy).run("x")
    assert len(fake.calls) == 2


async def test_mid_stream_stall_is_not_retried_and_closes_the_adapter() -> None:
    fake = StreamFake(
        [[Delta(content="a"), Delta(content="b")], _text("never")],
        delay_between=1.0,
    )

    with pytest.raises(ModelTimeoutError):
        await Agent(model=fake, model_timeout=0.05, retry=FAST).run("x")
    assert len(fake.calls) == 1
    assert fake.finally_ran[0] is True


async def test_no_model_timeout_means_unbounded() -> None:
    fake = StreamFake([_text("eventually")], delay_before_first=0.1)
    assert await Agent(model=fake, model_timeout=None).run("x") == "eventually"


# --------------------------------------------------------------------------- #
# Forced final answer at the step limit
# --------------------------------------------------------------------------- #
async def test_forced_final_uses_context_manager_and_no_tools() -> None:
    seen: list[int] = []

    def context_manager(messages: list[ChatMessage]) -> list[ChatMessage]:
        seen.append(len(messages))
        return messages

    fake = StreamFake([_call("add", '{"a": 1, "b": 2}'), _text("forced")])
    agent = Agent(model=fake, tools=[add], max_steps=1, context_manager=context_manager)

    final = await _final(agent)

    assert (final.answer, final.status) == ("forced", "max_steps")
    assert len(seen) == 2
    assert fake.tools[1] == []


async def test_forced_final_is_retried() -> None:
    fake = StreamFake(
        [_call("add", '{"a": 1, "b": 2}'), ModelServerError(), _text("forced")]
    )
    agent = Agent(model=fake, tools=[add], max_steps=1, retry=FAST)

    final = await _final(agent)

    assert (final.answer, final.status) == ("forced", "max_steps")
    assert len(fake.calls) == 3


async def test_interrupt_during_the_forced_final_retry_wait() -> None:
    fake = StreamFake([_call("add", '{"a": 1, "b": 2}'), ModelServerError()])
    policy = RetryPolicy(base_delay=10, jitter=0)
    session = Agent(model=fake, tools=[add], max_steps=1, retry=policy).start()

    async def collect() -> list[Event]:
        return [event async for event in session.run("go", stream=True)]

    events = asyncio.ensure_future(collect())
    await asyncio.sleep(0.05)
    session.interrupt()
    final = (await asyncio.wait_for(events, 1))[-1]

    assert isinstance(final, FinalEvent)
    assert (final.answer, final.status) == ("Run interrupted.", "interrupted")
    assert len(fake.calls) == 2


class _InterruptsTheForcedFinal:
    """Calls a tool, then interrupts its session during the forced final."""

    def __init__(self) -> None:
        self.session: AgentSession | None = None
        self.calls = 0

    async def stream(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolDefinition] = ()
    ) -> AsyncIterator[Delta]:
        self.calls += 1
        if self.calls == 1:
            yield Delta(tool_calls=[ToolCallDelta(0, "c1", "add", '{"a": 1, "b": 2}')])
            return
        yield Delta(content="par")
        assert self.session is not None
        self.session.interrupt()
        yield Delta(content="tial")


async def test_interrupt_during_the_forced_final_stream() -> None:
    model = _InterruptsTheForcedFinal()
    session = Agent(model=model, tools=[add], max_steps=1).start()
    model.session = session

    final = [event async for event in session.run("go", stream=True)][-1]

    assert isinstance(final, FinalEvent)
    assert (final.answer, final.status) == ("Run interrupted.", "interrupted")
    assert not any(isinstance(step, FinalStep) for step in session.memory.steps)


async def test_interrupt_before_the_forced_final_skips_its_model_call() -> None:
    fake = StreamFake([_call("add", '{"a": 1, "b": 2}'), _text("unreached")])
    session = Agent(model=fake, tools=[add], max_steps=1).start()
    session.step_callbacks.append(lambda step: session.interrupt())

    final = [event async for event in session.run("go", stream=True)][-1]

    assert isinstance(final, FinalEvent)
    assert final.status == "interrupted"
    assert len(fake.calls) == 1
    assert not any(isinstance(step, FinalStep) for step in session.memory.steps)


# --------------------------------------------------------------------------- #
# Interrupting the first model call, and resuming afterwards
# --------------------------------------------------------------------------- #
class _InterruptsFirstCall:
    """Interrupts its own session while streaming its first turn."""

    def __init__(self) -> None:
        self.session: AgentSession | None = None
        self.calls: list[list[ChatMessage]] = []
        self.finally_ran = False

    async def stream(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolDefinition] = ()
    ) -> AsyncIterator[Delta]:
        self.calls.append(list(messages))
        if len(self.calls) > 1:
            yield Delta(content="resumed")
            return
        try:
            yield Delta(content="par")
            assert self.session is not None
            self.session.interrupt()
            yield Delta(content="tial")
        finally:
            self.finally_ran = True


async def test_interrupt_during_the_first_call_keeps_only_the_task() -> None:
    model = _InterruptsFirstCall()
    session = Agent(model=model).start()
    model.session = session

    final = [event async for event in session.run("go", stream=True)][-1]

    assert isinstance(final, FinalEvent)
    assert final.status == "interrupted"
    assert session.memory.steps == [TaskStep(task="go")]
    assert model.finally_ran


async def test_resume_after_a_first_call_interrupt_sends_the_task_once() -> None:
    model = _InterruptsFirstCall()
    session = Agent(model=model).start()
    model.session = session

    assert await session.run("go") == "Run interrupted."
    assert await session.resume() == "resumed"

    assert [message.role for message in model.calls[1]] == ["system", "user"]
    assert sum(isinstance(step, TaskStep) for step in session.memory.steps) == 1


async def test_resume_after_a_failed_run_continues() -> None:
    fake = StreamFake(
        [_call("add", '{"a": 1, "b": 2}'), ModelServerError(), _text("done")]
    )
    session = Agent(model=fake, tools=[add], retry=None).start()

    with pytest.raises(ModelServerError):
        await session.run("go")
    assert isinstance(session.memory.steps[-1], ActionStep)

    assert await session.resume() == "done"
    assert sum(isinstance(step, TaskStep) for step in session.memory.steps) == 1


async def test_is_running_resets_after_a_run_raises() -> None:
    session = Agent(model=StreamFake([ModelAuthError("bad key")])).start()

    with pytest.raises(ModelAuthError):
        await session.run("go")

    assert session.is_running is False


# --------------------------------------------------------------------------- #
# Output cut off by the model's length limit
# --------------------------------------------------------------------------- #
_CUT_OFF = "Model output was cut off by its length limit while writing a tool call."


def _malformed() -> list[Delta | BaseException]:
    return _call("add", '{"a": ')


def _truncated() -> list[Delta | BaseException]:
    return [
        Delta(tool_calls=[ToolCallDelta(0, "c1", "add", '{"a": 1, "b":')]),
        Delta(finish_reason="length"),
    ]


def _corrections(fake: StreamFake) -> list[str]:
    return [
        message.content
        for call in fake.calls
        for message in call
        if message.role == "user"
        and message.content.startswith("Your previous response")
    ]


async def test_truncated_tool_call_raises_without_a_correction() -> None:
    fake = StreamFake([_truncated(), _text("unreached")])
    session = Agent(model=fake, tools=[add]).start()

    with pytest.raises(ModelOutputTruncatedError) as caught:
        await session.run("go")

    assert str(caught.value) == _CUT_OFF
    assert len(fake.calls) == 1
    assert _corrections(fake) == []
    assert session.memory.steps == [TaskStep(task="go")]


async def test_truncation_reported_on_the_same_delta_is_detected() -> None:
    cut = ToolCallDelta(0, "c1", "add", '{"a": 1, "b":')
    fake = StreamFake([[Delta(tool_calls=[cut], finish_reason="length")]])

    with pytest.raises(ModelOutputTruncatedError):
        await Agent(model=fake, tools=[add]).run("go")
    assert len(fake.calls) == 1


async def test_truncated_text_completes_with_length() -> None:
    fake = StreamFake([_text("partial answ", finish_reason="length")])

    final = await _final(Agent(model=fake))

    assert (final.answer, final.status) == ("partial answ", "completed")
    assert final.finish_reason == "length"


async def test_complete_tool_call_with_length_still_runs() -> None:
    fake = StreamFake(
        [_call("add", '{"a": 1, "b": 2}', finish_reason="length"), _text("3")]
    )
    session = Agent(model=fake, tools=[add]).start()

    assert await session.run("go") == "3"
    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    assert action.tool_results[0].content == "3"


async def test_truncation_on_the_forced_final_raises() -> None:
    fake = StreamFake([_call("add", '{"a": 1, "b": 2}'), _truncated()])

    with pytest.raises(ModelOutputTruncatedError):
        await Agent(model=fake, tools=[add], max_steps=1).run("go")
    assert len(fake.calls) == 2


# --------------------------------------------------------------------------- #
# Malformed turns and the step limit
# --------------------------------------------------------------------------- #
async def test_malformed_turn_does_not_use_a_step() -> None:
    fake = StreamFake([_malformed(), _text("ok")])

    final = await _final(Agent(model=fake, tools=[add], max_steps=1))

    assert (final.answer, final.status) == ("ok", "completed")
    assert len(fake.calls) == 2


async def test_malformed_turn_leaves_the_full_step_budget() -> None:
    fake = StreamFake(
        [
            _malformed(),
            _call("add", '{"a": 1, "b": 2}'),
            _call("add", '{"a": 2, "b": 2}', call_id="c2"),
            _text("forced"),
        ]
    )
    session = Agent(model=fake, tools=[add], max_steps=2).start()

    final = [event async for event in session.run("go", stream=True)][-1]

    assert isinstance(final, FinalEvent)
    assert final.status == "max_steps"
    assert sum(isinstance(step, ActionStep) for step in session.memory.steps) == 2
    assert len(fake.calls) == 4


async def test_three_malformed_turns_in_a_row_still_raise() -> None:
    fake = StreamFake([_malformed(), _malformed(), _malformed()])

    with pytest.raises(ModelError, match="3 consecutive turns") as caught:
        await Agent(model=fake, tools=[add]).run("go")

    assert type(caught.value) is ModelError
    assert len(fake.calls) == 3


# --------------------------------------------------------------------------- #
# provider_data must be JSON-serialisable
# --------------------------------------------------------------------------- #
async def test_non_json_provider_data_on_a_tool_turn() -> None:
    fake = StreamFake([_call("add", '{"a": 1, "b": 2}', provider_data={"x": object()})])
    session = Agent(model=fake, tools=[add]).start()

    with pytest.raises(ModelOutputError, match="object"):
        await session.run("go")
    assert [type(step) for step in session.memory.steps] == [TaskStep]


async def test_non_json_provider_data_on_the_final_text() -> None:
    fake = StreamFake([_text("answer", provider_data={"x": {1, 2}})])
    session = Agent(model=fake).start()

    with pytest.raises(ModelOutputError, match="set"):
        await session.run("go")
    assert not any(isinstance(step, FinalStep) for step in session.memory.steps)


async def test_non_json_provider_data_on_the_forced_final() -> None:
    fake = StreamFake(
        [
            _call("add", '{"a": 1, "b": 2}'),
            _text("forced", provider_data={"x": object()}),
        ]
    )

    with pytest.raises(ModelOutputError, match="object"):
        await Agent(model=fake, tools=[add], max_steps=1).run("go")


async def test_nested_json_provider_data_is_kept() -> None:
    data = {"anthropic": {"blocks": [{"type": "thinking", "signature": "s"}]}}
    fake = StreamFake(
        [_call("add", '{"a": 1, "b": 2}', provider_data=data), _text("done")]
    )
    session = Agent(model=fake, tools=[add]).start()

    assert await session.run("go") == "done"
    action = session.memory.steps[1]
    assert isinstance(action, ActionStep)
    assert action.model_message.provider_data == data


# --------------------------------------------------------------------------- #
# Usage and finish_reason on the FinalEvent
# --------------------------------------------------------------------------- #
async def test_unknown_usage_stays_none() -> None:
    final = await _final(Agent(model=StreamFake([_text("hi")])))
    assert final.usage is None


async def test_partial_usage_is_summed() -> None:
    fake = StreamFake(
        [
            _call("add", '{"a": 1, "b": 2}'),
            [Delta(content="done"), Delta(usage=Usage(3, 4))],
        ]
    )
    final = await _final(Agent(model=fake, tools=[add]))
    assert final.usage == Usage(3, 4)


async def test_interrupted_run_without_usage_reports_none() -> None:
    model = _InterruptsFirstCall()
    session = Agent(model=model).start()
    model.session = session

    final = [event async for event in session.run("x", stream=True)][-1]

    assert isinstance(final, FinalEvent)
    assert final.status == "interrupted"
    assert final.usage is None


@pytest.mark.parametrize("reason", ["stop", None])
async def test_finish_reason_on_a_text_answer(reason: Any) -> None:
    final = await _final(Agent(model=StreamFake([_text("hi", finish_reason=reason)])))
    assert final.finish_reason == reason


async def test_finish_reason_on_the_final_answer_path() -> None:
    fake = StreamFake(
        [_call("final_answer", '{"answer": "42"}', finish_reason="tool_calls")]
    )
    final = await _final(Agent(model=fake))

    assert final.answer == "42"
    assert final.finish_reason == "tool_calls"


async def test_finish_reason_on_the_forced_final() -> None:
    fake = StreamFake(
        [_call("add", '{"a": 1, "b": 2}'), _text("forced", finish_reason="length")]
    )
    final = await _final(Agent(model=fake, tools=[add], max_steps=1))
    assert final.finish_reason == "length"


# --------------------------------------------------------------------------- #
# The model check at construction
# --------------------------------------------------------------------------- #
def test_a_model_name_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match=r"^model must have a callable stream\(messages, tools\) method$",
    ):
        Agent(model="gpt-4o")  # type: ignore[arg-type]


class _NotCallableStream:
    stream = 1


class _GenerateOnly:
    async def generate(self, messages: Any, tools: Any = None) -> ChatMessage:
        raise NotImplementedError


@pytest.mark.parametrize("model", [_NotCallableStream(), _GenerateOnly()])
def test_a_model_without_a_callable_stream_is_rejected(model: object) -> None:
    with pytest.raises(TypeError, match="callable stream"):
        Agent(model=model)  # type: ignore[arg-type]


class _CoroutineStream:
    async def stream(self, messages: Any, tools: Any = ()) -> Any:
        return iter(())


def test_a_coroutine_stream_is_rejected_at_construction() -> None:
    with pytest.raises(TypeError, match="must be an async generator"):
        Agent(model=_CoroutineStream())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Performance budget
# --------------------------------------------------------------------------- #
def test_reassembly_and_provider_data_check_are_fast() -> None:
    deltas = [
        Delta(tool_calls=[ToolCallDelta(index, None, f"tool_{index}", '{"a": 1}')])
        for index in range(10)
    ]
    deltas.append(Delta(provider_data={"vendor": {"blocks": ["x"] * 10}}))

    timings = []
    for _ in range(100):
        started = time.perf_counter()
        _check_provider_data(agglomerate_deltas(deltas))
        timings.append(time.perf_counter() - started)

    assert statistics.median(timings) < 0.001
