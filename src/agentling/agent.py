"""The agent loop.

Agent holds the configuration (model, tools, skills, settings) and creates
sessions. AgentSession holds one conversation's state (memory, interrupt flag,
its own tool set) and runs the ReAct loop. One Agent can therefore serve many
concurrent runs without their memory, interrupts or tools mixing.

AgentSession._loop does the work and yields Events; _run_stream wraps it so a
session runs one task at a time. run() and resume() return that stream when
stream=True, or drain it and return the final answer.

Only skill names and descriptions go into the system prompt. A skill's full
instructions and tools arrive when the model calls load_skill.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Iterable,
    Sequence,
)
from contextlib import aclosing
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, overload

from .errors import (
    ModelError,
    ModelOutputError,
    ModelOutputTruncatedError,
    ModelTimeoutError,
    ToolTimeoutError,
)
from .events import (
    Event,
    FinalEvent,
    StepEvent,
    TextDelta,
    ToolCallEvent,
    ToolResultEvent,
)
from .memory import ActionStep, FinalStep, Memory, Step, TaskStep, ToolResult
from .models import (
    ChatMessage,
    Delta,
    Model,
    ToolCall,
    ToolDefinition,
    Usage,
    agglomerate_deltas,
)
from .retry import DEFAULT_RETRY, RetryPolicy, retry_delay
from .skills import Skill
from .tools import FinalAnswerTool, Tool, ToolCallError, _check_tool, tool

DEFAULT_INSTRUCTIONS = (
    "You are a helpful agent. Use the available tools to gather information or "
    "take actions, thinking step by step. When you have the answer, call the "
    "final_answer tool (or simply reply with plain text)."
)

_LOOP_NUDGE = (
    " Note: you already made this exact tool call and got the same result. "
    "Try a different approach."
)

# How many consecutive unparseable model turns to tolerate before giving up.
_MAX_MALFORMED_RETRIES = 2

# How often a retry wait checks for an interrupt.
_INTERRUPT_POLL_SECONDS = 0.05

_TRUNCATED_TOOL_CALL = (
    "Model output was cut off by its length limit while writing a tool call."
)

logger = logging.getLogger("agentling")

# Called with each ActionStep once it is recorded; async callbacks are awaited.
StepCallback = Callable[[Step], Awaitable[None] | None]


class Agent:
    """Configuration shared by every run, and a factory for sessions.

    Holds no per-run state, so one Agent can serve many concurrent runs. Call
    start() for a session, or run() for a one-off run.

    Args:
        model: Any object with an async-generator stream(messages, tools).
        tools: Tools the model can call. final_answer is always added, and a
            duplicate name raises ValueError.
        skills: Skill objects, or paths to skill folders.
        instructions: The system prompt. Defaults to DEFAULT_INSTRUCTIONS;
            the skill catalog is appended to it.
        max_steps: Model turns allowed before a forced, tool-free final
            answer. Turns that fail to parse don't count.
        step_callbacks: Called in order with each ActionStep before its
            StepEvent; async ones are awaited, and one that raises stops the run.
            They don't run for a plain-text final answer. Use FinalEvent to
            observe run completion.
        parallel_tools: Run a step's tool calls concurrently, unless any of
            them has parallel_safe=False.
        tool_timeout: Seconds allowed per tool call, for tools that don't set
            their own. None means no limit.
        model_timeout: Seconds to wait for each streamed chunk, not the
            whole turn. None means no limit.
        max_tool_output_chars: Cap on each tool result, for tools that don't
            set their own. The middle is cut.
        redact_errors: Log tool exceptions instead of showing their message to
            the model. ToolCallError messages are always shown.
        context_manager: Called with the full message list before each model
            call; the list it returns is what the model sees.
        retry: How failed model turns are retried. None turns retries off.
    """

    def __init__(
        self,
        model: Model,
        tools: Sequence[Tool] = (),
        skills: Sequence[Skill | str | Path] = (),
        instructions: str | None = None,
        max_steps: int = 15,
        step_callbacks: Sequence[StepCallback] = (),
        parallel_tools: bool = True,
        tool_timeout: float | None = None,
        model_timeout: float | None = None,
        max_tool_output_chars: int | None = None,
        redact_errors: bool = False,
        context_manager: Callable[[list[ChatMessage]], list[ChatMessage]] | None = None,
        retry: RetryPolicy | None = DEFAULT_RETRY,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        # Catch a model name or an old-style object here, not on the first run.
        stream = getattr(model, "stream", None)
        if not callable(stream):
            raise TypeError("model must have a callable stream(messages, tools) method")
        if inspect.iscoroutinefunction(stream):
            raise TypeError(
                "model.stream must be an async generator (an async def that "
                "yields Deltas), not a coroutine function"
            )

        self.model = model
        self.max_steps = max_steps
        self.parallel_tools = parallel_tools
        self.tool_timeout = tool_timeout
        self.model_timeout = model_timeout
        self.retry = retry
        self.max_tool_output_chars = max_tool_output_chars
        self.redact_errors = redact_errors
        self.context_manager = context_manager
        # Each session copies these and may add its own.
        self.step_callbacks = list(step_callbacks)

        # Fail on a duplicate name rather than let one tool shadow another. Each
        # session gets its own copy of this set.
        base_tools: dict[str, Tool] = {}
        for base_tool in (*tools, FinalAnswerTool()):
            _check_tool(base_tool)
            if base_tool.name in base_tools:
                raise ValueError(f"Duplicate tool name: {base_tool.name!r}")
            base_tools[base_tool.name] = base_tool
        self._base_tools = base_tools

        self.skills: dict[str, Skill] = {
            skill.name: skill for skill in (_as_skill(entry) for entry in skills)
        }
        self.instructions = instructions or DEFAULT_INSTRUCTIONS
        if self.skills:
            # A caller's tool named load_skill would hide the built-in one.
            if "load_skill" in self._base_tools:
                raise ValueError(
                    "'load_skill' is a reserved tool name when skills are "
                    "configured; rename the conflicting tool."
                )
            self.instructions += _skill_catalog(self.skills.values())

    def start(self) -> AgentSession:
        """Create a fresh session with its own memory, tools, and interrupt token."""

        return AgentSession(self)

    @overload
    def run(
        self,
        task: str,
        *,
        stream: Literal[False] = False,
        reset: bool = True,
        max_steps: int | None = None,
    ) -> Coroutine[Any, Any, str]: ...

    @overload
    def run(
        self,
        task: str,
        *,
        stream: Literal[True],
        reset: bool = True,
        max_steps: int | None = None,
    ) -> AsyncGenerator[Event, None]: ...

    def run(
        self,
        task: str,
        *,
        stream: bool = False,
        reset: bool = True,
        max_steps: int | None = None,
    ) -> Coroutine[Any, Any, str] | AsyncGenerator[Event, None]:
        """Run a task on a new session and discard it.

        Returns what AgentSession.run() returns. Concurrent calls each get their
        own session. For a multi-turn
        conversation, or to read the memory afterwards, use start().
        """

        session = self.start()
        if stream:
            return session.run(task, stream=True, reset=reset, max_steps=max_steps)
        return session.run(task, stream=False, reset=reset, max_steps=max_steps)


class AgentSession:
    """One conversation's mutable state plus the agent loop.

    A session owns its own Memory, interrupt token, and tool set (a copy of the
    Agent's base tools, so tools added here never leak into another session).
    Create one with Agent.start(). Change tools with add_tools()/remove_tools().
    """

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.memory = Memory()
        self.step_callbacks = list(agent.step_callbacks)
        self._interrupt = asyncio.Event()
        self._running = False

        self.tools: dict[str, Tool] = dict(agent._base_tools)
        if agent.skills:
            load_skill = self._build_load_skill_tool()
            self.tools[load_skill.name] = load_skill
        self._tool_definitions: list[ToolDefinition] = []
        self._rebuild_definitions()

    def add_tools(self, tools: Iterable[Tool], *, replace: bool = False) -> None:
        """Add tools to this session, visible from the model's next call.

        All or nothing: raises ValueError and adds nothing if any tool has a bad
        name or schema, a reserved name, or a name already taken (unless
        `replace` is True, which swaps it in place).
        """

        new_tools = list(tools)
        reserved = self._reserved_names()
        seen: set[str] = set()
        for new_tool in new_tools:
            _check_tool(new_tool)
            if new_tool.name in reserved:
                raise ValueError(f"{new_tool.name!r} is a reserved tool name.")
            if new_tool.name in seen or (new_tool.name in self.tools and not replace):
                raise ValueError(f"Duplicate tool name: {new_tool.name!r}")
            seen.add(new_tool.name)

        if not new_tools:
            return
        for new_tool in new_tools:
            self.tools[new_tool.name] = new_tool
        self._rebuild_definitions()

    def remove_tools(self, names: Iterable[str]) -> None:
        """Remove tools from this session, from the model's next call.

        All or nothing: raises ValueError and removes nothing if any name is
        reserved or unknown.
        """

        targets = list(dict.fromkeys(names))
        reserved = self._reserved_names()
        for name in targets:
            if name in reserved:
                raise ValueError(f"{name!r} is a reserved tool name.")
        for name in targets:
            if name not in self.tools:
                raise ValueError(f"Unknown tool: {name!r}")

        if not targets:
            return
        for name in targets:
            del self.tools[name]
        self._rebuild_definitions()

    def _reserved_names(self) -> frozenset[str]:
        if self.agent.skills:
            return frozenset({"final_answer", "load_skill"})
        return frozenset({"final_answer"})

    def _rebuild_definitions(self) -> None:
        # A new list, so one already handed to the model is never mutated.
        self._tool_definitions = [t.definition for t in self.tools.values()]

    @property
    def is_running(self) -> bool:
        """Whether a run or resume is in progress on this session."""

        return self._running

    @overload
    def run(
        self,
        task: str,
        *,
        stream: Literal[False] = False,
        reset: bool = True,
        max_steps: int | None = None,
    ) -> Coroutine[Any, Any, str]: ...

    @overload
    def run(
        self,
        task: str,
        *,
        stream: Literal[True],
        reset: bool = True,
        max_steps: int | None = None,
    ) -> AsyncGenerator[Event, None]: ...

    def run(
        self,
        task: str,
        *,
        stream: bool = False,
        reset: bool = True,
        max_steps: int | None = None,
    ) -> Coroutine[Any, Any, str] | AsyncGenerator[Event, None]:
        """Run a task in this session.

        Returns an awaitable of the final answer, or with stream=True an async
        generator of Events. reset=False keeps the existing memory and adds the
        task as a new user turn. Raises RuntimeError if the session is already
        running, and a ModelError subclass if the model fails (after any
        retries), times out, or keeps sending unparseable tool calls.
        """

        events = self._run_stream(task, reset=reset, max_steps=max_steps)
        if stream:
            return events
        return self._drain(events)

    @overload
    def resume(
        self, *, stream: Literal[False] = False, max_steps: int | None = None
    ) -> Coroutine[Any, Any, str]: ...

    @overload
    def resume(
        self, *, stream: Literal[True], max_steps: int | None = None
    ) -> AsyncGenerator[Event, None]: ...

    def resume(
        self, *, stream: bool = False, max_steps: int | None = None
    ) -> Coroutine[Any, Any, str] | AsyncGenerator[Event, None]:
        """Continue an interrupted or failed run from memory, with no new task.

        Same return types as run(); max_steps counts from this call. Raises
        ValueError if memory is empty or already ends with a final answer.
        """

        steps = self.memory.steps
        if not steps or isinstance(steps[-1], FinalStep):
            raise ValueError("Nothing to resume")

        events = self._run_stream(None, reset=False, max_steps=max_steps)
        if stream:
            return events
        return self._drain(events)

    async def _drain(self, events: AsyncIterator[Event]) -> str:
        """Consume the event stream and return the final answer."""

        answer = ""
        async for event in events:
            if isinstance(event, FinalEvent):
                answer = event.answer
        return answer

    async def _run_stream(
        self, task: str | None, *, reset: bool, max_steps: int | None
    ) -> AsyncGenerator[Event, None]:
        """Run the loop, refusing to start while another run is in progress.

        The flag is set on the first iteration rather than when run() is
        called, so a stream that is never iterated never holds the session.
        """

        if self._running:
            raise RuntimeError(
                "Session is already running. If a previous run(..., stream=True) "
                "was abandoned, close it with contextlib.aclosing."
            )
        self._running = True
        try:
            run = self._loop(task, reset=reset, max_steps=max_steps)
            async with aclosing(run) as events:
                async for event in events:
                    yield event
        finally:
            self._running = False

    async def _loop(
        self, task: str | None, *, reset: bool, max_steps: int | None
    ) -> AsyncGenerator[Event, None]:
        """The core loop: drive the model/tool cycle, yielding Events.

        A task of None resumes: memory is kept and no TaskStep is added.
        """

        # An interrupt requested while no run was active belongs to no run.
        self._interrupt.clear()

        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be at least 1.")

        if task is not None:
            if reset:
                self.memory.reset()
            self.memory.add(TaskStep(task=task))

        limit = self.agent.max_steps if max_steps is None else max_steps

        # The previous step's calls, to spot an exact repeat.
        previous_signature: tuple[tuple[str, str], ...] | None = None

        # Sent after unparseable output so the model can retry; cleared once a
        # turn parses.
        correction_notes: list[ChatMessage] = []
        malformed_retries = 0
        # Stays None unless the model reports usage, so "unknown" isn't "zero".
        total_usage: Usage | None = None
        # Only turns that parse count; a correction retry doesn't use a step.
        taken = 0

        while taken < limit:
            if self._interrupt.is_set():
                yield self._interrupted(total_usage)
                return

            started = time.monotonic()
            messages = self.memory.to_messages(self.agent.instructions)
            messages.extend(correction_notes)
            if self.agent.context_manager is not None:
                messages = self.agent.context_manager(messages)

            # Checking the interrupt per chunk lets a long stream stop early.
            deltas: list[Delta] = []
            async with aclosing(
                self._stream_turn(messages, self._tool_definitions)
            ) as turn:
                async for delta in turn:
                    if delta.content:
                        yield TextDelta(text=delta.content)
                    deltas.append(delta)
                    if self._interrupt.is_set():
                        break

            if self._interrupt.is_set():
                yield self._interrupted(total_usage)
                return

            # A malformed tool call gets a correction and another try, up to
            # _MAX_MALFORMED_RETRIES times in a row.
            try:
                response = agglomerate_deltas(deltas)
            except ModelOutputError as exc:
                # The model can't fix a call its output limit cut short.
                if _hit_length_limit(deltas):
                    raise ModelOutputTruncatedError(_TRUNCATED_TOOL_CALL) from exc
                malformed_retries += 1
                if malformed_retries > _MAX_MALFORMED_RETRIES:
                    raise ModelError(
                        f"Model produced unparseable tool calls on "
                        f"{malformed_retries} consecutive turns: {exc}"
                    ) from exc
                correction_notes.append(
                    ChatMessage(
                        role="user",
                        content=(
                            f"Your previous response could not be parsed: {exc} "
                            "Reply again with valid tool-call arguments as a JSON "
                            "object."
                        ),
                    )
                )
                continue

            # An adapter bug, not something the model can fix, so it is raised
            # rather than sent back as a correction.
            _check_provider_data(response)
            correction_notes.clear()
            malformed_retries = 0
            taken += 1
            total_usage = _add_usage(total_usage, response.usage)

            # A reply with no tool calls is the answer.
            if not response.tool_calls:
                self.memory.add(FinalStep(answer=response.content))
                yield FinalEvent(
                    answer=response.content,
                    usage=total_usage,
                    finish_reason=response.finish_reason,
                )
                return

            signature = tuple(
                (tc.name, json.dumps(tc.arguments, sort_keys=True))
                for tc in response.tool_calls
            )
            looping = signature == previous_signature
            previous_signature = signature

            # final_answer is how a run ends, not a tool call to show. It is
            # still recorded in the ActionStep so memory replays correctly.
            for tool_call in response.tool_calls:
                if tool_call.name != "final_answer":
                    yield ToolCallEvent(tool_call=tool_call)

            # One tool that isn't parallel_safe makes the whole step run in order.
            step_tools = [self.tools.get(tc.name) for tc in response.tool_calls]
            run_parallel = self.agent.parallel_tools and all(
                t is None or t.parallel_safe for t in step_tools
            )
            if run_parallel:
                results: list[ToolResult] = await asyncio.gather(
                    *(self._execute_tool(tc) for tc in response.tool_calls)
                )
            else:
                results = [await self._execute_tool(tc) for tc in response.tool_calls]

            # Same calls as the last step: nudge the model to change approach.
            if looping:
                results = [replace(r, content=r.content + _LOOP_NUDGE) for r in results]

            for result in results:
                if result.name != "final_answer":
                    yield ToolResultEvent(result=result)

            action = ActionStep(
                model_message=response,
                tool_results=results,
                usage=response.usage,
                duration=time.monotonic() - started,
            )
            self.memory.add(action)
            # A successful final_answer call ends the run. Its FinalStep is
            # written before the callbacks run, so one that raises can't leave
            # a finished run looking resumable.
            final = next(
                (r for r in results if r.name == "final_answer" and not r.is_error),
                None,
            )
            if final is not None:
                self.memory.add(FinalStep(answer=final.content))
            await self._notify(action)
            yield StepEvent(step=action)

            if final is not None:
                yield FinalEvent(
                    answer=final.content,
                    usage=total_usage,
                    finish_reason=response.finish_reason,
                )
                return

        # Step limit reached: force one tool-free answer, unless interrupted
        # during the last step.
        if self._interrupt.is_set():
            yield self._interrupted(total_usage)
            return

        messages = self.memory.to_messages(self.agent.instructions)
        messages.append(
            ChatMessage(
                role="user",
                content="Step limit reached. Give your best final answer now.",
            )
        )
        if self.agent.context_manager is not None:
            messages = self.agent.context_manager(messages)

        deltas = []
        async with aclosing(self._stream_turn(messages, ())) as turn:
            async for delta in turn:
                if delta.content:
                    yield TextDelta(text=delta.content)
                deltas.append(delta)
                if self._interrupt.is_set():
                    break

        if self._interrupt.is_set():
            yield self._interrupted(total_usage)
            return

        try:
            response = agglomerate_deltas(deltas)
        except ModelOutputError as exc:
            if _hit_length_limit(deltas):
                raise ModelOutputTruncatedError(_TRUNCATED_TOOL_CALL) from exc
            raise ModelError(
                f"Model produced unparseable output on the forced final answer: {exc}"
            ) from exc

        _check_provider_data(response)
        total_usage = _add_usage(total_usage, response.usage)
        self.memory.add(FinalStep(answer=response.content))
        yield FinalEvent(
            answer=response.content,
            usage=total_usage,
            status="max_steps",
            finish_reason=response.finish_reason,
        )

    async def _stream_turn(
        self, messages: list[ChatMessage], tools: Sequence[ToolDefinition]
    ) -> AsyncGenerator[Delta, None]:
        """Stream one model turn, retrying it if it fails before any output.

        model_timeout bounds the wait for each next Delta, never the time the
        caller spends between them. A turn is retried, under the agent's
        RetryPolicy, only when it raises a retryable ModelError before its first
        Delta: once text has been shown, a retry would repeat it. An interrupt
        during the wait between attempts ends the turn with no further attempt.
        """

        policy = self.agent.retry
        attempt = 0
        while True:
            received = False
            stream = aiter(self.agent.model.stream(messages, tools))
            try:
                while True:
                    try:
                        async with asyncio.timeout(self.agent.model_timeout):
                            delta = await anext(stream)
                    except StopAsyncIteration:
                        return
                    except TimeoutError as exc:
                        raise ModelTimeoutError(
                            f"Model sent nothing for {self.agent.model_timeout}s."
                        ) from exc
                    received = True
                    # Outside the timeout, so a slow consumer is never charged.
                    yield delta
            except ModelError as exc:
                if (
                    received
                    or not exc.retryable
                    or policy is None
                    or attempt >= policy.max_retries
                ):
                    raise
                delay = retry_delay(policy, attempt, exc)
                attempt += 1
                if policy.on_retry is not None:
                    outcome = policy.on_retry(attempt, policy.max_retries, exc, delay)
                    if inspect.isawaitable(outcome):
                        await outcome
                await self._sleep_unless_interrupted(delay)
                if self._interrupt.is_set():
                    return
            finally:
                # Frees the adapter's connection when a timeout, retry or
                # early stop abandons the stream.
                aclose = getattr(stream, "aclose", None)
                if aclose is not None:
                    await aclose()

    async def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Run one tool call, turning any failure into an error observation."""

        tool = self.tools.get(tool_call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=(
                    f"Unknown tool {tool_call.name!r}. Available: {list(self.tools)}"
                ),
                is_error=True,
                error_kind="unknown_tool",
            )
        timeout = tool.timeout if tool.timeout is not None else self.agent.tool_timeout
        try:
            if timeout is not None:
                output = await asyncio.wait_for(tool.call(tool_call.arguments), timeout)
            else:
                output = await tool.call(tool_call.arguments)
        except TimeoutError:
            # A slow tool becomes an observation. CancelledError is a
            # BaseException, so a real cancellation still stops the run.
            err = ToolTimeoutError(
                f"tool {tool_call.name!r} exceeded its {timeout}s timeout"
            )
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"{type(err).__name__}: {err}",
                is_error=True,
                error_kind="timeout",
            )
        except ToolCallError as exc:
            # Shown in full even with redact_errors: the model needs it to fix
            # the call.
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"{type(exc).__name__}: {exc}",
                is_error=True,
                error_kind="validation",
            )
        except Exception as exc:
            # The message may hold secrets, so redact_errors logs it instead of
            # showing it to the model.
            if self.agent.redact_errors:
                logger.exception("Tool %r raised", tool_call.name)
                content = (
                    f"{type(exc).__name__} while running {tool_call.name!r} "
                    "(details suppressed)."
                )
            else:
                content = f"{type(exc).__name__}: {exc}"
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=content,
                is_error=True,
                error_kind="execution",
            )

        content = str(output)
        limit = (
            tool.max_output_chars
            if tool.max_output_chars is not None
            else self.agent.max_tool_output_chars
        )
        if limit is not None:
            content = _truncate_middle(content, limit)
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content=content,
            is_error=False,
        )

    def _build_load_skill_tool(self) -> Tool:
        """Build the built-in load_skill tool, bound to this session."""

        @tool
        def load_skill(name: str) -> str:
            """Load a skill's full instructions and enable any tools it provides.

            Args:
                name: The name of the skill to load, taken from the catalog in
                    the system prompt.
            """

            skill = self.agent.skills.get(name)
            if skill is None:
                raise ToolCallError(
                    f"Unknown skill {name!r}. Available: {sorted(self.agent.skills)}"
                )

            loaded = skill.load_tools()
            # Reloading a skill returns the same objects; skip those.
            new_tools = [t for t in loaded if self.tools.get(t.name) is not t]
            for skill_tool in new_tools:
                if skill_tool.name in self.tools:
                    raise ToolCallError(
                        f"Skill {name!r} declares tool {skill_tool.name!r}, "
                        "which conflicts with an existing tool."
                    )
            self.add_tools(new_tools)

            body = skill.instructions
            if loaded:
                names = ", ".join(skill_tool.name for skill_tool in loaded)
                body += f"\n\nTools now available: {names}."
            return body

        return load_skill

    async def _sleep_unless_interrupted(self, delay: float) -> None:
        """Wait `delay` seconds, or less if the session is interrupted.

        Polls the flag instead of awaiting the Event: Event.wait() binds the
        Event to one event loop, and interrupt() must stay a plain flag that
        works whichever loop or task sets it.
        """

        loop = asyncio.get_running_loop()
        deadline = loop.time() + delay
        while not self._interrupt.is_set():
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            await asyncio.sleep(min(remaining, _INTERRUPT_POLL_SECONDS))

    async def _notify(self, step: Step) -> None:
        """Run the step callbacks in order, awaiting the async ones."""

        for callback in self.step_callbacks:
            outcome = callback(step)
            if inspect.isawaitable(outcome):
                await outcome

    def _interrupted(self, usage: Usage | None) -> FinalEvent:
        """Build the event that ends an interrupted run."""

        return FinalEvent(answer="Run interrupted.", usage=usage, status="interrupted")

    def interrupt(self) -> None:
        """Ask the current run to stop at its next step, chunk or retry wait.

        The run ends with a FinalEvent whose status is "interrupted", and
        resume() carries on from where it stopped.
        Has no effect when no run is active.
        """

        self._interrupt.set()


def _truncate_middle(text: str, limit: int) -> str:
    """Cut the middle of a long string, keeping the start and end.

    Tool output often has a summary at the top and the result at the bottom.
    """

    if len(text) <= limit:
        return text
    keep = limit // 2
    if keep <= 0:
        # Too small to keep both ends, and text[-0:] would return everything.
        return text[:limit]
    omitted = len(text) - 2 * keep
    return f"{text[:keep]}\n... [{omitted} characters omitted] ...\n{text[-keep:]}"


def _add_usage(total: Usage | None, usage: Usage | None) -> Usage | None:
    """Add a turn's usage to the run total, keeping None until any is seen."""

    if usage is None:
        return total
    return usage if total is None else total + usage


def _hit_length_limit(deltas: Sequence[Delta]) -> bool:
    """Whether the turn's last finish_reason says it hit the output limit."""

    reasons = [delta.finish_reason for delta in deltas if delta.finish_reason]
    return bool(reasons) and reasons[-1] == "length"


def _check_provider_data(message: ChatMessage) -> None:
    """Fail now if provider_data couldn't be saved with the memory later."""

    if message.provider_data is None:
        return
    try:
        json.dumps(message.provider_data)
    except (TypeError, ValueError) as exc:
        raise ModelOutputError(
            f"provider_data is not JSON-serialisable: {exc}"
        ) from exc


def _as_skill(entry: Skill | str | Path) -> Skill:
    """Coerce a skill entry (a Skill, or a path to a skill folder) into a Skill."""

    return entry if isinstance(entry, Skill) else Skill.from_path(entry)


def _skill_catalog(skills: Iterable[Skill]) -> str:
    """Render the skill names and descriptions appended to the system prompt."""

    lines = "\n".join(f"- {skill.name}: {skill.description}" for skill in skills)
    return (
        "\n\nYou can load skills: focused instruction sets for particular kinds "
        "of task. When the task matches one, call load_skill(name) to load its "
        "full instructions and any tools it provides before continuing. "
        "Available skills:\n" + lines
    )
