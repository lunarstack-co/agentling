# Reference

Settings and signatures for Agentling's public API. Concepts are in the
[README](README.md); internals are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Agent

`Agent(model, tools=(), skills=(), ...)`

| Parameter | Default | What it does |
| --- | --- | --- |
| `model` | required | Any object with an async-generator `stream(messages, tools)`. |
| `tools` | `()` | Tools the model can call. `final_answer` is always added; a duplicate name raises `ValueError`. |
| `skills` | `()` | `Skill` objects, or paths to skill folders. |
| `instructions` | built-in prompt | The system prompt. The skill catalog is appended when there are skills. |
| `max_steps` | `15` | Model turns before a forced, tool-free answer. At least 1. Unparseable turns don't count. |
| `step_callbacks` | `()` | Called in order with each `ActionStep` before its `StepEvent`. Async ones are awaited; one that raises stops the run. |
| `parallel_tools` | `True` | Run a step's tool calls concurrently, unless one has `parallel_safe=False`. |
| `tool_timeout` | `None` | Seconds per tool call, for tools that don't set their own. A timeout becomes an error result. |
| `model_timeout` | `None` | Seconds to wait for each streamed chunk. Exceeding it raises `ModelTimeoutError`, which is retried if nothing had arrived. |
| `max_tool_output_chars` | `None` | Cap on each tool result, for tools that don't set their own. The middle is cut. |
| `redact_errors` | `False` | Log tool exceptions instead of showing their message to the model. `ToolCallError` messages are always shown. |
| `context_manager` | `None` | `messages -> messages`, applied before each model call, to trim or summarise. |
| `retry` | `RetryPolicy()` | How failed model turns are retried. `None` turns retries off. |

`agent.start()` returns a new `AgentSession`. `agent.run(...)` runs on a new
session and takes the same arguments as `session.run`.

## AgentSession

`session.run(task, *, stream=False, reset=True, max_steps=None)`

- `stream=False` returns an awaitable of the answer; `stream=True` returns an
  async generator of events.
- `reset=False` keeps the memory and adds `task` as the next user turn.
- `max_steps` overrides the agent's limit for this run.
- Raises `RuntimeError` if the session is already running, and a `ModelError`
  subclass if the model fails after retries.

`session.resume(*, stream=False, max_steps=None)` continues an interrupted or
failed run from memory, with no new task. Same return types as `run`. Raises
`ValueError("Nothing to resume")` when memory is empty or ends with an answer.

`session.interrupt()` stops the current run at its next step, chunk or retry
wait. It does nothing when no run is active.

`session.add_tools(tools, *, replace=False)` and `session.remove_tools(names)`
change the session's tools from the model's next call. Both are all or nothing:
they raise `ValueError` and change nothing if any tool is invalid, reserved
(`final_answer`, and `load_skill` when there are skills), a duplicate, or
unknown.

`session.memory` is the `Memory`; `session.is_running` is `True` while a run
or resume is in progress.

## Tool

`@tool` or `@tool(timeout=None, parallel_safe=True, max_output_chars=None,
validate_arguments=True)`.

| Attribute | Default | What it does |
| --- | --- | --- |
| `timeout` | `None` | Seconds for this tool; `None` uses the agent's `tool_timeout`. |
| `parallel_safe` | `True` | `False` makes any step that calls it run its tools in order. |
| `max_output_chars` | `None` | Cap for this tool; `None` uses the agent's `max_tool_output_chars`. |
| `validate_arguments` | `True` | `False` skips Agentling's argument check, for tools validated elsewhere. |

A tool name is letters, digits, `_` or `-`, up to 64 characters.

## RetryPolicy

| Parameter | Default | What it does |
| --- | --- | --- |
| `max_retries` | `2` | Retries after the first attempt. |
| `base_delay` | `0.5` | First delay in seconds; doubles each retry unless the backend sent `retry_after`. |
| `max_delay` | `30.0` | Cap on any delay, including a backend's `retry_after`. |
| `jitter` | `0.2` | Each delay is scaled by a random factor in `[1 - jitter, 1 + jitter]`. |
| `on_retry` | `None` | Called with `(attempt, max_retries, error, delay)` before each wait; awaited if it returns an awaitable. If it raises, the run stops. |

A turn is retried only when it raised a `ModelError` with `retryable=True`
before yielding anything.

## Errors

| Error | Raised when |
| --- | --- |
| `ModelError` and subclasses | The model failed. The table in the README's [Write a model adapter](README.md#write-a-model-adapter) lists each subclass and whether it's retried. |
| `ToolCallError` | A tool's arguments don't fit its schema, or a tool raises it. The model sees it as a result; it never reaches the caller. |
| `MemoryLoadError` | `Memory.load_json` or `from_dict` got an unknown version or a bad shape. |
| `ValueError`, `TypeError` | Bad configuration: an invalid tool, duplicate names, `max_steps < 1`, a model without `stream`. |

`AgentRunError`, `ToolExecutionError` and `SkillLoadError` are exported but not
raised yet.

## agentling.testing

Import it as `agentling.testing`; it isn't re-exported from `agentling`.

| Name | What it does |
| --- | --- |
| `ScriptedModel(turns)` | Replays one turn per `stream()` call: a `ChatMessage`, a list of `Delta`s, or an exception to raise. Records `calls` and `tools`. Raises `AssertionError("ScriptedModel has no more turns")` when the script runs out. |
| `check_model(factory, *, recorded=None, live=False, expects_tool_calls=True)` | Runs an adapter through a text turn and a tool turn and returns a `ConformanceReport` with `ok`, `rules`, `turns` and `raise_for_failures()`. Needs exactly one of `recorded` (a fixture path) or `live=True`. |
| `RecordingTransport(inner=None)` | An httpx transport that records each exchange, with API keys in headers redacted. |
| `write_fixture(path, exchanges)`, `read_fixture(path)` | Save and load recorded exchanges, one JSON object per line. |

Recorded mode and `RecordingTransport` need `httpx`; `ScriptedModel` and live
mode don't.
