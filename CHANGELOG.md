# Changelog

All notable changes to agentling are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [semantic versioning](https://semver.org/).

## [0.2.0] - Unreleased

agentling no longer ships a model adapter or depends on `openai`. Any object
with a `stream(messages, tools)` method that yields `Delta`s is a model, so an
app can use OpenAI, Anthropic, Gemini or a local server through an adapter of
its own. `examples/providers/openai_chat.py` is one to copy.

### Added

- `ToolDefinition`, the provider-neutral description of a tool that adapters
  render in their own format, and `Tool.definition` to get one.
- `reasoning`, `finish_reason` and `provider_data` on `Delta` and
  `ChatMessage`, and `cached_input_tokens` and `reasoning_tokens` on `Usage`.
  `provider_data` must be JSON-serialisable; the loop raises
  `ModelOutputError` when it isn't.
- `FinishReason`, and `FinalEvent.finish_reason` for the last model turn.
- `ModelError` subclasses for each kind of model failure
  (`ModelConnectionError`, `ModelRateLimitError`, `ModelServerError`,
  `ModelTimeoutError`, `ModelAuthError`, `ModelBadRequestError`,
  `ModelContextLengthError`), with `retryable`, `retry_after` and `status`.
- `RetryPolicy` and `Agent(retry=...)`. The loop retries a turn that fails
  with a retryable error before its first chunk, with backoff and jitter,
  honouring `retry_after` up to `max_delay`. `retry=None` turns retries off.
- `deltas_from_message`, for adapters whose backend doesn't stream.
- `AgentSession.resume()`, which continues an interrupted or failed run from
  memory without adding the task again, and `AgentSession.is_running`.
- `ModelOutputTruncatedError`, raised when the model's output limit cuts a tool
  call short. The loop no longer asks the model to fix such a call.
- Step callbacks can be `async`. Callbacks run in order and are awaited before
  the step's `StepEvent`. `StepCallback` is their type, and
  `AgentSession.step_callbacks` is a `list[StepCallback]`.
- A reference adapter for OpenAI-compatible endpoints in
  `examples/providers/openai_chat.py`. It needs `httpx`, which agentling
  itself doesn't depend on.

### Changed

- `model_timeout` bounds the wait for each streamed chunk, not the whole turn.
- A tool call that arrives without an id gets one
  (`agentling_call_<index>_<8 hex>`) instead of failing the turn.
- `ModelOutputError` is now a subclass of `ModelError`.
- `FinalEvent.usage` is `None` when the model never reported usage, instead of
  zero counts.
- `interrupt()` also ends a wait between retries.
- `interrupt()` while no run is active does nothing. It used to stop the next
  run before its first step and leave its task in memory.
- A session runs one task at a time: `run()` or `resume()` during a run raises
  `RuntimeError("Session is already running")`.
- `final_answer` no longer produces a `ToolCallEvent` or `ToolResultEvent`; the
  step's `StepEvent` is followed by the `FinalEvent`. The call is still
  recorded in memory.
- Turns whose tool calls can't be parsed no longer count towards `max_steps`.
- `run(..., stream=True)` is typed as an async generator, so it can be closed
  early with `contextlib.aclosing`, and `run()` without it as a coroutine, so
  it can be passed to `asyncio.create_task`.
- The answer forced at the step limit goes through `context_manager`, retries
  and interrupts like any other turn.
- `Agent()` raises `TypeError` straight away for a model without a usable
  `stream` method.
- Memory saves as format 2, which adds the new fields. Format 1 saves still
  load, but 0.1.x can't read format 2.
- The examples that need an API key run as
  `uv run python -m examples.cli_math_tutor` (and likewise for the others).

### Removed

- `OpenAIModel` and the `openai` dependency. The only runtime dependency is
  `pyyaml`.
- `ToolSpec` and `Tool.to_schema()`; use `Tool.definition`.
- `Model.generate`. `Model` has one method, `stream`, which takes
  `ToolDefinition`s.

### Fixed

- A slow consumer of `run(stream=True)` no longer receives a `CancelledError`
  caused by `model_timeout`.
- Retries no longer stack. agentling used to retry on top of the OpenAI SDK's
  own retries, up to nine attempts per request.
- Interrupting a run mid-stream now closes the adapter's stream.
- An interrupt during the last step stops the run instead of making one more
  model call for the step-limit answer.
- A step callback that raises on the step where the model called
  `final_answer` no longer loses the answer: it is in memory, and the run is
  not left looking resumable.

## [0.1.1] - 2026-10-29

Tools whose schemas come from somewhere else (an MCP server, an OpenAPI spec, a
plugin) now work, and a running session can gain or lose tools.

### Added

- `AgentSession.add_tools(tools, replace=False)` and
  `AgentSession.remove_tools(names)` change a live session's tools. The model
  sees the change from its next call. Each call is all or nothing, and
  `final_answer` (plus `load_skill` when skills are configured) can't be
  removed or replaced.
- `Tool.validate_arguments` (default `True`) and the matching
  `@tool(validate_arguments=...)` option. Set it to `False` when something else,
  such as a remote server, validates the arguments.
- `Tool.validate(arguments)`, a public way to check arguments without running
  the tool. `call()` now goes through it.
- `examples/cli_dynamic_tools.py`, an offline demo of a tool built from a JSON
  Schema being added and removed mid-conversation.

### Fixed

- A property whose `type` is a list, such as `["string", "null"]`, no longer
  crashes validation with `TypeError`. A value passes if it matches any listed
  type, and `"null"` matches `None`.
- A schema with no `properties` no longer rejects every argument.
- Extra arguments are rejected only when `additionalProperties` is `false`.
- Validation skips schema keywords it doesn't implement (`anyOf`, `$ref`,
  `format`, nested objects) instead of rejecting valid calls, and it only ever
  raises `ToolCallError`.

### Changed

- Tool names and schemas are checked wherever a tool is registered: in
  `Agent(...)`, `add_tools()` and skill loading, not only in `@tool`. A `Tool`
  subclass with an invalid name, or with `parameters` that isn't an object
  schema, now fails with `ValueError` when it is registered instead of later as
  a provider error.
- A `Tool` subclass whose schema doesn't set `"additionalProperties": false` now
  accepts extra arguments, as JSON Schema intends. `@tool` schemas set it, so
  they still reject extras.
- Loading a skill that declares a tool whose name is already taken by a
  different tool now returns an error to the model instead of silently keeping
  the existing tool.
- Docstrings now say which errors are reserved and not yet raised
  (`AgentRunError`, `ToolExecutionError`, `SkillLoadError`), and that provider
  errors reach the caller unwrapped.

## [0.1.0] - 2026-07-07

The first real release: the core framework plus a production-hardening pass.

### Added

- Async ReAct agent loop with a single streaming code path (`Agent` config and
  factory, `AgentSession` run state); blocking and streaming `run()`.
- Provider-neutral model layer and an OpenAI-compatible adapter with retries.
- `@tool` decorator with JSON Schema generation, argument validation, and
  per-tool metadata (`timeout`, `parallel_safe`, `max_output_chars`).
- Typed memory (`TaskStep` / `ActionStep` / `FinalStep`) with JSON persistence
  and load validation (`MemoryLoadError`).
- Streaming events, a `print_events` renderer, and a terminal run `status`
  (completed / interrupted / max_steps).
- Progressive-disclosure skills (`SKILL.md`) with a built-in `load_skill` tool.
- An exception hierarchy under `AgentlingError`, public API exports, and
  `__all__`.
- Timeouts (`tool_timeout`, `model_timeout`), prompt cancellation, sync tools
  run off the event loop, optional error redaction (`redact_errors`), cumulative
  token usage on the final event, and a `context_manager` hook for long runs.

[0.2.0]: https://github.com/lunarstack-co/agentling/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/lunarstack-co/agentling/releases/tag/v0.1.1
[0.1.0]: https://github.com/lunarstack-co/agentling/releases/tag/v0.1.0
