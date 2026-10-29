# Changelog

All notable changes to agentling are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [semantic versioning](https://semver.org/).

## [0.1.1] - Unreleased

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

[0.1.1]: https://github.com/lunarstack-co/agentling/releases/tag/v0.1.1
[0.1.0]: https://github.com/lunarstack-co/agentling/releases/tag/v0.1.0
