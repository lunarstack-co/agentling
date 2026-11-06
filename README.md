<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://agentling.dev/brand/logo/agentling-horizontal-white.svg">
    <img alt="Agentling" src="https://agentling.dev/brand/logo/agentling-horizontal-black.svg" height="56">
  </picture>
</p>

# Agentling

[![PyPI](https://img.shields.io/pypi/v/agentling.svg)](https://pypi.org/project/agentling/)
[![CI](https://github.com/lunarstack-co/agentling/actions/workflows/ci.yml/badge.svg)](https://github.com/lunarstack-co/agentling/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A small agent engine for Python: the tool-calling loop, typed memory and
streamed events, with any model you bring. It's for developers who want to own
the code around their agent instead of adopting a framework.

> **Status: alpha (0.x).** The API may change before 1.0.

## Why Agentling

An agent is a loop: ask the model, run the tools it asks for, record what
happened, repeat until it answers. Agentling is that loop and nothing else. You
bring the model (any object with a `stream()` method), the tools (plain
functions) and whatever sits around the loop, such as a UI, approvals or a
server.

Over a hand-written loop you get retries that never repeat a tool call,
interrupt and resume, tool errors fed back to the model instead of crashing the
run, memory you can save and reload, and a checker that proves a model adapter
follows the contract. The loop is one file, [`agent.py`](src/agentling/agent.py).

## Install

```bash
pip install agentling
```

Or `uv add agentling`. Requires Python 3.11 or newer; the only runtime
dependency is `pyyaml`. Agentling is async: call it from async code or with
`asyncio.run`.

## Quickstart

This runs with no API key. `ScriptedModel` plays the model's part from a script:
first it calls the `add` tool, then it answers.

```python
import asyncio

from agentling import Agent, ChatMessage, ToolCall, tool
from agentling.testing import ScriptedModel


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b


model = ScriptedModel(
    [
        ChatMessage("assistant", tool_calls=[ToolCall("c1", "add", {"a": 19, "b": 23})]),
        ChatMessage("assistant", "19 + 23 = 42."),
    ]
)


async def main() -> None:
    agent = Agent(model=model, tools=[add])
    print(await agent.run("What is 19 + 23?"))


asyncio.run(main())
```

To use a real model, swap `ScriptedModel` for an adapter. Agentling ships none;
copy one from [`examples/providers/`](examples/providers/) into your app (they
need `httpx`). With the OpenAI-compatible adapter:

```python
# not executed: needs an OpenAI API key
from examples.providers.openai_chat import OpenAIChatModel

agent = Agent(model=OpenAIChatModel("gpt-4o-mini"), tools=[add])  # reads OPENAI_API_KEY
```

Snippets in this README import the adapters from where they sit in this
repository.

## Core concepts

### Agents and sessions

An `Agent` is configuration: the model, tools, skills and settings. It holds no
per-run state, so build one and share it. An `AgentSession` holds one
conversation: its memory, its interrupt flag and its own copy of the tools.

`agent.run(task)` runs on a new session and discards it, so concurrent calls on
one agent never share memory. Hold a session with `agent.start()` for
multi-turn work, to read the memory, or to interrupt a run. A session runs one
task at a time; a second `run()` while `session.is_running` raises
`RuntimeError`.

```python
# not executed: a fragment; `agent` comes from earlier examples
session = agent.start()
await session.run("First question")
await session.run("A follow-up", reset=False)  # keeps the earlier turn
print(session.memory.steps)
```

Every setting is listed in [REFERENCE.md](REFERENCE.md#agent).

### Tools

`@tool` turns a function into a tool. The name becomes the tool name, the
docstring summary its description, and the type hints plus a Google-style
`Args:` section the JSON Schema the model sees. Sync functions run in a worker
thread so they can't block the loop.

```python
from agentling import tool


@tool
async def search(query: str, limit: int = 5) -> str:
    """Search the docs and return the top matches.

    Args:
        query: What to search for.
        limit: How many results to return.
    """
    ...
```

Supported hints are `str`, `int`, `float`, `bool`, `list[...]`, `dict[...]`,
`X | None` (the argument may be omitted) and `Literal[...]` (an enum).
Parameters without a default are required. `*args`, `**kwargs` and
positional-only parameters raise `TypeError` when the tool is built, because a
tool is called with a JSON object of named arguments.

When the model sends arguments that don't fit the schema, or a tool raises, the
loop shows the error to the model as the tool's result and the run carries on.
For tools with state or a ready-made schema, subclass `Tool`; see
[`cli_dynamic_tools.py`](examples/cli_dynamic_tools.py).

`tool_timeout` stops the loop waiting on a tool, but a sync tool already
running in its thread finishes in the background. Prefer async tools when
timeouts matter.

### Running and streaming

`run()` returns the final answer. With `stream=True` it returns an async
generator of events from the same loop:

```python
# not executed: a fragment; `agent` comes from earlier examples
from agentling import FinalEvent, TextDelta, ToolCallEvent

async for event in agent.run("Summarize this.", stream=True):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, ToolCallEvent):
        print(f"\n[calling {event.tool_call.name}]")
    elif isinstance(event, FinalEvent):
        print(f"\nDone: {event.answer}")
```

`print_events(agent.run(task, stream=True))` prints the stream as it arrives and
returns the answer. If you stop iterating early, close the stream with
`contextlib.aclosing`, or the session counts as running until Python collects
it. The event types and their order are in
[ARCHITECTURE.md](ARCHITECTURE.md#events).

### Memory, interruption and resume

A session records the run as typed steps (`TaskStep`, `ActionStep`,
`FinalStep`). `session.memory.dump_json()` saves them and `Memory.load_json()`
restores them, in this process or another.

`session.interrupt()` stops the run at its next step, streamed chunk or retry
wait. The run ends with a `FinalEvent` whose status is `"interrupted"` and
keeps its memory, and `session.resume()` carries on without adding a new user
turn:

```python
# not executed: a fragment; `agent` comes from earlier examples
session = agent.start()
job = asyncio.create_task(session.run("Long job"))
await asyncio.sleep(0)  # let the run start
session.interrupt()
await job  # "Run interrupted."
await session.resume()  # continues from memory
```

An `interrupt()` while no run is active does nothing. `resume()` raises
`ValueError("Nothing to resume")` when memory is empty or already ends with an
answer. From another thread, call `loop.call_soon_threadsafe(session.interrupt)`.

### Skills

A skill is a folder with a `SKILL.md`: YAML frontmatter with a `name` and
`description`, then a markdown body of instructions. The model sees only the
names and descriptions until it calls the built-in `load_skill(name)`, which
returns the body. A large skill library therefore costs little prompt space.

```python
# not executed: needs an OpenAI API key and a read_file tool
agent = Agent(
    model=OpenAIChatModel("gpt-4o-mini"),
    tools=[read_file],
    skills=["examples/skills/code-reviewer"],
)
```

A skill can list tool entry points (`tools: [my_package.lint_tools:run_ruff]`)
that are imported and registered when it loads. Importing runs that module's
code, so load skills only from sources you trust; see [SECURITY.md](SECURITY.md).
[`examples/skills/code-reviewer`](examples/skills/code-reviewer) is a complete
skill.

### Models

Any object with this method is a model:

```python
# not executed: the shape of the protocol; see src/agentling/models.py
class Model(Protocol):
    def stream(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition] = (),
    ) -> AsyncIterator[Delta]: ...
```

[`examples/providers/`](examples/providers/) has four adapters to copy:
`openai_chat.py` for OpenAI and compatible servers (LM Studio, vLLM, Ollama's
`/v1`, OpenRouter, Gemini's compatibility endpoint), `anthropic_messages.py`
for Claude, `ollama_native.py` for Ollama's own API, and `gemini_native.py`
for Gemini's native API. Point `base_url` at
your endpoint:

```python
from examples.providers.openai_chat import OpenAIChatModel

model = OpenAIChatModel(
    "llama-3.1-70b",
    base_url="http://localhost:8000/v1",
    api_key="not-needed-locally",
)
```

Adapters raise `ModelError` subclasses and never retry; the loop retries under
the agent's `RetryPolicy`.

### Write a model adapter

An adapter is one class with one method. It turns Agentling's messages into
your backend's request, and your backend's response into `Delta`s. Start from
the reference adapter closest to your backend:

- [`openai_chat.py`](examples/providers/openai_chat.py): OpenAI chat
  completions and every server that copies it. Handles usage via
  `stream_options`, finish reasons, and tool calls with missing ids.
- [`anthropic_messages.py`](examples/providers/anthropic_messages.py): the
  Anthropic Messages API. Lifts the system prompt, groups tool results into
  one user message, and replays thinking blocks through `provider_data`.
- [`ollama_native.py`](examples/providers/ollama_native.py): Ollama's
  `/api/chat`. Reads newline-delimited JSON, whole tool calls, and usage from
  the final chunk.
- [`gemini_native.py`](examples/providers/gemini_native.py): Gemini's native
  `streamGenerateContent` API. Maps function calls, cached and thinking tokens,
  and replays model parts with their thought signatures through `provider_data`.
  Reads `GEMINI_API_KEY` (or `GOOGLE_API_KEY`); `base_url` includes `/v1beta`.
  Use a current model id (for example `gemini-3.8-flash` in examples); older ids
  such as `gemini-2.5-flash` may return 404 for newly issued API keys while Google
  rotates model availability.

**What you yield.** Each `Delta` carries any of these, all optional:

| Field | Meaning |
| --- | --- |
| `content` | A chunk of answer text. |
| `reasoning` | A chunk of visible reasoning, kept apart from the answer. |
| `tool_calls` | `ToolCallDelta` fragments, grouped by `index`. `id`, `name` and a piece of the JSON `arguments` may each arrive in a different fragment. A missing id is made up for you. |
| `usage` | Token counts. The last one wins. |
| `finish_reason` | One of `stop`, `tool_calls`, `length`, `content_filter`, `error`, `unknown`. |
| `provider_data` | Data your adapter needs back on a later turn, under your own key. |

**What you receive.** A list of `ChatMessage`s with a `role` (`system`,
`user`, `assistant` or `tool`), `content`, and for assistant turns the
`tool_calls`, `reasoning` and `provider_data` you produced earlier. A tool
result carries the `tool_call_id` it answers. You also get the tools as
`ToolDefinition`s: a `name`, a `description` and a JSON Schema `parameters`.

**The rules.** These are the `Model` contract, checked by `check_model`:

1. Yield no `Delta` until it carries content, reasoning, a tool-call
   fragment, usage or a finish_reason. Swallow metadata-only chunks: the
   loop only retries a turn that failed before its first Delta.
2. Yield `usage` at least once when the backend reports it (the last one
   wins), and none when it doesn't.
3. Yield `finish_reason` exactly once, on or before the last Delta.
4. Raise only `ModelError` subclasses, with `retryable` set correctly. Never
   let a vendor or transport exception escape, and never retry yourself.
5. Never mutate the `messages` or `tools` you were given.
6. Render the system message and tool results however the backend requires
   (a top-level parameter, grouped blocks, and so on).
7. Read back only your own `provider_data` namespace and ignore the others.
   Everything you put there must be JSON-serialisable. It is kept on
   tool-calling turns, which are the ones sent back to you.

**Errors.** Raise the class that says what went wrong. Its `retryable`
default tells the loop whether trying again can help; pass `retry_after`
when the backend says how long to wait.

| Class | Raise it when | Retryable by default |
| --- | --- | --- |
| `ModelConnectionError` | The backend can't be reached: DNS, refused or dropped connection. | yes |
| `ModelRateLimitError` | HTTP 429. Pass the `Retry-After` value as `retry_after`. | yes |
| `ModelServerError` | HTTP 5xx, or the backend reports an overload. | yes |
| `ModelTimeoutError` | The request timed out. The loop also raises it for `model_timeout`. | yes |
| `ModelAuthError` | HTTP 401 or 403, or a backend's own "bad key" error. | no |
| `ModelBadRequestError` | Any other 4xx: an unknown model, a bad field. | no |
| `ModelContextLengthError` | The conversation no longer fits the context window. | no |
| `ModelOutputError` | The output can't be used, such as `provider_data` that isn't JSON-serialisable. For an unparseable tool call the loop asks the model to fix it, then raises `ModelError` if the model keeps failing. | no |
| `ModelOutputTruncatedError` | Raised by the loop when the output limit cuts a tool call short. Adapters don't raise it. | no |

How often and how long the loop retries is set by the agent's `RetryPolicy`;
see [REFERENCE.md](REFERENCE.md#retrypolicy).

**`provider_data`.** Put only what your adapter needs on a later turn, under
one top-level key named after it (`{"anthropic": {...}}`), and read back only
that key. Yield it once, on the last `Delta`. Agentling stores it with the
turn and never reads it.

**Check your adapter.** `check_model` runs a text turn and a tool turn and
reports each rule as passed, failed or skipped. With `live=True` it calls your
backend for real. Here a `ScriptedModel` stands in for one:

```python
import asyncio

from agentling import ChatMessage, ToolCall, Usage
from agentling.testing import ScriptedModel, check_model

turns = [
    ChatMessage("assistant", "pong", usage=Usage(5, 1), finish_reason="stop"),
    ChatMessage(
        "assistant",
        tool_calls=[ToolCall("c1", "echo", {"text": "ping"})],
        usage=Usage(9, 4),
        finish_reason="tool_calls",
    ),
]

report = asyncio.run(check_model(lambda transport: ScriptedModel(turns), live=True))
print(report)
report.raise_for_failures()
```

For tests that run offline, record the backend once and replay it.
`check_model` hands your factory an httpx transport that serves the recording:

```python
import asyncio

from agentling.testing import check_model
from examples.providers.openai_chat import OpenAIChatModel


def build(transport):
    return OpenAIChatModel("gpt-4o-mini", api_key="unused", transport=transport)


report = asyncio.run(
    check_model(build, recorded="examples/providers/fixtures/openai_chat.jsonl")
)
assert report.ok, report
```

`examples/providers/record.py` makes the recording, with API keys redacted:

```bash
uv run python -m examples.providers.record ollama_native \
    --base-url http://localhost:11434 --model qwen3:8b --out my_fixture.jsonl
```

Recorded mode and `RecordingTransport` need `httpx`; `ScriptedModel` and live
mode don't.

## Examples

Run each one from the repository root. These need no API key:

```bash
uv run python -m examples.cli_failure_recovery        # tool errors the model recovers from
uv run python -m examples.cli_memory_chat             # save a session and continue it
uv run python -m examples.cli_advanced_observer       # every event type, and context_manager
uv run python -m examples.cli_dynamic_tools           # add and remove tools on a live session
uv run python -m examples.custom_model_generate_only  # a backend that can't stream
```

These call OpenAI. Set `OPENAI_API_KEY`, and optionally
`AGENTLING_EXAMPLE_MODEL` (default `gpt-4o-mini`):

```bash
uv run python -m examples.cli_math_tutor              # the smallest useful agent
uv run python -m examples.cli_repo_assistant "..."    # streaming, file tools and a skill
uv run python -m examples.cli_notes_agent             # timeouts, redaction and parallel_safe
```

## Project layout

| Path | What's there |
| --- | --- |
| [`src/agentling/`](src/agentling/) | The package. [ARCHITECTURE.md](ARCHITECTURE.md) explains each module. |
| [`examples/`](examples/) | Runnable examples. |
| [`examples/providers/`](examples/providers/) | Reference model adapters, their recorded fixtures and `record.py`. |
| [`tests/`](tests/) | The test suite. It runs offline and also runs this README's code. |
| [REFERENCE.md](REFERENCE.md) | Every setting and the `agentling.testing` API. |
| [MIGRATING.md](MIGRATING.md) | Moving from 0.1 to 0.2. |
| [ROADMAP.md](ROADMAP.md), [CHANGELOG.md](CHANGELOG.md) | Where it's going and what changed. |

## Getting help

Ask a question or share an idea in
[Discussions](https://github.com/lunarstack-co/agentling/discussions). Open an
[issue](https://github.com/lunarstack-co/agentling/issues) for a bug. Report
security problems privately, as described in [SECURITY.md](SECURITY.md).

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) covers setup, the checks CI runs, and how to
add a tool or a model adapter.

## Security

Tools and skill tools run as trusted code in your process, and their output
goes back to the model. [SECURITY.md](SECURITY.md) has the trust model and the
`redact_errors` and `max_tool_output_chars` settings.

## License

[MIT](LICENSE) (c) Folarin Akinloye.

The loop and code-first tools follow smolagents, and the skill format follows
the one Claude uses.
