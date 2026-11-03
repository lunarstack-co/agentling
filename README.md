# agentling

[![PyPI](https://img.shields.io/pypi/v/agentling.svg)](https://pypi.org/project/agentling/)
[![CI](https://github.com/lunarstack-co/agentling/actions/workflows/ci.yml/badge.svg)](https://github.com/lunarstack-co/agentling/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A small agent engine for Python: the tool-calling loop, typed memory and
streamed events, with any model you bring.

> **Status: alpha (0.x).** The API may still change before 1.0.

agentling runs the loop at the centre of an agent: ask the model, run the tools
it asks for, record what happened, and repeat until it has an answer. You bring
the model (anything with a `stream()` method), the tools (plain functions) and
whatever sits around the loop, such as a UI, approvals or a server. That outer
layer is often called a harness; agentling is the engine inside it.

What you get over writing the loop yourself: retries that never repeat a tool
call, interrupt and resume, truncated output detected, tool errors fed back to
the model instead of crashing the run, streamed events, typed memory you can
save and reload, and a checker that proves a model adapter follows the
contract.

```python
# not executed: needs an OpenAI API key
import asyncio

from agentling import Agent, tool
from examples.providers.openai_chat import OpenAIChatModel


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city to look up.
    """
    return f"It is 22C and sunny in {city}."


async def main() -> None:
    agent = Agent(model=OpenAIChatModel("gpt-4o-mini"), tools=[get_weather])
    print(await agent.run("What's the weather in Paris?"))


asyncio.run(main())
```

## Contents

- [Why agentling](#why-agentling)
- [Install](#install)
- [Quickstart](#quickstart)
- [Examples](#examples)
- [Usage](#usage)
  - [Tools](#tools)
  - [Running: blocking vs streaming](#running-blocking-vs-streaming)
  - [Sessions and concurrency](#sessions-and-concurrency)
  - [Skills](#skills)
  - [Memory and resuming a run](#memory-and-resuming-a-run)
  - [Interruption](#interruption)
  - [Models and other providers](#models-and-other-providers)
  - [Write a model adapter](#write-a-model-adapter)
- [Architecture](#architecture)
- [Configuration reference](#configuration-reference)
- [Development](#development)
- [License](#license)

## Why agentling

- **Any model.** A model is any object with a `stream(messages, tools)` method.
  agentling ships none; copy a reference adapter from
  [`examples/providers/`](examples/providers/) or write your own.
- **Testable.** `agentling.testing` has a scripted model for running agents
  offline and a checker that tells you which contract rule an adapter broke.
- **Concurrent tools.** Tool calls in a single step run concurrently by
  default.
- **One code path.** Blocking and streaming share the exact same loop. There is
  a single async generator; blocking mode just drains it.
- **Typed memory.** A run is a list of typed steps, not a bag of raw messages.
  Steps know how to render themselves back into model messages and serialize to
  JSON for persistence and replay.
- **Progressive-disclosure skills.** Drop a `SKILL.md` folder in and the model
  sees only its name and description until it decides to load it. Big skill
  libraries stay cheap.
- **Self-healing.** A tool that raises becomes an observation the model can
  recover from, not a crash.
- **Small and readable.** No metaclasses, no plugin registry, no DSL. The loop
  is one file, [`agent.py`](src/agentling/agent.py), that you can read in a
  sitting.

## Install

```bash
pip install agentling
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add agentling
```

Requires Python 3.11 or newer. The only runtime dependency is `pyyaml` (for
skill frontmatter). agentling is async only: it runs inside your event loop, so
call it from async code or with `asyncio.run`.

agentling ships no model adapter. The quickstart below needs no API key. To use
a real model, copy one of the adapters in
[`examples/providers/`](examples/providers/) into your app (they need `httpx`),
or write your own against the `Model` protocol. Snippets in this README import
those adapters as they sit in this repository.

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

Swap `ScriptedModel` for a real adapter and nothing else changes. With the
OpenAI-compatible adapter from [`examples/providers/`](examples/providers/):

```python
# not executed: needs an OpenAI API key
from examples.providers.openai_chat import OpenAIChatModel

agent = Agent(model=OpenAIChatModel("gpt-4o-mini"), tools=[add])  # reads OPENAI_API_KEY
```

## Examples

Runnable examples live in [`examples/`](examples/). The offline ones need no API
key; the rest use an OpenAI-compatible model (set `OPENAI_API_KEY`, and
optionally `AGENTLING_EXAMPLE_MODEL`).

Offline (no API key):

```bash
uv run python examples/cli_failure_recovery.py   # recover from tool errors
uv run python examples/cli_memory_chat.py        # persist and continue a session
uv run python examples/cli_advanced_observer.py  # watch every event type
uv run python examples/cli_dynamic_tools.py      # add and remove tools on a live session
uv run python examples/custom_model_generate_only.py  # a model whose backend can't stream
```

Needs an API key:

```bash
uv run python -m examples.cli_math_tutor               # smallest useful agent
uv run python -m examples.cli_repo_assistant "..."     # streaming + safe file tools
uv run python -m examples.cli_notes_agent              # file-backed notes tools
```

What each one covers:

| Feature | Example |
| --- | --- |
| Basic tool calling | `cli_math_tutor.py` |
| Streaming events (`print_events`) | `cli_repo_assistant.py`, `cli_advanced_observer.py` |
| Every event type + `context_manager` | `cli_advanced_observer.py` |
| Safe, sandboxed file tools | `cli_repo_assistant.py`, `cli_notes_agent.py` |
| Failure recovery (tool errors) | `cli_failure_recovery.py` |
| Memory persistence + `reset=False` | `cli_memory_chat.py` |
| Skills (progressive disclosure) | `cli_repo_assistant.py`, [`skills/code-reviewer`](examples/skills/code-reviewer) |
| Timeouts, redaction, `parallel_safe` | `cli_notes_agent.py` |
| `max_tool_output_chars` | `cli_repo_assistant.py` |
| Tools from a JSON Schema, `add_tools` / `remove_tools` | `cli_dynamic_tools.py` |
| Offline runs with `ScriptedModel` | `cli_failure_recovery.py`, `cli_memory_chat.py` |
| A backend that can't stream (`deltas_from_message`) | `custom_model_generate_only.py` |
| Reference model adapters and fixtures | [`providers/`](examples/providers/) |

## Usage

### Tools

A tool is any Python function wrapped with `@tool`. The function name becomes
the tool name, the docstring summary becomes the description, and the type hints
plus a Google-style `Args:` section become the JSON Schema the model sees. Both
synchronous and asynchronous functions are supported; a synchronous tool runs in
a worker thread so it cannot block the event loop.

> **`tool_timeout` caveat:** a timeout stops the agent from *waiting* on a tool
> and turns it into an observation, but a synchronous tool already running in a
> thread cannot be forcibly cancelled and will finish in the background. Prefer
> async tools, or make blocking tools cooperative, when timeouts matter.

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

Supported parameter types map to JSON Schema: `str`, `int`, `float`, `bool`,
`list[...]`, `dict[...]`, `Optional[...]` / `X | None` (treated as "may be
omitted"), and `Literal[...]` (becomes an enum). Parameters without a default
are marked required. `*args`, `**kwargs`, and positional-only parameters are
rejected at registration time, because a tool is always called with a JSON
object of named arguments.

If the model sends arguments that do not match the schema (missing a required
field, wrong type, unknown key), the tool raises a `ToolCallError` which the
loop feeds back to the model as an error observation. The model gets a chance to
fix its call rather than the run blowing up.

Every agent also has a built-in `final_answer` tool. The model can call it to
end the run explicitly, or it can just reply with plain text (see
[forgiving termination](#forgiving-termination)).

### Running: blocking vs streaming

`run()` has two modes that share one implementation.

Blocking mode returns the final answer:

```python
# not executed: a fragment; `agent` and `session` come from earlier examples
answer: str = await agent.run("Summarize this.")
```

Streaming mode returns an async iterator of typed events:

```python
# not executed: a fragment; `agent` and `session` come from earlier examples
from agentling import FinalEvent, TextDelta, ToolCallEvent

async for event in agent.run("Summarize this.", stream=True):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, ToolCallEvent):
        print(f"\n[calling {event.tool_call.name}]")
    elif isinstance(event, FinalEvent):
        print(f"\nDone: {event.answer}")
```

There is a ready-made renderer, `print_events`, that consumes the stream, prints
text as it arrives along with each tool call and result, and returns the final
answer. It is the "streaming CLI" in about thirty lines:

```python
# not executed: needs an OpenAI API key
import asyncio

from agentling import Agent, print_events, tool
from examples.providers.openai_chat import OpenAIChatModel


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b


async def main() -> None:
    agent = Agent(model=OpenAIChatModel("gpt-4o-mini"), tools=[add])
    answer = await print_events(agent.run("What is 19 + 23?", stream=True))
    print("\nFinal answer:", answer)


asyncio.run(main())
```

### Skills

A skill is a folder containing a `SKILL.md` file: YAML frontmatter followed by a
markdown body of instructions.

```markdown
---
name: code-reviewer
description: Review a code change for bugs, security issues, and style problems.
---

# Code Reviewer

You are reviewing a code change. Work through it methodically and report only
findings you are confident about...
```

Skills use **progressive disclosure**. When you pass skills to an agent, only
their names and descriptions are added to the system prompt as a catalog. The
full instruction body stays out of context until the model calls the built-in
`load_skill(name)` tool, at which point the body is returned as an observation
and any tools the skill declares are registered. This keeps the base prompt
small even with a large library of skills installed.

```python
# not executed: needs an OpenAI API key
import asyncio

from agentling import Agent, tool
from examples.providers.openai_chat import OpenAIChatModel


@tool
def read_file(path: str) -> str:
    """Read a UTF-8 text file and return its contents.

    Args:
        path: Path to the file to read.
    """
    with open(path, encoding="utf-8") as handle:
        return handle.read()


async def main() -> None:
    agent = Agent(
        model=OpenAIChatModel("gpt-4o-mini"),
        tools=[read_file],
        skills=["examples/skills/code-reviewer"],
    )
    print(await agent.run("Review the code in src/agentling/agent.py"))


asyncio.run(main())
```

A skill can also bring its own tools. List Python entry points in the
frontmatter and they are imported and registered when the skill loads:

```markdown
---
name: linting
description: Lint Python files and report issues.
tools:
  - my_package.lint_tools:run_ruff
---
```

Each entry point is a `"module.path:attribute"` string that must resolve to a
`Tool` (a function decorated with `@tool`). You can pass skills as folder paths
(strings or `Path`) or as pre-built `Skill` objects.

> **Security:** a skill's `tools:` entry point is imported, which runs that
> module's code. Load skills only from sources you trust, exactly as you would a
> Python import. See [SECURITY.md](SECURITY.md) for the full trust model.

### Sessions and concurrency

An `Agent` is immutable configuration (model, tools, skills, settings) and is
safe to build once and share. The per-run state (memory, the interrupt token,
and any skill tools loaded during a run) lives on an `AgentSession`.

`agent.run(task)` is a one-shot convenience: it spins up a fresh session, runs
it, and returns the answer. Because each call gets its own session, concurrent
calls on one shared agent never touch each other's memory or tools. When you
need multi-turn, inspection, or interruption, hold a session with
`agent.start()`.

A session runs one task at a time. Calling `run()` or `resume()` while
`session.is_running` is `True` raises `RuntimeError("Session is already
running")`; use a separate session for work that should run in parallel.
`resume()` checks that there is something to resume first, so on a session
whose run has already finished it raises `ValueError("Nothing to resume")`.

```python
# not executed: a fragment; `agent` and `session` come from earlier examples
# one-shot: simplest, and safe under concurrency
answer = await agent.run("A single question")

# hold a session for multi-turn, inspection, or interruption
session = agent.start()
answer = await session.run("First question")
print(session.memory.steps)
```

### Memory and resuming a run

Each session keeps a `Memory` of typed steps. You can serialize it and reload
it later:

```python
# not executed: a fragment; `agent` and `session` come from earlier examples
session = agent.start()
await session.run("First question")

saved = session.memory.dump_json()
# ... later, in another process ...
from agentling import Memory
restored = agent.start()
restored.memory = Memory.load_json(saved)
```

By default each `run()` starts fresh. Pass `reset=False` to keep the session's
memory and add the new task as the next user turn, which is how a multi-turn
conversation works:

```python
# not executed: a fragment; `agent` and `session` come from earlier examples
session = agent.start()
await session.run("First question")
await session.run("A follow-up", reset=False)   # sees the earlier turn
```

To carry on with an interrupted or failed run without adding a new user turn,
call `session.resume()` (see [Interruption](#interruption)).

### Interruption

Call `session.interrupt()` to request a graceful stop. The current run does not
crash: it stops at the next step, streamed chunk or retry wait, ends with a
`FinalEvent` whose status is `"interrupted"`, and keeps its memory. Call
`session.resume()` to carry on from there without adding a new user turn.

```python
# not executed: a fragment; `agent` and `session` come from earlier examples
session = agent.start()
job = asyncio.create_task(session.run("Long job"))
await asyncio.sleep(0)            # let the run start: a real model suspends on its first request
session.interrupt()
await job                         # "Run interrupted."
await session.resume()            # continues from memory, same task
```

An `interrupt()` made while no run is active does nothing, so a stray key press
between runs never stops the next one. `resume()` raises
`ValueError("Nothing to resume")` when memory is empty or already ends with a
final answer. Call `resume()` after the stream has finished, not from inside the
loop that consumes it: the session counts as running until its last event has
been consumed.

`interrupt()` is meant to be called on the run's event loop, for example from
another task. From another thread, schedule it with
`loop.call_soon_threadsafe(session.interrupt)`.

If you stop iterating `run(..., stream=True)` early, close the stream so the
session is free for its next run. `contextlib.aclosing` does that:

```python
# not executed: a fragment; `agent` and `session` come from earlier examples
from contextlib import aclosing

async with aclosing(session.run(task, stream=True)) as events:
    async for event in events:
        if should_stop(event):
            break
```

The same applies when an exception escapes the loop body. Without `aclosing`,
the stream stays open until Python collects it, and until then the session
reports `is_running` and refuses another run.

### Models and other providers

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

agentling ships no adapter of its own.
[`examples/providers/`](examples/providers/) has three to copy into your app:
`openai_chat.py` for OpenAI and OpenAI-compatible servers (LM Studio, vLLM,
Ollama's `/v1`, OpenRouter, Gemini's compatibility endpoint),
`anthropic_messages.py` for Claude, and `ollama_native.py` for Ollama's own API.
Point `base_url` at your endpoint:

```python
from examples.providers.openai_chat import OpenAIChatModel

model = OpenAIChatModel(
    "llama-3.1-70b",
    base_url="http://localhost:8000/v1",
    api_key="not-needed-locally",
)
```

Adapters raise `ModelError` subclasses and never retry; the loop does the
retrying. [Write a model adapter](#write-a-model-adapter) covers the rest.

### Write a model adapter

An adapter is one class with one method. It turns agentling's messages into
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
| `ModelOutputError` | The output can't be used, such as `provider_data` that isn't JSON-serialisable. The loop raises it for unparseable tool calls. | no |
| `ModelOutputTruncatedError` | Raised by the loop when the output limit cuts a tool call short. Adapters don't raise it. | no |

How often and how long the loop retries is set by the agent's `RetryPolicy`;
see the [configuration reference](#configuration-reference).

**`provider_data`.** Put only what your adapter needs on a later turn, under
one top-level key named after it (`{"anthropic": {...}}`), and read back only
that key. Yield it once, on the last `Delta`. agentling stores it with the
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

## Architecture

agentling is a small set of focused modules. Each one owns a single concept, and they depend
on each other in one direction only (agent depends on skills, tools, memory,
events, models; nothing depends on agent).

| Module | Responsibility |
| --- | --- |
| [`models.py`](src/agentling/models.py) | Provider-neutral message types (`ChatMessage`, `ToolCall`, `Usage`), streaming types (`Delta`, `ToolCallDelta`), the `ToolDefinition` handed to models, and the `Model` protocol. |
| [`tools.py`](src/agentling/tools.py) | The `Tool` abstraction, the `@tool` decorator, JSON Schema generation from function signatures, argument validation, and the built-in `final_answer` tool. |
| [`memory.py`](src/agentling/memory.py) | Typed steps (`TaskStep`, `ActionStep`, `FinalStep`), the `Memory` container, rendering to model messages, and JSON serialization. |
| [`events.py`](src/agentling/events.py) | The streaming event types, the `Event` union, and the `print_events` renderer. |
| [`errors.py`](src/agentling/errors.py) | The exception hierarchy: `AgentlingError` and its domain subclasses. |
| [`skills.py`](src/agentling/skills.py) | The `Skill` dataclass, the `SKILL.md` loader (frontmatter plus body), and entry-point tool resolution. |
| [`agent.py`](src/agentling/agent.py) | The `Agent` config/factory, the `AgentSession` that holds one run's state, and the loop that ties everything together. |
| [`retry.py`](src/agentling/retry.py) | `RetryPolicy` and the delay calculation for retried model turns. |
| [`testing.py`](src/agentling/testing.py) | `ScriptedModel`, `check_model` and the fixture recorder. Imported as `agentling.testing`. |

### The agent loop

Everything hangs off a single async generator, `AgentSession._loop`, wrapped
by `_run_stream` so a session runs one task at a time. `run()` is a thin
dispatcher: in streaming mode it hands back that generator; in blocking mode it
drains the generator and returns the final answer. There is no second
implementation to keep in sync.

```
  run(task, stream=True)                run(task)  (stream=False)
          │                                   │
          ▼                                   ▼
   _run_stream(task)  ◀───────────────  _drain(_run_stream(task))
   (async generator)                    (awaits, returns the answer str)
          │
          │  one iteration == one "step"
          ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ 1. interrupt requested?  -> yield FinalEvent, stop (resumable)  │
  │ 2. Memory.to_messages(instructions)  -> the full prompt         │
  │ 3. Model.stream(messages, tools)  -> Delta stream               │
  │       - each text chunk is yielded as a TextDelta               │
  │       - agglomerate_deltas() rebuilds one ChatMessage           │
  │ 4. no tool calls?  -> that text is the answer; finish           │
  │ 5. yield ToolCallEvent per call (none for final_answer)         │
  │ 6. execute tools (concurrently by default) -> ToolResults       │
  │       - a raised exception becomes an error observation         │
  │       - an exact repeat of last step's calls gets a nudge       │
  │ 7. yield ToolResultEvent per result (none for final_answer);    │
  │    record an ActionStep, run step_callbacks, yield StepEvent    │
  │ 8. final_answer called?  -> finish with FinalEvent              │
  └───────────────────────────────────────────────────────────────┘
          │
          ▼
  step limit reached -> ask once for a tool-free answer, then finish
```

Each iteration is a step. A step streams one model turn, runs whatever tools the
model asked for, records the outcome, and checks whether the run is done. The
loop ends in one of three ways: the model calls `final_answer`, the model
replies with plain text and no tool calls, or the step limit is hit and the loop
asks for one last tool-free answer.

### Message and model layer

Everything above the provider speaks in provider-neutral types, not vendor
payloads:

- **`ChatMessage`** is the one message type used internally. It has a role,
  content, optional tool calls, an optional tool-call id (for tool results), and
  optional usage.
- **`ToolCall`** is a provider-neutral tool call: an id, a name, and a parsed
  arguments dict.
- **`Usage`** is input and output token counts (plus cached and reasoning
  breakdowns when the backend reports them), with a `total_tokens` property.

A model adapter is the only place that knows a backend's wire format. It
converts `ChatMessage` lists into the backend's messages on the way out and
converts responses back into `Delta`s on the way in, so the rest of
agentling never sees a provider-specific shape. Swapping providers means
writing one adapter, not touching the loop.

**Streaming and reassembly.** `Model.stream` yields `Delta` objects: small
chunks of content or fragments of a tool call. Tool-call arguments in particular
arrive in pieces across many deltas. The module-level `agglomerate_deltas`
function reassembles a delta stream back into a single `ChatMessage`: it
concatenates content, merges tool-call fragments by their index, and captures
the final usage. This is why the loop can stream text to the user and still work
with a complete `ChatMessage` for tool execution: it streams and reassembles at
the same time.

### Memory and typed steps

A run is recorded as a list of typed steps rather than a flat list of messages:

- **`TaskStep`** is the user's task (or a continued turn).
- **`ActionStep`** is one loop iteration: the assistant message the model
  produced, the tool results from it, and metadata (token usage, wall-clock
  duration).
- **`FinalStep`** is the terminal answer.

Each step knows how to render itself into the `ChatMessage` list the model sees,
via `to_messages()`. `Memory.to_messages(system_prompt)` prepends the system
prompt (which is runtime configuration, not history, so it is never stored in a
step) and then renders every step in order. This separation is what makes the
memory serializable: `dump_json` / `load_json` round-trip the whole run, tagging
each step with its kind so it can be rebuilt, which gives you free persistence,
replay, and multi-turn continuation.

### Events and streaming

The loop communicates progress through a small set of frozen event types:

| Event | Meaning |
| --- | --- |
| `TextDelta` | A chunk of streamed assistant text. |
| `ToolCallEvent` | Emitted just before a tool call runs. Never for `final_answer`. |
| `ToolResultEvent` | Emitted after a tool call completes (success or error). Never for `final_answer`. |
| `StepEvent` | Emitted after a step is recorded to memory; carries the step. |
| `FinalEvent` | Emitted once when the run ends; carries the answer and usage. |

`final_answer` is how the model ends a run, not a tool call to show, so it
produces no tool events: its step's `StepEvent` is followed by the
`FinalEvent`. The call and its result are still in the `ActionStep` the
`StepEvent` carries, so saved memory keeps the whole exchange. A `final_answer`
call that fails validation emits no events either: its error result is on that
`ActionStep`, and the model gets another turn.

`StepEvent` is the bridge between the live event stream and the durable memory:
it carries the exact `ActionStep` that was just written. `print_events` is a
reference consumer of this stream, but you can write your own to drive a UI, log
to a database, or compute metrics.

### Tools and schema generation

`@tool` wraps a function into a `Tool`. At registration time it reads the
function's signature and docstring and builds a JSON Schema `parameters` object:
type hints become JSON types, the `Args:` section becomes per-parameter
descriptions, and defaults decide what is required. Before a tool runs, the loop
validates the model's arguments against that schema and raises `ToolCallError`
on a mismatch. For stateful tools you can subclass `Tool` directly instead of
using the decorator.

### Skills and progressive disclosure

`Skill.from_path` parses a `SKILL.md` into a name, description, instruction body,
and an optional list of tool entry points. When an agent is built with skills,
it registers a single built-in `load_skill` tool and appends a catalog of
`name: description` lines to the system prompt. It does not put any skill bodies
in the prompt.

When the model calls `load_skill(name)`, the agent returns that skill's full
instruction body as the tool observation and registers any tools the skill
declared. Because the loop reads the live tool set fresh on every step, those
newly registered tools are available on the very next turn. The result is that
the cost of a skill (its instructions, its tools) is only paid once the model
actually decides to use it.

### Cross-cutting behaviors

These are small rules layered onto the loop that make agents robust in practice.

#### Forgiving termination

Not every model reliably calls `final_answer`. So if the model replies with
plain text and no tool calls, agentling treats that text as the answer and ends
the run. Explicit `final_answer` and plain-text replies both work.

#### Self-healing tool errors

`_execute_tool` catches any exception a tool raises and turns it into a
`ToolResult` with `is_error=True`, tagged by kind. The observation the model
sees carries a hint that depends on the failure: an invalid-argument error says
"Fix the arguments and try again", while an execution, timeout, or unknown-tool
error says "Consider a different approach or tool". So the model can correct
course, and one bad tool call does not kill the run.

#### Loop detection

If a step's tool calls are an exact repeat of the previous step's (same names,
same arguments), the loop appends a short nudge to the observations telling the
model it already made that call and got the same result. This helps the model
break out of a stuck cycle without a hard failure.

#### Graceful interruption and resume

`interrupt()` sets a flag that the loop checks at the top of each step, after
each streamed chunk, during a wait between retries, and before the step-limit
answer. When set, the loop emits a final "Run interrupted." event and returns
without writing a terminal `FinalStep`. Every run clears the flag when it
starts, so an interrupt made between runs is ignored. Because memory is left
intact, `resume()` picks the run back up exactly where it paused.

#### Unparseable and cut-off tool calls

If a turn's tool calls can't be parsed, the loop tells the model what was wrong
and asks again, up to twice in a row before raising `ModelError`. These
corrections don't count towards `max_steps`, so one run makes at most
`max_steps * 3 + 1` model turns, not counting retries of a turn that failed
before its first chunk: each step can follow two corrections, plus the
step-limit answer. If the turn also stopped because
the model hit its output limit (`finish_reason == "length"`), asking again
can't help, so the loop raises `ModelOutputTruncatedError` straight away; raise
the model's output limit or ask for less. A plain-text answer cut off the same
way still completes, with `FinalEvent.finish_reason == "length"`.

#### Concurrent tool execution

When a single model turn requests several tool calls, they run concurrently with
`asyncio.gather` by default. Set `parallel_tools=False` to run them in order
instead (useful when tools share state or must not interleave).

## Configuration reference

`Agent(...)`:

| Parameter | Default | Description |
| --- | --- | --- |
| `model` | required | Any object implementing the `Model` protocol. |
| `tools` | `()` | Tools to register (a `final_answer` tool is always added). |
| `skills` | `()` | Skills as folder paths or `Skill` objects. |
| `instructions` | built-in default | The system prompt. A skill catalog is appended when skills are present. |
| `max_steps` | `15` | Maximum loop iterations before a forced answer. Must be at least 1. |
| `step_callbacks` | `()` | Called in order with each `ActionStep` once it is recorded, before its `StepEvent`. Async callbacks are awaited; one that raises stops the run. |
| `parallel_tools` | `True` | Run a turn's tool calls concurrently, or in order when `False`. |
| `tool_timeout` | `None` | Per-call time budget (seconds) for tools; a timeout becomes a recoverable observation. |
| `model_timeout` | `None` | Longest wait (seconds) for each streamed chunk; exceeding it raises `ModelTimeoutError`, retried if nothing had arrived yet. |
| `max_tool_output_chars` | `None` | Truncate tool observations head and tail beyond this length. |
| `redact_errors` | `False` | Hide unexpected tool-exception messages from the model and log them instead. |
| `context_manager` | `None` | Callable `messages -> messages` applied before each model call, to trim or summarize. |
| `retry` | `RetryPolicy()` | How to retry a model turn that fails before its first chunk; `None` disables retries. |

`RetryPolicy(...)`:

| Parameter | Default | Description |
| --- | --- | --- |
| `max_retries` | `2` | Retries after the first attempt. |
| `base_delay` | `0.5` | First delay in seconds; doubles each retry unless the backend sent `retry_after`. |
| `max_delay` | `30.0` | Cap on any single delay, including a backend's `retry_after`. |
| `jitter` | `0.2` | Each delay is scaled by a random factor in `[1 - jitter, 1 + jitter]`. |
| `on_retry` | `None` | Called with `(attempt, max_retries, error, delay)` before each wait; awaited if it returns an awaitable. |

`agent.run(task, *, stream=False, reset=True, max_steps=None)`:

- `stream=False` returns an awaitable that resolves to the final answer string.
- `stream=True` returns an async generator of `Event` objects.
- `reset=False` keeps existing memory and adds the task as a new user turn.
- `max_steps` overrides the agent's limit for this run only.

`session.resume(*, stream=False, max_steps=None)` continues an interrupted or
failed run from memory without a new task, with the same return types as
`run()`. `session.is_running` is `True` while a run or resume is in progress.

`agentling.testing` (import it by that name; it isn't re-exported from
`agentling`):

| Name | What it does |
| --- | --- |
| `ScriptedModel(turns)` | Replays one turn per `stream()` call: a `ChatMessage`, a list of `Delta`s, or an exception to raise. Records `calls` and `tools`. Raises `AssertionError("ScriptedModel has no more turns")` when the script runs out. |
| `check_model(factory, *, recorded=None, live=False, expects_tool_calls=True)` | Runs an adapter through a text turn and a tool turn and returns a `ConformanceReport` with `ok`, `rules`, `turns` and `raise_for_failures()`. Needs exactly one of `recorded` (a fixture path) or `live=True`. |
| `RecordingTransport(inner=None)` | An httpx transport that records each exchange, with API keys redacted. Needs `httpx`. |
| `write_fixture(path, exchanges)`, `read_fixture(path)` | Save and load recorded exchanges, one JSON object per line. |

## Development

The project uses [uv](https://docs.astral.sh/uv/) for environment and
dependency management.

```bash
uv sync                                # install everything, including dev deps
uv run pytest                          # run the test suite
uv run ruff check src tests examples            # lint
uv run ruff format --check src tests examples   # formatting (run `ruff format` to fix)
uv run mypy src tests examples                  # type-check
uv build                               # verify the package builds
```

## Limitations

agentling is deliberately small, so some things are out of scope for now:

- **Schema validation is shallow.** Tool arguments are checked against a
  practical subset of JSON Schema (primitives, `list`/`dict`, `Optional`,
  `Literal`), not the full spec.
- **No built-in tracing backend.** `step_callbacks` and the event stream are the
  hooks; wiring them to a tracer or metrics store is up to you.
- **No sandboxing.** Tools and skill-provided tools run as trusted code (see
  [Security](#security)).
- **No automatic context summarization.** Long runs can outgrow the context
  window; supply a `context_manager` to trim or summarize.

## Security

Tools and skill-provided tools run as trusted code in your process, and tool
output is fed back to the model. See [SECURITY.md](SECURITY.md) for the trust
model, the `redact_errors` and `max_tool_output_chars` knobs, and how to report
a vulnerability.

## License

[MIT](LICENSE) (c) Folarin Akinloye.

## Acknowledgements

The design borrows the best ideas from the broader agent ecosystem: the clean
ReAct loop and code-first tools popularized by smolagents, and the
progressive-disclosure skill format used by Claude. agentling's contribution is
packaging those ideas into a small, typed, async codebase you can read end to
end.
