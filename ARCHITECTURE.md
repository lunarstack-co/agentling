# How Agentling works

Agentling is one loop and the types around it. The loop sends the conversation
to a model, runs the tools the model asks for, records the outcome as a step,
and repeats until the model answers. Everything else in the package is a type
the loop reads or writes.

## Modules

Imports run one way: `agent.py` uses the other modules, and nothing in the
package imports `agent.py` except `__init__.py`.

| Module | What it owns |
| --- | --- |
| [`models.py`](src/agentling/models.py) | The provider-neutral types (`ChatMessage`, `ToolCall`, `Usage`, `ToolDefinition`), the streaming types (`Delta`, `ToolCallDelta`), `agglomerate_deltas`, and the `Model` protocol. |
| [`tools.py`](src/agentling/tools.py) | `Tool`, the `@tool` decorator, schema generation from signatures, argument validation and the built-in `final_answer` tool. |
| [`memory.py`](src/agentling/memory.py) | The step types (`TaskStep`, `ActionStep`, `FinalStep`), `Memory`, rendering steps to messages, and JSON save and load. |
| [`events.py`](src/agentling/events.py) | The event types and `print_events`. |
| [`errors.py`](src/agentling/errors.py) | `AgentlingError` and its subclasses, including the `ModelError` family adapters raise. |
| [`retry.py`](src/agentling/retry.py) | `RetryPolicy` and the delay calculation. |
| [`skills.py`](src/agentling/skills.py) | `Skill`, the `SKILL.md` loader and tool entry-point resolution. |
| [`agent.py`](src/agentling/agent.py) | `Agent`, `AgentSession` and the loop. |
| [`testing.py`](src/agentling/testing.py) | `ScriptedModel`, `check_model` and the fixture recorder, imported as `agentling.testing`. |

## The loop

`AgentSession._loop` is an async generator that yields events. `_run_stream`
wraps it so a session runs one task at a time. `run(stream=True)` returns that
generator; `run()` drains it and returns the answer, so both modes run the same
code.

```
  run(task, stream=True)                run(task)
          │                                 │
          ▼                                 ▼
     _run_stream(task)  ◀──────────  _drain(_run_stream(task))
          │
          │  one iteration is one step
          ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ 1. interrupt requested?  -> yield FinalEvent, stop (resumable)   │
  │ 2. Memory.to_messages(instructions), then context_manager        │
  │ 3. Model.stream(messages, tools)  -> Deltas                      │
  │       - each text chunk is yielded as a TextDelta                │
  │       - agglomerate_deltas() rebuilds one ChatMessage            │
  │ 4. no tool calls?  -> that text is the answer; finish            │
  │ 5. yield a ToolCallEvent per call (none for final_answer)        │
  │ 6. run the tools, concurrently by default  -> ToolResults        │
  │       - an exception becomes an error result                     │
  │       - an exact repeat of the last step's calls gets a nudge    │
  │ 7. yield a ToolResultEvent per result (none for final_answer);   │
  │    record an ActionStep, run step_callbacks, yield StepEvent     │
  │ 8. final_answer called?  -> finish with a FinalEvent             │
  └──────────────────────────────────────────────────────────────────┘
          │
          ▼
  step limit reached  -> ask once for a tool-free answer, then finish
```

A run ends in one of three ways: the model calls `final_answer`, the model
replies with text and no tool calls, or the step limit is reached and the loop
asks for one last answer with no tools offered.

## Messages and models

Above the adapter, everything uses Agentling's own types. A `ChatMessage` has
a role, content, and optionally tool calls, the id of the call it answers,
usage, reasoning, a finish reason and `provider_data`. A `ToolCall` is an id, a
name and a parsed arguments dict.

An adapter is the only code that knows a backend's wire format. It converts
messages on the way out and yields `Delta`s on the way in. A tool call's
arguments usually arrive in pieces, so `agglomerate_deltas` joins the deltas
back into one `ChatMessage`: it concatenates text, merges tool-call fragments
by index, and keeps the last usage. The loop streams text to the caller and
reassembles the message from the same deltas.

## Memory

A run is a list of typed steps, not raw messages:

- `TaskStep`: the user's task, or a follow-up turn.
- `ActionStep`: one model turn with tool calls, its tool results, usage and
  duration.
- `FinalStep`: the answer.

Each step renders itself with `to_messages()`. `Memory.to_messages()` puts the
system prompt first; it is configuration, so it's never stored as a step.
`dump_json()` and `load_json()` round-trip the steps, tagged by kind, which is
what makes saving, reloading and multi-turn work.

## Events

| Event | When |
| --- | --- |
| `TextDelta` | A chunk of streamed answer text. |
| `ToolCallEvent` | Before a tool call runs. |
| `ToolResultEvent` | After a tool call finishes, successfully or not. |
| `StepEvent` | After a step is recorded; carries the `ActionStep`. |
| `FinalEvent` | Once, when the run ends: the answer, usage, status and finish reason. |

Within a step the order is fixed: the text chunks, every `ToolCallEvent`, every
`ToolResultEvent` in the same order, then the `StepEvent`. A `final_answer`
call gets no tool events, because it is how the run ends; the call and its
result are still in the `ActionStep`. A run that ends normally yields exactly
one `FinalEvent`, last. A run that fails raises instead.

## Tools

`@tool` reads the function's signature and docstring once, when the tool is
built: type hints become JSON types, the `Args:` section becomes parameter
descriptions, and parameters without defaults become required. Before each
call, the loop checks the model's arguments against that schema and turns a
mismatch into a `ToolCallError`. The check covers a subset of JSON Schema
(types, required, enums, `additionalProperties: false`) and ignores keywords
outside it, so it never refuses a call the full schema allows.

## Skills

`Skill.from_path` reads a `SKILL.md` into a name, description, instructions and
optional tool entry points. An agent with skills gets a built-in `load_skill`
tool, and a catalog of `name: description` lines is appended to its system
prompt. When the model calls `load_skill(name)`, the loop returns the
instructions as the tool result and registers the skill's tools on the
session, so they're offered from the next turn.

## Behaviour worth knowing

**Plain text ends the run.** Not every model calls `final_answer` reliably, so
a reply with no tool calls is taken as the answer.

**Tool errors go to the model.** `_execute_tool` turns any exception into a
`ToolResult` with `is_error=True` and a kind. The model sees a hint that
depends on the kind: "Fix the arguments and try again" for bad arguments, and
"Consider a different approach or tool" for a failure, a timeout or an
unknown tool.

**Repeats get a nudge.** If a step's tool calls exactly repeat the previous
step's, same names and arguments, each result gets a note saying so.

**Interrupts are flags.** `interrupt()` sets a flag the loop checks at the top
of each step, after each streamed chunk, during a retry wait and before the
step-limit answer. The run then yields a "Run interrupted." `FinalEvent` and
writes no `FinalStep`, so `resume()` picks up from memory. Every run clears the
flag when it starts, so an interrupt between runs is ignored.

**Unparseable tool calls get a correction.** When a turn's tool calls can't be
parsed, the loop tells the model what was wrong and asks again, up to twice in
a row, then raises `ModelError`. Corrections don't count towards `max_steps`,
so a run makes at most `max_steps * 3 + 1` model turns, plus retries of turns
that failed before their first chunk. If the turn stopped at the model's output
limit (`finish_reason == "length"`), asking again can't help, so the loop raises
`ModelOutputTruncatedError` at once. A text answer cut off the same way still
completes, with `FinalEvent.finish_reason == "length"`.

**Retries never repeat output.** A turn is retried only when it raised a
retryable `ModelError` before its first chunk, so the caller never sees the
same text twice and no tool runs twice.

**Tools run concurrently.** A step's tool calls run with `asyncio.gather`,
unless `parallel_tools=False` or any of the tools has `parallel_safe=False`.

## What it doesn't do

- Validate the full JSON Schema spec. Tool arguments get the subset above.
- Trace. `step_callbacks` and the event stream are the hooks; connecting them
  to a tracer is up to you.
- Sandbox tools. Tools and skill tools run as trusted code; see
  [SECURITY.md](SECURITY.md).
- Summarise context. A long run can outgrow the context window; pass a
  `context_manager` to trim it.
