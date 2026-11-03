# agentling roadmap

agentling is a tiny async framework for reliable, observable, tool-using agents:
a clean ReAct loop, typed memory, streaming events, recoverable failures, and
progressive-disclosure skills, in a codebase small enough to read in one
sitting.

This roadmap is **directional, not a schedule**. agentling is alpha (0.x) and
the API may change before 1.0. Priorities can shift with community feedback, and
versions below are indicative — pre-1.0 milestones may split or merge.

## Shipped — v0.1.0

The first release: the core framework plus a production-hardening pass —
`Agent`/`AgentSession`, an OpenAI-compatible model adapter, `@tool`, typed
memory with JSON persistence, streaming events, progressive-disclosure skills,
timeouts/cancellation, malformed-output recovery, and a runnable examples suite.

## Shipped — v0.1.1

Tools defined at runtime: `add_tools` / `remove_tools` on a live session,
validation that tolerates real-world JSON Schemas, `validate_arguments`, and an
MCP bridge recipe in the docs (no `mcp` dependency).

## Next — Reliability (v0.2)

**Top priority.** Harden and clean up the v0.1 surface before adding features.
No point building on foundations with sharp edges.

- **Session lifecycle.** An idle `interrupt()` no longer silently kills the next
  run, and using one session concurrently raises a clear error instead of
  quietly corrupting memory.
- **Honest errors.** Make the exception hierarchy real — the documented error
  types are actually raised, so `except AgentlingError` behaves as promised.
- **Broader compatibility.** Stop rejecting valid tool calls from
  OpenAI-compatible backends (some omit ids mid-stream); synthesize them
  instead.
- **Sampling controls.** Expose `temperature`, `max_tokens`, `seed`, and friends
  on the model, so deterministic evals and cost caps are possible.
- **Cleaner semantics.** `final_answer` no longer leaks into the event stream,
  context-window trimming applies on every path, and resuming a session no
  longer duplicates the task.
- **Robustness.** Safe, async-capable step callbacks; Python 3.13 and 3.14 in
  CI; stricter typing; and a batch of small correctness fixes.

## Then — Observability and evaluation (v0.3)

Make every run inspectable and testable.

- **Lifecycle tracing.** A dependency-free tracing layer over the whole
  lifecycle — run, step, model call, tool call — capturing inputs, outputs,
  token usage, timing, and errors.
- **OpenTelemetry adapter.** Emit standard GenAI spans so traces flow to
  Langfuse, LangSmith, Arize Phoenix, and any OTLP backend through one
  integration, not a bespoke plugin per vendor.
- **Evals.** An evaluation harness, with datasets and evaluators, that works
  locally or against a hosted experiment backend. The deterministic testing
  model it builds on shipped early, in 0.2.0: `agentling.testing` has
  `ScriptedModel` and the `check_model` conformance checker.

## Model adapters: none in the package

agentling ships no model providers, and that is a decision, not a gap. Any
object with a `stream(messages, tools)` method is a model, and
`agentling.testing.check_model` proves an adapter follows the contract.
[`examples/providers/`](examples/providers/) has reference adapters to copy:
OpenAI-compatible (which also covers OpenRouter, Groq, Together, DeepSeek,
Mistral, xAI, vLLM, LM Studio and Gemini's compatibility endpoint), Anthropic
Messages and native Ollama. New reference adapters are welcome when they come
with a recorded fixture that passes `check_model`.

## Exploring — beyond

Ideas we like but have not committed to a milestone:

- Structured output (JSON mode / response schemas).
- A human-in-the-loop hook to approve, deny, or modify a tool call before it
  runs (guardrails).
- An exhaustive failure-mode test suite and a trust policy for skill-provided
  tools.

## Toward 1.0

Once reliability, observability, and the model surface settle, 1.0 is about
committing to a stable public API and semantic-versioning guarantees.

## Influence the roadmap

This is an open, early project and the priorities above are open to input. Open
an issue to propose something, describe a use case we are missing, or tell us
which item matters most to you. Bug reports and small PRs are especially
welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
