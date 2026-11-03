# Migrating from 0.1 to 0.2

agentling 0.2 stops shipping a model provider. It defines the contract a model
must follow, and your app brings the model. Most apps change in two places:
where they build the model, and anything that caught OpenAI's exceptions. The
rest of this guide covers every removed or changed API, each with the 0.1 code
and its 0.2 replacement.

Saved sessions keep working: 0.2 loads memory saved by 0.1.

## OpenAIModel is gone

agentling no longer imports or depends on `openai`. Copy the reference adapter
[`examples/providers/openai_chat.py`](examples/providers/openai_chat.py) into
your app, or write an adapter for your backend (see the README's "Write a
model adapter").

Before:

```python
from agentling import Agent, OpenAIModel

agent = Agent(model=OpenAIModel("gpt-4o-mini", base_url="http://localhost:1234/v1"))
```

After:

```python
from agentling import Agent
from myapp.models.openai_chat import OpenAIChatModel  # your copy of the adapter

agent = Agent(model=OpenAIChatModel("gpt-4o-mini", base_url="http://localhost:1234/v1"))
```

`OpenAIModel.context_window` went with it. If your app needs a context window
size, keep it on your adapter or in your own config.

## The openai dependency is gone

`pip install agentling` now installs `pyyaml` only. The reference adapters use
`httpx`; install whatever your adapter needs.

Before:

```python
import openai  # installed by agentling 0.1

client = openai.AsyncOpenAI()
```

After:

```python
import httpx  # install it yourself: pip install httpx

client = httpx.AsyncClient()
```

## ToolSpec and Tool.to_schema() became ToolDefinition

Models now receive a provider-neutral `ToolDefinition`, not OpenAI's
`{"type": "function", ...}` envelope. Each adapter builds its backend's format.

Before:

```python
schema = my_tool.to_schema()  # {"type": "function", "function": {...}}
name = schema["function"]["name"]
```

After:

```python
definition = my_tool.definition  # ToolDefinition(name, description, parameters)
name = definition.name
```

## Model.generate is gone

`Model` has one method, `stream`. A backend that only returns whole replies
wraps them with `deltas_from_message`.

Before:

```python
class MyModel:
    async def generate(self, messages, tools=()):
        return await my_backend(messages)

    async def stream(self, messages, tools=()):
        ...
```

After:

```python
from agentling import deltas_from_message


class MyModel:
    async def stream(self, messages, tools=()):
        reply = await my_backend(messages)  # a ChatMessage
        async for delta in deltas_from_message(reply):
            yield delta
```

## Provider exceptions became ModelError subclasses

Adapters translate their backend's failures into `ModelError` subclasses, so
your app never imports a vendor SDK to handle errors. `ModelOutputError` is now
a `ModelError` too.

Before:

```python
import openai

try:
    answer = await agent.run(task)
except openai.RateLimitError:
    answer = "Busy, try again."
except openai.AuthenticationError:
    answer = "Check your API key."
```

After:

```python
from agentling import ModelAuthError, ModelRateLimitError

try:
    answer = await agent.run(task)
except ModelRateLimitError:
    answer = "Busy, try again."
except ModelAuthError:
    answer = "Check your API key."
```

## Retries moved into the loop

Adapters never retry. The loop retries a turn that failed before its first
chunk with a retryable error, under `RetryPolicy`. Remove any retry wrapper
around your model, and turn off retries in a vendor SDK you still use.

Before:

```python
model = OpenAIModel("gpt-4o-mini", max_retries=3, retry_base_delay=1.0)
agent = Agent(model=model)
```

After:

```python
from agentling import Agent, RetryPolicy

agent = Agent(model=model, retry=RetryPolicy(max_retries=3, base_delay=1.0))
# retry=None turns retries off.
```

## Tool call ids are optional

Some servers (Ollama, LM Studio) send tool calls without ids. agentling 0.1
treated that as unparseable output; 0.2 makes one up. An adapter can leave
`ToolCallDelta.id` as `None`.

Before:

```python
ToolCallDelta(index=0, id=chunk_id or "missing", name=name, arguments=args)
```

After:

```python
ToolCallDelta(index=0, id=chunk_id, name=name, arguments=args)  # None is fine
```

## Memory saves as format 2

`Memory.dump_json()` writes format 2, which adds `reasoning`, `finish_reason`
and `provider_data` to messages. Format 1 saves still load. Version 0.1 can't
read format 2, so don't share saves between apps on different versions.

Before:

```python
saved = session.memory.dump_json()  # format 1, readable by 0.1 and 0.2
```

After:

```python
saved = session.memory.dump_json()  # format 2, readable by 0.2 only
restored = Memory.load_json(old_saved_by_0_1)  # still works
```

## FinalEvent.usage can be None

When the model reports no token usage, `FinalEvent.usage` is `None` instead of
zero counts.

Before:

```python
print(final.usage.total_tokens)  # 0 when nothing was reported
```

After:

```python
if final.usage is not None:
    print(final.usage.total_tokens)
```

## An idle interrupt() does nothing

In 0.1, `interrupt()` while no run was active stopped the next run before its
first step and left its task in memory. In 0.2 it has no effect, and every run
clears the flag when it starts. Remove any code that cleared the private flag.

Before:

```python
session._interrupt.clear()  # so a stale interrupt doesn't kill the next run
answer = await session.run(task, reset=False)
```

After:

```python
answer = await session.run(task, reset=False)
```

## Resume with resume()

`run(task, reset=False)` still adds `task` as a new user turn. To carry on an
interrupted or failed run without repeating the task, call `resume()`. It
raises `ValueError("Nothing to resume")` when memory is empty or already ends
with an answer.

Before:

```python
await session.run(original_task, reset=False)  # added the task a second time
```

After:

```python
await session.resume()
```

## One run per session

A session runs one task at a time. `run()` or `resume()` during a run raises
`RuntimeError("Session is already running...")`. Use `session.is_running` to
check, or a separate session for parallel work.

Before:

```python
await asyncio.gather(session.run("a"), session.run("b"))  # corrupted memory
```

After:

```python
await asyncio.gather(agent.run("a"), agent.run("b"))  # a session each
if not session.is_running:
    await session.run("c", reset=False)
```

## final_answer produces no tool events

A call to `final_answer` is how the model ends a run, so it no longer produces
a `ToolCallEvent` or `ToolResultEvent`. Its step's `StepEvent` is followed by
the `FinalEvent`, and the call is still recorded in memory. Remove any filter
you added for it.

Before:

```python
async for event in agent.run(task, stream=True):
    if isinstance(event, ToolCallEvent) and event.tool_call.name == "final_answer":
        continue
    show(event)
```

After:

```python
async for event in agent.run(task, stream=True):
    show(event)
```

## Step callbacks can be async

`step_callbacks` may return an awaitable, which is awaited before the step's
`StepEvent`. `StepCallback` is the type.

Before:

```python
def save_step(step):
    asyncio.get_event_loop().create_task(db.write(step))  # fire and forget
```

After:

```python
async def save_step(step):
    await db.write(step)


agent = Agent(model=model, step_callbacks=[save_step])
```

## Cut-off tool calls raise ModelOutputTruncatedError

When the model's output limit cuts a tool call short, the loop raises
`ModelOutputTruncatedError` instead of asking the model to fix JSON it can't
finish. Raise the model's output limit or ask for less.

Before:

```python
except ModelError as exc:  # "unparseable tool calls on 3 consecutive turns"
    ...
```

After:

```python
from agentling import ModelOutputTruncatedError

try:
    answer = await agent.run(task)
except ModelOutputTruncatedError:
    answer = "The answer was too long. Try a smaller request."
```

## Malformed turns don't use up max_steps

A turn whose tool calls can't be parsed gets a correction and another try, but
no longer counts towards `max_steps`. A run can therefore make up to
`max_steps * 3 + 1` model calls. Lower `max_steps` if you budgeted calls from it.

Before:

```python
agent = Agent(model=model, max_steps=15)  # at most 16 model calls
```

After:

```python
agent = Agent(model=model, max_steps=15)  # at most 46 model calls, plus retries
```

## Typing: AsyncGenerator and Coroutine

`run(..., stream=True)` is typed as an `AsyncGenerator`, so it works with
`contextlib.aclosing`. `run()` without it is typed as a `Coroutine`, so it
works with `asyncio.create_task`. Code annotated with `AsyncIterator` and
`Awaitable` still type-checks.

Before:

```python
events: AsyncIterator[Event] = session.run(task, stream=True)
answer: Awaitable[str] = session.run(task)
```

After:

```python
async with aclosing(session.run(task, stream=True)) as events:
    ...
job = asyncio.create_task(session.run(task))
```
