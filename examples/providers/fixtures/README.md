# Reference adapter fixtures

Each file is one HTTP exchange per line, in the format `agentling.testing`
reads and writes (`read_fixture`, `write_fixture`). `check_model(...,
recorded=path)` replays them in order: line 1 answers the text turn, line 2
the tool turn, and line 3, when present, must make the adapter raise a
`ModelError`.

Request headers that carry a key (`authorization`, `x-api-key`, `api-key`, `x-goog-api-key`)
are stored as `‹redacted›`.

## Where each line came from

Lines marked "recorded" came from a live backend through
`examples/providers/record.py`. Lines marked "hand-written" were produced by
running the adapter through `RecordingTransport` over a mock server that
answered with the response shapes the provider documents, so the request
side is exactly what the adapter sends. Replace a hand-written file with a
live recording by running `record.py`, then update this table.

| File | Line 1 (text) | Line 2 (tool) | Line 3 (error) | Source of the response shape |
| --- | --- | --- | --- | --- |
| `openai_chat.jsonl` | hand-written | hand-written | hand-written, 429 with `retry-after` | OpenAI chat completions streaming docs |
| `openai_chat_lmstudio.jsonl` | hand-written | hand-written, tool call id `""` | none | LM Studio's OpenAI-compatible server, which sends empty tool call ids |
| `anthropic_messages.jsonl` | hand-written | hand-written, a signed thinking block, a redacted thinking block and a tool call | hand-written, 529 `overloaded_error` | Anthropic Messages streaming and errors docs |
| `ollama_native.jsonl` | recorded 3 Nov 2026, Ollama 0.40.14 with `qwen3:8b` | recorded 3 Nov 2026, same server; the call carries `id` and `function.index` | hand-written, 404 model not found | Ollama `/api/chat` docs and `server/routes.go` for line 3 |
| `gemini_native.jsonl` | hand-written | hand-written, thought and function-call signatures, cached and thinking tokens | hand-written, 429 `RESOURCE_EXHAUSTED` with `retry-after` | [Gemini generateContent REST reference](https://ai.google.dev/api/generate-content), including `Part`, `FunctionCall` and `UsageMetadata` |

The Gemini fixture was captured through `record.py` over `httpx.MockTransport`;
it is not a live Gemini recording. Replace it with a free-tier recording when a
key is available:

```bash
uv run python -m examples.providers.record gemini_native \\
  --model gemini-2.5-flash --api-key-env GEMINI_API_KEY \\
  --out examples/providers/fixtures/gemini_native.jsonl
```

The recorder writes the two successful turns; keep or replace the third error
exchange separately, as described above. Read the request bodies before committing.

Request bodies are recorded as sent. The recorder redacts keys in headers,
not in bodies, so read a new fixture before committing it; the test suite
also scans these files for common key prefixes.
