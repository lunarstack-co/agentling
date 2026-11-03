# Reference adapter fixtures

Each file is one HTTP exchange per line, in the format `agentling.testing`
reads and writes (`read_fixture`, `write_fixture`). `check_model(...,
recorded=path)` replays them in order: line 1 answers the text turn, line 2
the tool turn, and line 3, when present, must make the adapter raise a
`ModelError`.

Request headers that carry a key (`authorization`, `x-api-key`, `api-key`)
are stored as `‹redacted›`.

## Where each line came from

None of these lines is a live recording yet. Each one was produced by
running the adapter through `RecordingTransport` over a mock server that
answered with the response shapes the provider documents, so the request
side is exactly what the adapter sends. Replace a file with a live
recording by running `examples/providers/record.py`, then update this table.

| File | Line 1 (text) | Line 2 (tool) | Line 3 (error) | Source of the response shape |
| --- | --- | --- | --- | --- |
| `openai_chat.jsonl` | hand-written | hand-written | hand-written, 429 with `retry-after` | OpenAI chat completions streaming docs |
| `openai_chat_lmstudio.jsonl` | hand-written | hand-written, tool call id `""` | none | LM Studio's OpenAI-compatible server, which sends empty tool call ids |
| `anthropic_messages.jsonl` | hand-written | hand-written, a signed thinking block, a redacted thinking block and a tool call | hand-written, 529 `overloaded_error` | Anthropic Messages streaming and errors docs |
| `ollama_native.jsonl` | hand-written | hand-written, the tool call in one chunk | hand-written, 404 model not found | Ollama `/api/chat` docs and `server/routes.go` |

Request bodies are recorded as sent. The recorder redacts keys in headers,
not in bodies, so read a new fixture before committing it; the test suite
also scans these files for common key prefixes.
