# Contributing to Agentling

Bug reports, fixes and new reference adapters are welcome. Small PRs get
reviewed fastest.

## Set up

Agentling uses [uv](https://docs.astral.sh/uv/) and Python 3.11 or newer.

```bash
git clone https://github.com/lunarstack-co/agentling.git
cd agentling
uv sync
```

`uv sync` creates `.venv` with the package and the dev tools: pytest, ruff,
mypy and httpx.

## Run the checks

CI runs these on Python 3.11 and 3.12, and all must pass:

```bash
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv run mypy src tests examples
uv run pytest
uv build
```

`uv run ruff format src tests examples` fixes formatting. The package also
type-checks under `uv run mypy --strict src`; keep it that way. The test suite
runs offline in a few seconds and executes the README's python blocks, so a
README change can fail a test.

## Branches, commits and PRs

Branch from `main`. Commit messages follow
[Conventional Commits](https://www.conventionalcommits.org/):
`feat(tools): ...`, `fix(agent): ...`, `docs: ...`, with `!` after the scope
for a breaking change. A PR describes what changed and why, says how it was
tested, and adds a line to `CHANGELOG.md` under the unreleased version when
users would notice the change.

## Code style

- Keep it small and readable. The loop should stay one file you can read in a
  sitting.
- Type everything.
- Test new behaviour, including how it fails.
- Comments say why, not what. Docstrings on public names say what the reader
  can't get from the signature.
- Avoid new runtime dependencies. The package depends on `pyyaml` only.

## Add a tool or an example

Tools belong in the app that uses them, not in the package. To show a new tool
pattern, add an example under `examples/`: a module docstring that says what it
demonstrates and how to run it, a `build_agent()` the tests can call, and a
test in `tests/test_examples.py`. An example that needs a real model takes an
optional `model` in `build_agent()` so the test can pass a `ScriptedModel`.

## Add a model adapter

Adapters live in `examples/providers/`, not in the package.

1. Copy the closest adapter and follow the contract in the README's
   [Write a model adapter](README.md#write-a-model-adapter).
2. Add it to the `ADAPTERS` tuple and the `build()` function in
   `examples/providers/record.py`, then record a fixture against the real
   backend:
   `uv run python -m examples.providers.record <adapter> --base-url <url> --model <model> --out examples/providers/fixtures/<adapter>.jsonl`.
   Read it before committing: keys in headers are redacted, request bodies are
   not.
3. Add tests like `tests/test_providers_ollama.py`, including a
   `check_model(recorded=...)` run on the fixture.
4. Say where each fixture line came from in
   `examples/providers/fixtures/README.md`.

## Report a bug

Open a [GitHub issue](https://github.com/lunarstack-co/agentling/issues) with
the Agentling and Python versions, the smallest code that shows the problem,
and what you expected. For security problems, follow [SECURITY.md](SECURITY.md)
instead.
