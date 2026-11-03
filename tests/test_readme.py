"""The README's code runs, and the docs say what the code does."""

import ast
import asyncio
import inspect
import re
from pathlib import Path

import pytest

from agentling import Model, ModelError

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
MIGRATING = REPO / "MIGRATING.md"
NOT_EXECUTED = "# not executed:"

_FENCE = re.compile(r"^```python\n(.*?)^```", re.S | re.M)


def _python_blocks(path: Path) -> list[tuple[int, str]]:
    """Every python block in a file, with the line its fence opens on."""

    text = path.read_text(encoding="utf-8")
    return [
        (text.count("\n", 0, match.start()) + 1, match.group(1))
        for match in _FENCE.finditer(text)
    ]


def _run(source: str) -> None:
    code = compile(source, str(README), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    result = eval(code, {"__name__": "__readme__"})
    if inspect.iscoroutine(result):
        asyncio.run(result)


_BLOCKS = _python_blocks(README)


@pytest.mark.parametrize(
    "source", [source for _, source in _BLOCKS], ids=[f"L{line}" for line, _ in _BLOCKS]
)
def test_readme_block(source: str, monkeypatch: pytest.MonkeyPatch) -> None:
    first = source.splitlines()[0]
    if first.startswith(NOT_EXECUTED):
        reason = first.removeprefix(NOT_EXECUTED).strip()
        assert reason, "a block marked not executed must say why"
        pytest.skip(reason)
    monkeypatch.chdir(REPO)

    _run(source)


def _section(text: str, heading: str) -> str:
    start = text.index(heading)
    following = re.search(r"^##? ", text[start + len(heading) :], re.M)
    end = start + len(heading) + following.start() if following else len(text)
    return text[start:end]


def test_readme_quickstart_prints_the_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    quickstart = _section(README.read_text(encoding="utf-8"), "## Quickstart")
    (first, *_) = _FENCE.findall(quickstart)
    monkeypatch.chdir(REPO)

    _run(first)

    assert "ScriptedModel" in first
    assert "OpenAIChatModel(" not in first
    assert capsys.readouterr().out.strip() == "19 + 23 = 42."
    assert "examples/providers/" in quickstart


def _model_errors() -> list[type[ModelError]]:
    found: list[type[ModelError]] = []
    pending = list(ModelError.__subclasses__())
    while pending:
        cls = pending.pop(0)
        found.append(cls)
        pending.extend(cls.__subclasses__())
    return found


def _squash(text: str) -> str:
    return " ".join(text.split())


def _contract_rules() -> list[str]:
    doc = inspect.getdoc(Model) or ""
    rules = re.findall(r"^\d\. (.*?)(?=^\d\. |\Z)", doc, re.S | re.M)
    assert len(rules) == 7
    return [_squash(rule) for rule in rules]


def test_readme_adapter_section_covers_the_contract() -> None:
    section = _section(README.read_text(encoding="utf-8"), "### Write a model adapter")

    for cls in _model_errors():
        assert f"`{cls.__name__}`" in section, cls.__name__
    for name in (
        "provider_data",
        "RetryPolicy",
        "check_model",
        "record.py",
        "anthropic_messages.py",
        "ollama_native.py",
    ):
        assert name in section, name
    for rule in _contract_rules():
        assert rule in _squash(section), rule


def test_readme_error_table_matches_the_classes() -> None:
    section = _section(README.read_text(encoding="utf-8"), "### Write a model adapter")
    rows = dict(re.findall(r"^\| `(Model\w+)` \|.*\| (yes|no) \|$", section, re.M))

    assert set(rows) == {cls.__name__ for cls in _model_errors()}
    for cls in _model_errors():
        expected = "yes" if cls.default_retryable else "no"
        assert rows[cls.__name__] == expected, cls.__name__


_MIGRATING_TERMS = (
    "OpenAIModel",
    "ToolSpec",
    "to_schema",
    "generate",
    "context_window",
    "openai",
    "ToolDefinition",
    "RetryPolicy",
    "ModelError",
    "format 2",
    "FinalEvent.usage",
    "interrupt()",
    "resume()",
    "is_running",
    "RuntimeError",
    "final_answer",
    "step_callbacks",
    "ModelOutputTruncatedError",
    "max_steps",
    "AsyncGenerator",
    "create_task",
)


def test_migrating_covers_every_change() -> None:
    text = MIGRATING.read_text(encoding="utf-8")

    for term in _MIGRATING_TERMS:
        assert term in text, term
    sections = re.split(r"^## ", text, flags=re.M)[1:]
    assert sections
    for section in sections:
        heading = section.splitlines()[0]
        assert section.count("```python") >= 2, heading
    assert "—" not in text and "–" not in text


_LINK = re.compile(r"\]\(([^)\s#]+)(?:#[^)\s]*)?\)")


@pytest.mark.parametrize("name", ["README.md", "MIGRATING.md", "ROADMAP.md"])
def test_doc_links_resolve(name: str) -> None:
    path = REPO / name
    for target in _LINK.findall(path.read_text(encoding="utf-8")):
        if re.match(r"[a-z]+:", target):
            continue
        assert (path.parent / target).exists(), f"{name} links to missing {target}"
