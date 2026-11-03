"""A notes assistant whose tools read and write files in one directory.

Needs OPENAI_API_KEY; AGENTLING_EXAMPLE_MODEL overrides the model name:

    uv run python -m examples.cli_notes_agent

Shows an async tool (search_notes), a writing tool marked
parallel_safe=False (add_note), and the tool_timeout and redact_errors
settings. build_agent() takes an optional model and directory so tests run
it offline.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from agentling import Agent, Model, ToolCallError, tool
from examples.providers.openai_chat import OpenAIChatModel


def _resolve_within(base: Path, candidate: str) -> Path:
    """Resolve candidate under base, rejecting anything that escapes the root."""

    target = (base / candidate).resolve()
    if target != base and base not in target.parents:
        raise ToolCallError(f"path {candidate!r} escapes the notes directory")
    return target


def build_agent(model: Model | None = None, notes_dir: str = "notes") -> Agent:
    """Build a notes assistant whose tools live under `notes_dir`."""

    base = Path(notes_dir).resolve()

    @tool(parallel_safe=False)
    def add_note(name: str, text: str) -> str:
        """Save a note (overwrites one with the same name).

        Args:
            name: The note's name (no path separators).
            text: The note body.
        """
        base.mkdir(parents=True, exist_ok=True)
        _resolve_within(base, f"{name}.txt").write_text(text, encoding="utf-8")
        return f"saved note {name!r}"

    @tool
    async def search_notes(query: str) -> str:
        """Return the names of notes whose text contains the query.

        Args:
            query: Text to search for.
        """
        base.mkdir(parents=True, exist_ok=True)
        hits = [
            note.stem
            for note in sorted(base.glob("*.txt"))
            if query in note.read_text(encoding="utf-8")
        ]
        return "\n".join(hits) or "no matches"

    @tool
    def list_notes() -> str:
        """List the names of all saved notes."""
        base.mkdir(parents=True, exist_ok=True)
        return "\n".join(sorted(note.stem for note in base.glob("*.txt"))) or "no notes"

    return Agent(
        model=model
        or OpenAIChatModel(os.environ.get("AGENTLING_EXAMPLE_MODEL", "gpt-4o-mini")),
        tools=[add_note, search_notes, list_notes],
        tool_timeout=10.0,
        redact_errors=True,
    )


async def main() -> None:
    agent = build_agent()
    print(
        await agent.run("Add a note called groceries: milk, eggs. Then list my notes.")
    )


if __name__ == "__main__":
    asyncio.run(main())
