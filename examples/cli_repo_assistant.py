"""A repo assistant with read-only file tools, streamed with print_events.

Needs OPENAI_API_KEY; AGENTLING_EXAMPLE_MODEL overrides the model name:

    uv run python -m examples.cli_repo_assistant "Summarize the README"

The file tools refuse paths outside the project root. Also shows
max_tool_output_chars and the bundled code-reviewer skill. build_agent() takes
an optional model so tests run it offline.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from agentling import Agent, Model, ToolCallError, print_events, tool
from examples.providers.openai_chat import OpenAIChatModel


def _resolve_within(base: Path, candidate: str) -> Path:
    """Resolve candidate under base, rejecting anything that escapes the root."""

    target = (base / candidate).resolve()
    if target != base and base not in target.parents:
        raise ToolCallError(f"path {candidate!r} escapes the project root")
    return target


def build_agent(model: Model | None = None, root: str = ".") -> Agent:
    """Build a repo assistant whose file tools are sandboxed to `root`."""

    base = Path(root).resolve()

    @tool
    def read_file(path: str) -> str:
        """Read a UTF-8 text file under the project root.

        Args:
            path: Path relative to the project root.
        """
        return _resolve_within(base, path).read_text(encoding="utf-8")

    @tool
    def list_files(subdir: str = ".") -> str:
        """List the entries of a directory under the project root.

        Args:
            subdir: Directory relative to the project root.
        """
        target = _resolve_within(base, subdir)
        return "\n".join(sorted(entry.name for entry in target.iterdir()))

    return Agent(
        model=model
        or OpenAIChatModel(os.environ.get("AGENTLING_EXAMPLE_MODEL", "gpt-4o-mini")),
        tools=[read_file, list_files],
        skills=[Path(__file__).parent / "skills" / "code-reviewer"],
        max_tool_output_chars=2000,
    )


async def main() -> None:
    prompt = " ".join(sys.argv[1:]) or (
        "List the files in the project root, then summarize the README."
    )
    await print_events(build_agent().run(prompt, stream=True))


if __name__ == "__main__":
    asyncio.run(main())
