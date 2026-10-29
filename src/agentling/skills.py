"""Markdown skills, loaded on demand.

A skill is a folder with a SKILL.md: YAML frontmatter (name, description and
optional tool entry points) and a markdown body of instructions. The model sees
only the name and description until it calls load_skill, which returns the body
and registers the skill's tools. This keeps the prompt small.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .tools import Tool

SKILL_FILE = "SKILL.md"


@dataclass
class Skill:
    """Instructions plus optional tools that the model loads when it needs them.

    `tools` holds entry points ("package.module:attribute") that resolve to Tool
    objects when the skill is loaded. `path` is the skill's folder, so the
    instructions can refer to files next to SKILL.md.
    """

    name: str
    description: str
    instructions: str
    path: Path
    tools: list[str] = field(default_factory=list)

    @classmethod
    def from_path(cls, path: str | Path) -> Skill:
        """Load a Skill from a folder containing a SKILL.md file.

        Raises ValueError if the frontmatter is malformed, lacks name or
        description, or has a `tools` value that isn't a list of strings.
        """

        folder = Path(path)
        source = (folder / SKILL_FILE).read_text(encoding="utf-8")
        frontmatter, body = _split_frontmatter(source)

        try:
            name = frontmatter["name"]
            description = frontmatter["description"]
        except KeyError as exc:
            raise ValueError(
                f"{folder / SKILL_FILE} is missing required frontmatter key {exc}."
            ) from exc

        raw_tools = frontmatter.get("tools", [])
        if raw_tools is None:
            raw_tools = []
        if not isinstance(raw_tools, list) or not all(
            isinstance(item, str) for item in raw_tools
        ):
            raise ValueError(
                f"{folder / SKILL_FILE}: 'tools' must be a list of strings."
            )

        return cls(
            name=name,
            description=description,
            instructions=body.strip(),
            path=folder,
            tools=raw_tools,
        )

    def load_tools(self) -> list[Tool]:
        """Import and return the Tool objects named in this skill's frontmatter.

        Raises ValueError for an entry point without a colon, TypeError if it
        doesn't resolve to a Tool, and the usual import errors if it can't load.
        """

        return [_resolve_tool(spec) for spec in self.tools]


def _split_frontmatter(source: str) -> tuple[dict[str, Any], str]:
    """Split a SKILL.md into its YAML frontmatter dict and markdown body.

    Frontmatter sits between two lines that are exactly '---'. A file that
    doesn't start with one is all body, and only the first closing fence counts,
    so a '---' rule in the body is left alone.
    """

    lines = source.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, source

    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            data = yaml.safe_load("".join(lines[1:index])) or {}
            if not isinstance(data, dict):
                raise ValueError("SKILL.md frontmatter must be a YAML mapping.")
            return data, "".join(lines[index + 1 :])

    raise ValueError("Unterminated SKILL.md frontmatter (missing closing '---').")


def _resolve_tool(spec: str) -> Tool:
    """Resolve a 'package.module:attribute' entry point to a Tool instance."""

    module_path, sep, attr = spec.partition(":")
    if not sep:
        raise ValueError(f"Tool entry point {spec!r} must be 'module.path:attribute'.")

    obj = getattr(importlib.import_module(module_path), attr)
    if not isinstance(obj, Tool):
        raise TypeError(
            f"Tool entry point {spec!r} resolved to {type(obj).__name__}, "
            "expected a Tool (did you forget the @tool decorator?)."
        )
    return obj
