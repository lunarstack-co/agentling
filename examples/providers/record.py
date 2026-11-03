"""Record a reference adapter against a live backend and save a fixture.

Runs check_model's two-turn script (a text turn, then a tool turn) through a
RecordingTransport, writes every exchange to --out with keys redacted, and
prints the conformance report. Exits 1 if any rule failed.

    uv run python -m examples.providers.record ollama_native \\
        --base-url http://localhost:11434 --model qwen3:8b \\
        --out examples/providers/fixtures/ollama_native.jsonl

    uv run python -m examples.providers.record anthropic_messages \\
        --model claude-sonnet-5 --api-key-env ANTHROPIC_API_KEY \\
        --out examples/providers/fixtures/anthropic_messages.jsonl

The fixture holds the two successful turns. To make check_model test an
error too, append a third exchange by hand (see fixtures/README.md).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx

from agentling import Model
from agentling.testing import (
    ConformanceReport,
    RecordingTransport,
    check_model,
    write_fixture,
)
from examples.providers.anthropic_messages import AnthropicMessagesModel
from examples.providers.ollama_native import OllamaModel
from examples.providers.openai_chat import OpenAIChatModel

ADAPTERS = ("openai_chat", "anthropic_messages", "ollama_native")


def build(
    adapter: str,
    *,
    model: str,
    base_url: str | None,
    api_key: str | None,
    transport: httpx.AsyncBaseTransport,
) -> Model:
    """Construct one of the reference adapters around `transport`."""

    built: OpenAIChatModel | AnthropicMessagesModel | OllamaModel
    if adapter == "openai_chat":
        built = OpenAIChatModel(model, api_key=api_key, transport=transport)
    elif adapter == "anthropic_messages":
        built = AnthropicMessagesModel(model, api_key=api_key, transport=transport)
    elif adapter == "ollama_native":
        built = OllamaModel(model, api_key=api_key, transport=transport)
    else:
        raise ValueError(f"Unknown adapter {adapter!r}. Choose one of {ADAPTERS}.")
    if base_url:
        built.base_url = base_url.rstrip("/")
    return built


async def record(
    adapter: str,
    *,
    model: str,
    out: str | Path,
    base_url: str | None = None,
    api_key: str | None = None,
    inner: httpx.AsyncBaseTransport | None = None,
) -> ConformanceReport:
    """Run the conformance script live, save the exchanges, return the report.

    `inner` replaces the network transport; tests pass an httpx.MockTransport.
    """

    recorder = RecordingTransport(inner)

    def factory(_: httpx.AsyncBaseTransport | None) -> Model:
        return build(
            adapter, model=model, base_url=base_url, api_key=api_key, transport=recorder
        )

    report = await check_model(factory, live=True)
    write_fixture(out, recorder.exchanges)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the command line, record, print the report and return the exit code."""

    parser = argparse.ArgumentParser(
        prog="python -m examples.providers.record",
        description="Record a reference adapter against a live backend.",
    )
    parser.add_argument("adapter", choices=ADAPTERS)
    parser.add_argument("--model", required=True, help="the model name to request")
    parser.add_argument("--out", required=True, help="where to write the fixture")
    parser.add_argument("--base-url", help="the backend's base URL")
    parser.add_argument(
        "--api-key-env",
        metavar="NAME",
        help="the environment variable holding the API key",
    )
    args = parser.parse_args(argv)

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        parser.error(f"{args.api_key_env} is not set")

    report = asyncio.run(
        record(
            args.adapter,
            model=args.model,
            out=args.out,
            base_url=args.base_url,
            api_key=api_key,
        )
    )
    print(report)
    print(f"Wrote {args.out}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
