"""What ships: the wheel's contents, and no secrets in the fixtures."""

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "examples" / "providers" / "fixtures"

_MODULES = {
    "__init__",
    "agent",
    "errors",
    "events",
    "memory",
    "models",
    "retry",
    "skills",
    "testing",
    "tools",
}

# Strings that only appear in a real credential.
_SECRET_MARKERS = ("sk-", "sk-ant-", "Bearer ")


def _secret_hits(directory: Path) -> list[tuple[str, str]]:
    """Every (file name, marker) pair where a file holds a secret marker."""

    hits: list[tuple[str, str]] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        hits += [(path.name, marker) for marker in _SECRET_MARKERS if marker in text]
    return hits


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to build the wheel")
def test_wheel_contains_only_the_package(tmp_path: Path) -> None:
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    (wheel,) = tmp_path.glob("agentling-*.whl")

    names = zipfile.ZipFile(wheel).namelist()

    assert not [n for n in names if n.startswith("examples/")]
    assert all(n.startswith(("agentling/", "agentling-")) for n in names), names
    package = {
        n.removeprefix("agentling/") for n in names if n.startswith("agentling/")
    }
    assert package == {f"{m}.py" for m in _MODULES} | {"py.typed"}


def test_fixtures_contain_no_secrets() -> None:
    assert list(FIXTURES.glob("*.jsonl")), "no fixtures found"
    assert _secret_hits(FIXTURES) == []


def test_secret_scan_catches_a_planted_key(tmp_path: Path) -> None:
    (tmp_path / "leak.jsonl").write_text(
        '{"Authorization": "Bearer abc"}', encoding="utf-8"
    )

    assert _secret_hits(tmp_path) == [("leak.jsonl", "Bearer ")]
