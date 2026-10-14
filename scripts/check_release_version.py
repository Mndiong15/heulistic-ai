#!/usr/bin/env python3
"""Fail a release whose git tag disagrees with the version being published.

The tag names the container images (`heulistic/api:v0.1.3`); the PyPI version
comes from `cli/pyproject.toml`. Nothing connected the two, so a tag of v0.1.3
could publish PyPI 0.1.2 — which is how `heulistic 0.1.3` came to sit beside
`api:v0.1.2`, each claiming to be the same release.

Usage:
    python scripts/check_release_version.py v0.1.3
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CLI_PYPROJECT = REPO / "cli" / "pyproject.toml"

_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)


def declared_version(pyproject: Path = CLI_PYPROJECT) -> str:
    match = _VERSION_RE.search(pyproject.read_text(encoding="utf-8"))
    if not match:
        raise SystemExit(f"could not find a version in {pyproject}")
    return match.group(1)


def check(tag: str, version: str) -> str | None:
    """Return an error message, or None when the tag and version agree."""
    if not tag.startswith("v"):
        return (
            f"tag {tag!r} does not start with 'v'. Release tags are the "
            f"published version prefixed with 'v' — expected 'v{version}'."
        )
    tagged = tag[1:]
    if tagged != version:
        return (
            f"tag {tag!r} does not match the version being published.\n"
            f"  cli/pyproject.toml declares {version!r}\n"
            f"  the tag implies           {tagged!r}\n"
            f"Container images would be tagged {tag} while PyPI received "
            f"{version}. Set them to the same value and re-tag."
        )
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    version = declared_version()
    problem = check(argv[1], version)
    if problem:
        print(f"✗ {problem}", file=sys.stderr)
        return 1
    print(f"✓ tag {argv[1]} matches cli/pyproject.toml version {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
