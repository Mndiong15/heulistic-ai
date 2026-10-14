"""Inspect the running local stack.

Reads `docker compose ps` for service state and asks the API for job counts.
Everything degrades to "unknown" rather than raising — `heulistic status` is
the command people run precisely when things are broken.
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Service",
    "services",
    "is_running",
    "published_port",
    "api_url",
    "api_url_is_overridden",
    "job_counts",
]

#: Point the CLI at a stack it did not start — a dev stack from the repo's own
#: docker-compose.yml, a non-default port, or another machine.
API_URL_ENV = "HEULISTIC_API_URL"

_HEULISTIC_DIR = Path.home() / ".heulistic"
_COMPOSE_FILE = _HEULISTIC_DIR / "local_compose.yml"

# Local mode accepts this in place of a Cognito JWT; the API strips "dev-".
_DEV_TOKEN = "dev-localuser"
_TIMEOUT = 5


@dataclass(frozen=True)
class Service:
    name: str
    state: str
    health: str = ""
    port: int | None = None

    @property
    def running(self) -> bool:
        return self.state == "running"

    @property
    def display_state(self) -> str:
        if self.health:
            return f"{self.state} ({self.health})"
        return self.state


def _compose_ps_output() -> str | None:
    """Raw `docker compose ps --format json`, or None if it cannot be read.

    Isolated so tests can supply canned output without mocking subprocess.
    """
    if not _COMPOSE_FILE.exists():
        return None
    try:
        result = subprocess.run(
            ["docker", "compose", "-f", str(_COMPOSE_FILE), "ps", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _parse_rows(output: str) -> list[dict]:
    """Compose has emitted both a JSON array and NDJSON across versions."""
    text = output.strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(parsed, list):
            return [row for row in parsed if isinstance(row, dict)]
        if isinstance(parsed, dict):
            return [parsed]
        return []

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # a warning line mixed into the output
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _first_published_port(row: dict) -> int | None:
    for publisher in row.get("Publishers") or []:
        port = publisher.get("PublishedPort")
        if port:
            return int(port)
    return None


def services() -> list[Service]:
    """Every service in the local stack, in compose order."""
    output = _compose_ps_output()
    if not output:
        return []
    return [
        Service(
            name=row.get("Service", ""),
            state=row.get("State", ""),
            health=row.get("Health", "") or "",
            port=_first_published_port(row),
        )
        for row in _parse_rows(output)
        if row.get("Service")
    ]


def is_running() -> bool:
    return any(service.running for service in services())


def published_port(service_name: str) -> int | None:
    for service in services():
        if service.name == service_name:
            return service.port
    return None


def api_url_is_overridden() -> bool:
    return bool(os.environ.get(API_URL_ENV, "").strip())


def api_url() -> str | None:
    """Where the API is. `HEULISTIC_API_URL` overrides compose discovery."""
    override = os.environ.get(API_URL_ENV, "").strip()
    if override:
        return override.rstrip("/")
    port = published_port("api")
    return f"http://localhost:{port}" if port else None


def _get_json(url: str):
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {_DEV_TOKEN}"})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        return None


def job_counts(base_url: str) -> dict[str, int] | None:
    """Jobs grouped by status, or None if the API cannot be reached."""
    payload = _get_json(f"{base_url}/api/jobs")
    if not isinstance(payload, list):
        return None

    counts: dict[str, int] = {}
    for job in payload:
        if isinstance(job, dict) and job.get("status"):
            counts[job["status"]] = counts.get(job["status"], 0) + 1
    return counts
