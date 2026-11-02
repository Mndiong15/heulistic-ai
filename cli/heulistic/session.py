"""Which backend a command talks to: the local stack, or Heulistic Cloud.

Logged out, everything goes to the local stack. `heulistic login` saves an API
key (created at heulistic.com/account), and from then on train, download and
export go to the cloud with it; `--local` sends one command to the local stack
instead, and `heulistic logout` goes back. On a server or in CI, set
HEULISTIC_API_KEY instead of logging in.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import typer

from . import client, stack

CLOUD_API_URL = "https://api.heulistic.com"
CLOUD_WEB_URL = "https://heulistic.com"
KEY_ENV = "HEULISTIC_API_KEY"


@dataclass(frozen=True)
class Session:
    api_url: str | None
    token: str
    cloud: bool


def credentials_path() -> Path:
    return Path.home() / ".heulistic" / "credentials.json"


def _saved() -> dict | None:
    if key := os.environ.get(KEY_ENV, "").strip():
        return {"key": key, "api_url": CLOUD_API_URL}
    try:
        data = json.loads(credentials_path().read_text())
    except (OSError, ValueError):
        return None
    return data if data.get("key") and data.get("api_url") else None


def save(key: str, api_url: str) -> Path:
    """Store the key where only this user can read it."""
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Created 600 from the start, so the key is never briefly world-readable.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"key": key, "api_url": api_url.rstrip("/")}, f)
    os.chmod(path, 0o600)  # in case the file already existed with wider permissions
    return path


def clear() -> bool:
    """Forget the saved key. True if there was one."""
    try:
        credentials_path().unlink()
        return True
    except FileNotFoundError:
        return False


def is_safe_url(url: str) -> bool:
    """Whether a key may be sent here: over HTTPS, or to this machine."""
    parts = urlsplit(url)
    return parts.scheme == "https" or (
        parts.scheme == "http" and parts.hostname in ("localhost", "127.0.0.1", "::1")
    )


def resolve(local: bool = False) -> Session:
    saved = None if local else _saved()
    if saved:
        return Session(saved["api_url"], saved["key"], cloud=True)
    return Session(stack.api_url(), client.DEV_TOKEN, cloud=False)


def connect(local: bool = False) -> str:
    """Pick the backend for this command, say which, and return its URL.

    Exits with a hint when the command needs the local stack and it is down.
    """
    current = resolve(local)
    client.use(current)
    if current.cloud:
        typer.echo(f"Using Heulistic Cloud ({current.api_url}) — add --local for your local stack.")
        return current.api_url
    if not current.api_url:
        typer.secho("Heulistic is not running.", fg=typer.colors.RED)
        typer.echo("Start it with: heulistic start  (or `heulistic login` to use Heulistic Cloud)")
        raise typer.Exit(code=1)
    return current.api_url


def job_hint(current: Session, job_id: str) -> str:
    """Where to keep an eye on a job the command is no longer following."""
    if not current.cloud:
        return "check it with: heulistic status"
    if current.api_url == CLOUD_API_URL:
        return f"watch it at {CLOUD_WEB_URL}/jobs/{job_id}"
    return f"it runs on {current.api_url}"
