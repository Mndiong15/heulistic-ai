"""Talk to the local Heulistic API.

Standard library only, so `pip install heulistic` stays light — which is why
multipart bodies are assembled here rather than handed to a library.

Local mode accepts a fixed bearer token in place of a Cognito JWT. When cloud
support lands this is the seam where a real credential goes.
"""

from __future__ import annotations

import json
import mimetypes
import secrets
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

__all__ = [
    "ApiError",
    "encode_multipart",
    "submit",
    "get_job",
    "download_url",
    "stream_logs",
]

DEV_TOKEN = "dev-localuser"
_TIMEOUT = 30
# Training runs are long; the log stream must not time out mid-job.
_STREAM_TIMEOUT = 60 * 60 * 24


class ApiError(Exception):
    """The API refused the request, carrying its `detail` where there is one."""


def _headers(content_type: str | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {DEV_TOKEN}"}
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def _raise_for_error(exc: urllib.error.HTTPError) -> None:
    try:
        detail = json.loads(exc.read()).get("detail", "")
    except Exception:
        detail = ""
    raise ApiError(detail or f"HTTP {exc.code}") from exc


def _request(
    method: str,
    url: str,
    *,
    body: bytes | None = None,
    content_type: str | None = None,
    stream: bool = False,
):
    """Perform one HTTP request. Tests replace this wholesale."""
    request = urllib.request.Request(
        url, data=body, method=method, headers=_headers(content_type)
    )
    try:
        response = urllib.request.urlopen(
            request, timeout=_STREAM_TIMEOUT if stream else _TIMEOUT
        )
    except urllib.error.HTTPError as exc:
        _raise_for_error(exc)
    except urllib.error.URLError as exc:
        raise ApiError(
            f"Could not reach the Heulistic API at {url} — is it running? "
            "Start it with: heulistic start"
        ) from exc
    return response if stream else response.read()


def encode_multipart(
    fields: dict[str, str | None], files: dict[str, tuple[str, bytes]]
) -> tuple[str, bytes]:
    """Build a multipart/form-data body. Returns (content_type, body)."""
    boundary = secrets.token_hex(16)
    parts: list[bytes] = []

    for name, value in fields.items():
        if value in (None, ""):
            continue  # an empty optional field must not be sent at all
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode()
        )

    for name, (filename, content) in files.items():
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: {mime}\r\n\r\n".encode()
        )
        parts.append(content)
        parts.append(b"\r\n")

    parts.append(f"--{boundary}--\r\n".encode())
    return f"multipart/form-data; boundary={boundary}", b"".join(parts)


def submit(
    api_url: str,
    config: Path,
    dataset: Path | None = None,
    hf_token: str | None = None,
) -> dict:
    """Submit a training job. Returns the API's job record."""
    files = {"config": (config.name, config.read_bytes())}
    if dataset is not None:
        files["dataset"] = (dataset.name, dataset.read_bytes())

    content_type, body = encode_multipart({"hf_token": hf_token}, files)
    raw = _request(
        "POST", f"{api_url}/api/jobs", body=body, content_type=content_type
    )
    return json.loads(raw)


def get_job(api_url: str, job_id: str) -> dict:
    return json.loads(_request("GET", f"{api_url}/api/jobs/{job_id}"))


def start_export(api_url: str, job_id: str, quant: str) -> dict:
    """Queue an export of a finished job's model. Returns the export job."""
    raw = _request(
        "POST", f"{api_url}/api/jobs/{job_id}/export",
        body=json.dumps({"quant": quant}).encode(), content_type="application/json",
    )
    return json.loads(raw)


def download_url(api_url: str, job_id: str) -> str:
    raw = _request("GET", f"{api_url}/api/jobs/{job_id}/download")
    return json.loads(raw)["url"]


def _stream_lines(url: str) -> Iterator[bytes]:
    response = _request("GET", url, stream=True)
    for raw_line in response:
        yield raw_line.rstrip(b"\n")


def stream_logs(api_url: str, job_id: str) -> Iterator[tuple[str, str]]:
    """Yield ("line", text) for each log line, then ("done", status) at the end.

    Server-sent events; anything that is not a well-formed `data:` frame — a
    keepalive, a blank separator — is skipped rather than surfaced.
    """
    url = f"{api_url}/api/jobs/{job_id}/logs"
    for raw_line in _stream_lines(url):
        if not raw_line.startswith(b"data: "):
            continue
        try:
            payload = json.loads(raw_line[len(b"data: "):])
        except json.JSONDecodeError:
            continue
        if payload.get("event") == "done":
            yield "done", payload.get("status", "")
            return
        if "line" in payload:
            yield "line", payload["line"]
