"""`heulistic download` — fetch a finished model.

Locally the artifact sits in MinIO inside Docker, so without this the output of
a run is hard to reach. The API hands back a URL; this follows it.
"""

from __future__ import annotations

import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import typer

from . import client, stack

_SUFFIX = "-model.tar.gz"


def _fetch(url: str, dest: Path) -> Path:
    """Stream the artifact to disk. Replaced in tests."""
    with urllib.request.urlopen(url, timeout=300) as response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out)
    return dest


def _resolve_destination(output: Optional[Path], job_id: str) -> Path:
    default_name = f"{job_id}{_SUFFIX}"
    if output is None:
        return Path.cwd() / default_name
    if output.is_dir():
        return output / default_name
    return output


def download(
    job_id: str = typer.Argument(..., help="Job ID to download the model for."),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="File or directory to write to."
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite an existing file."),
    url_only: bool = typer.Option(
        False, "--url-only", help="Print the download URL instead of fetching it."
    ),
) -> None:
    """Download the trained model for a completed job."""
    api_url = stack.api_url()
    if not api_url:
        typer.secho("Heulistic is not running.", fg=typer.colors.RED)
        typer.echo("Start it with: heulistic start")
        raise typer.Exit(code=1)

    try:
        job = client.get_job(api_url, job_id)
    except client.ApiError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1)

    status = job.get("status", "unknown")
    if status != "completed":
        typer.secho(f"Job {job_id} is '{status}' — a model is only available once "
                    "the job has completed.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        url = client.download_url(api_url, job_id)
    except client.ApiError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if url_only:
        typer.echo(url)
        raise typer.Exit(code=0)

    dest = _resolve_destination(output, job_id)
    if dest.exists() and not force:
        typer.secho(f"{dest} already exists — pass --force to overwrite.",
                    fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"Downloading {job_id} → {dest}")
    try:
        _fetch(url, dest)
    except (urllib.error.URLError, OSError) as exc:
        typer.secho(f"Download failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    size_mb = dest.stat().st_size / (1024 * 1024)
    typer.secho(f"Saved {dest} ({size_mb:.1f} MB)", fg=typer.colors.GREEN)
    typer.echo(f"  tar xzf {dest.name}")
