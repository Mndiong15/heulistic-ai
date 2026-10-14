"""`heulistic train` — submit a job and follow it, without leaving the terminal.

Validates locally first, so an obvious mistake costs nothing and is caught
before anything is uploaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from . import client, stack
from ._deps import require_axolotl

_TERMINAL_OK = "completed"


def _require_api() -> str:
    api_url = stack.api_url()
    if not api_url:
        typer.secho("Heulistic is not running.", fg=typer.colors.RED)
        typer.echo("Start it with: heulistic start")
        raise typer.Exit(code=1)
    return api_url


def _validate_or_exit(config: Path) -> None:
    diagnostics = require_axolotl().validate(config.read_text())
    errors = [d for d in diagnostics if d.level == "error"]
    warnings = [d for d in diagnostics if d.level == "warning"]

    for diagnostic in errors + warnings:
        location = f"line {diagnostic.line}" if diagnostic.line else ""
        colour = typer.colors.RED if diagnostic.level == "error" else typer.colors.YELLOW
        typer.echo(
            "  "
            + typer.style(f"{diagnostic.level:<8}", fg=colour)
            + f"{location:<9}"
            + f"{diagnostic.field or '':<24}"
            + diagnostic.message
        )

    if errors:
        typer.echo()
        typer.secho(
            f"  {len(errors)} error(s) — nothing submitted. "
            "Use --no-validate to submit anyway.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    if warnings:
        typer.echo()


def train(
    config: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True,
        help="Path to an Axolotl config YAML.",
    ),
    dataset: Optional[Path] = typer.Option(
        None, "--dataset", exists=True, dir_okay=False, readable=True,
        help="JSONL dataset to upload. Omit if the config names a HuggingFace dataset.",
    ),
    hf_token: Optional[str] = typer.Option(
        None, "--hf-token", envvar="HF_TOKEN", help="HuggingFace token for gated models."
    ),
    detach: bool = typer.Option(
        False, "--detach", "-d", help="Submit and return immediately, without following logs."
    ),
    do_validate: bool = typer.Option(
        True, "--validate/--no-validate", help="Check the config before submitting."
    ),
) -> None:
    """Submit a fine-tuning job and stream its logs."""
    api_url = _require_api()

    if do_validate:
        _validate_or_exit(config)

    try:
        job = client.submit(api_url, config, dataset=dataset, hf_token=hf_token)
    except client.ApiError as exc:
        typer.secho(f"Submission failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    job_id = job["job_id"]
    typer.secho(f"Submitted {job_id}", fg=typer.colors.GREEN)

    if detach:
        typer.echo(f"Follow it with: heulistic logs   ·   status: heulistic status")
        raise typer.Exit(code=0)

    typer.echo()
    try:
        for kind, value in client.stream_logs(api_url, job_id):
            if kind == "line":
                typer.echo(f"  {value}")
    except KeyboardInterrupt:
        # Detaching from the logs must not read as a failed run.
        typer.echo()
        typer.secho(
            f"Stopped following. The job is still running — "
            f"check it with: heulistic status",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=0)
    except client.ApiError as exc:
        typer.secho(f"Lost the log stream: {exc}", fg=typer.colors.YELLOW)

    _report_outcome(api_url, job_id)


def _report_outcome(api_url: str, job_id: str) -> None:
    try:
        final = client.get_job(api_url, job_id)
    except client.ApiError as exc:
        typer.secho(f"Could not read the final status: {exc}", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    status = final.get("status", "unknown")
    typer.echo()

    if status == _TERMINAL_OK:
        typer.secho(f"Job {status}.", fg=typer.colors.GREEN)
        cost = (final.get("costs") or {}).get("actual")
        if cost:
            typer.echo(f"  cost: ${float(cost):.2f}")
        typer.echo(f"  get the model: heulistic download {job_id}")
        raise typer.Exit(code=0)

    typer.secho(f"Job {status}.", fg=typer.colors.RED)
    if final.get("error_message"):
        typer.echo(f"  {final['error_message']}")
    raise typer.Exit(code=1)
