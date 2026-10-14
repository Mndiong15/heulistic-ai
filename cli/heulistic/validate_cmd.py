"""`heulistic validate-config` — check an Axolotl config before spending a GPU hour.

Deliberately offline: no Docker, no running stack, no network. The rules come
from `heulistic-axolotl`, which carries a schema generated from Axolotl's own
config reference.
"""

from __future__ import annotations

import json as json_lib
from pathlib import Path
from typing import Optional

import typer

from ._deps import require_axolotl

_LEVEL_COLOUR = {
    "error": typer.colors.RED,
    "warning": typer.colors.YELLOW,
    "info": typer.colors.BLUE,
}

# Shown in the summary line; "info" reads oddly in a count.
_LEVEL_NOUN = {"error": "error", "warning": "warning", "info": "suggestion"}


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


def _summary(counts: dict[str, int]) -> str:
    parts = [
        _plural(counts[level], _LEVEL_NOUN[level])
        for level in ("error", "warning", "info")
        if counts[level]
    ]
    return ", ".join(parts)


def _resolve_target(use_gpu: bool, vram: Optional[float], gpus: int):
    """Return (target, notice). A notice is printed but is not a failure."""
    if vram is not None:
        Hardware = require_axolotl().Hardware
        return Hardware(total_vram_gb=vram, gpu_count=gpus), None
    if not use_gpu:
        return None, None

    from . import gpu as gpu_module

    detected = gpu_module.detect_hardware()
    if detected is None:
        return None, (
            "No NVIDIA GPU detected — skipping the VRAM check. "
            "Use --vram to check against a specific card."
        )
    return detected, None


def _emit_json(config: Path, diagnostics, counts: dict[str, int], ok: bool) -> None:
    typer.echo(
        json_lib.dumps(
            {
                "config": str(config),
                "ok": ok,
                "counts": counts,
                "diagnostics": [
                    {
                        "level": d.level,
                        "code": d.code,
                        "field": d.field,
                        "line": d.line,
                        "message": d.message,
                    }
                    for d in diagnostics
                ],
            },
            indent=2,
        )
    )


def validate_config(
    config: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to an Axolotl config YAML.",
    ),
    use_gpu: bool = typer.Option(
        False, "--gpu", help="Also check the config fits the local GPU."
    ),
    vram: Optional[float] = typer.Option(
        None, "--vram", help="Check against this much VRAM (GB, total across all GPUs) instead of detecting."
    ),
    gpus: int = typer.Option(1, "--gpus", min=1, help="GPU count to pair with --vram."),
    strict: bool = typer.Option(False, "--strict", help="Treat warnings as failures."),
    quiet: bool = typer.Option(False, "--quiet", help="Show errors only."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Check an Axolotl config for problems before submitting it."""
    if use_gpu and vram is not None:
        typer.secho(
            "Use either --gpu or --vram, not both.", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=2)

    target, notice = _resolve_target(use_gpu, vram, gpus)
    diagnostics = require_axolotl().validate(config.read_text(), target=target)

    counts = {
        level: sum(1 for d in diagnostics if d.level == level)
        for level in ("error", "warning", "info")
    }
    failed = bool(counts["error"]) or (strict and bool(counts["warning"]))

    if as_json:
        _emit_json(config, diagnostics, counts, ok=not failed)
        raise typer.Exit(code=1 if failed else 0)

    header = f"Checking {config}"
    if target is not None:
        vram = (f"{target.total_vram_gb:g} GB" if target.gpu_count == 1
                else f"{target.gpu_count} x {target.per_gpu_vram_gb:g} GB")
        header += f" against {target.label} ({vram})" if target.label else f" against {vram}"
    typer.echo(header)
    if notice:
        typer.secho(f"  {notice}", fg=typer.colors.YELLOW)
    typer.echo()

    shown = [d for d in diagnostics if not quiet or d.level == "error"]
    for diagnostic in shown:
        location = f"line {diagnostic.line}" if diagnostic.line else ""
        typer.echo(
            "  "
            + typer.style(f"{diagnostic.level:<8}", fg=_LEVEL_COLOUR[diagnostic.level])
            + f"{location:<9}"
            + f"{diagnostic.field or '':<26}"
            + diagnostic.message
        )

    if not diagnostics:
        typer.secho("  No problems found.", fg=typer.colors.GREEN)
    elif not shown:
        typer.secho("  No errors found.", fg=typer.colors.GREEN)
    else:
        typer.echo()
        typer.echo(f"  {_summary(counts)}")

    raise typer.Exit(code=1 if failed else 0)
