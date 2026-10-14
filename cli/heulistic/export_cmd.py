"""`heulistic export` — turn a finished job into a model Ollama can run.

Merges the trained adapter into the base model, quantizes it to GGUF and writes
an Ollama Modelfile whose prompt template matches training — all inside the
local stack, so it is free. The one real limit is disk: for a moment the
export holds two full-precision copies of the model, which is where a large
model is better sent to the cloud.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import typer

from . import client, gpu, stack
from ._deps import require_axolotl

CLOUD_URL = "https://heulistic.com"


# ── Seams replaced in tests ───────────────────────────────────────────────────


def _fetch(url: str, dest: Path) -> Path:
    with urllib.request.urlopen(url, timeout=600) as response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out)
    return dest


def _free_disk_gb() -> Optional[float]:
    """Free space where Docker keeps its data, or None if it cannot be seen —
    Docker Desktop keeps it inside a VM."""
    try:
        root = subprocess.run(
            ["docker", "info", "--format", "{{.DockerRootDir}}"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if root and Path(root).exists():
            return shutil.disk_usage(root).free / 1e9
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _system_ram_gb() -> Optional[float]:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (AttributeError, ValueError, OSError):
        return None


def _ollama() -> Optional[str]:
    return shutil.which("ollama")


def _run(cmd: list[str], cwd: Path) -> int:
    return subprocess.run(cmd, cwd=cwd).returncode


# ── Command ───────────────────────────────────────────────────────────────────


def _fail(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED)
    raise typer.Exit(code=1)


def _choose_quant(params_b: float) -> str:
    ax = require_axolotl()
    hardware = gpu.detect_hardware()
    if hardware is not None:
        where, available = f"{hardware.display_name} ({hardware.total_vram_gb:g} GB VRAM)", hardware.total_vram_gb
    else:
        available = _system_ram_gb()
        where = f"{available:.0f} GB of RAM" if available else None

    quant = ax.recommend_quant(params_b, available)
    if quant is None:
        need = ax.inference_memory_gb(params_b, "Q4_K_M")
        typer.secho(
            f"  Even Q4_K_M needs about {need:.1f} GB to run — it will not fit {where}. "
            "Exporting Q4_K_M anyway; run it on a bigger machine.",
            fg=typer.colors.YELLOW,
        )
        return "Q4_K_M"
    if where:
        typer.echo(f"  {quant} — the best quality that fits {where}.")
    return quant


def export(
    job_id: str = typer.Argument(..., help="A completed training job."),
    quant: Optional[str] = typer.Option(
        None, "--quant", "-q",
        help="Q4_K_M, Q5_K_M or Q8_0. Default: the best that fits this machine.",
    ),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Ollama model name."),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Directory for the GGUF and Modelfile (default ./<name>)."
    ),
    register: bool = typer.Option(
        True, "--ollama/--no-ollama", help="Run `ollama create` if Ollama is installed."
    ),
) -> None:
    """Export a trained model to GGUF and set it up for Ollama."""
    api_url = stack.api_url()
    if not api_url:
        typer.secho("Heulistic is not running.", fg=typer.colors.RED)
        typer.echo("Start it with: heulistic start")
        raise typer.Exit(code=1)

    try:
        source = client.get_job(api_url, job_id)
    except client.ApiError as exc:
        _fail(str(exc))
    if source.get("kind", "train") != "train" or source.get("status") != "completed":
        _fail(f"Job {job_id} is not a completed training job — only those can be exported.")

    ax = require_axolotl()
    import yaml  # comes with heulistic-axolotl; lazy for the reason in _deps.py

    cfg = yaml.safe_load(source.get("config_yaml") or "") or {}
    params_b = ax.parse_param_billions(str(cfg.get("base_model") or ""))

    if quant is None:
        quant = _choose_quant(params_b)
    quant = quant.upper()
    if quant not in ax.GGUF_QUANTS:
        _fail(f"Unknown format {quant} — choose one of {', '.join(ax.GGUF_QUANTS)}.")

    name = name or f"heulistic-{job_id[:8]}"
    out_dir = output or Path.cwd() / name

    existing = next(
        (e for e in source.get("exports") or []
         if e.get("quant") == quant and e.get("status") not in ("failed", "cancelled")),
        None,
    )
    if existing:
        export_id = existing["job_id"]
        typer.echo(f"Using the existing {quant} export ({export_id}).")
    else:
        need = ax.export_scratch_gb(params_b)
        free = _free_disk_gb()
        if free is not None and free < need:
            _fail(
                f"Exporting this {params_b:g}B model needs about {need:.0f} GB of free disk "
                f"for Docker, and there is {free:.0f} GB. Free some space, or export it on "
                f"Heulistic Cloud ({CLOUD_URL}), which sizes the disk to the model."
            )
        try:
            export_id = client.start_export(api_url, job_id, quant)["job_id"]
        except client.ApiError as exc:
            _fail(str(exc))
        typer.echo(f"Exporting to {quant} (job {export_id}) — merging, converting, quantizing…")
        for kind, value in client.stream_logs(api_url, export_id):
            if kind == "line":
                typer.echo(f"  {value}")

    result = client.get_job(api_url, export_id)
    if result.get("status") != "completed":
        _fail(f"Export failed: {result.get('error_message') or result.get('status')}")

    url = client.download_url(api_url, export_id)
    gguf_name = url.rsplit("/", 1)[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Downloading {gguf_name} → {out_dir}")
    try:
        _fetch(url, out_dir / gguf_name)
    except (urllib.error.URLError, OSError) as exc:
        _fail(f"Download failed: {exc}")
    (out_dir / "Modelfile").write_text(result.get("modelfile") or f"FROM ./{gguf_name}\n")
    typer.secho(f"Saved {out_dir / gguf_name} and its Modelfile", fg=typer.colors.GREEN)

    ollama = _ollama() if register else None
    if ollama and _run([ollama, "create", name, "-f", "Modelfile"], cwd=out_dir) == 0:
        typer.secho(f"Added to Ollama as {name}.", fg=typer.colors.GREEN)
        typer.echo(f"  ollama run {name}")
        return
    typer.echo()
    typer.echo("To run it with Ollama:")
    typer.echo(f"  cd {out_dir}")
    typer.echo(f"  ollama create {name} -f Modelfile")
    typer.echo(f"  ollama run {name}")
