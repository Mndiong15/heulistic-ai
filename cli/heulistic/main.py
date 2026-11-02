from typing import Optional

import typer

from .download_cmd import download
from .export_cmd import export
from .init_cmd import init
from .status_cmd import status
from .train_cmd import train
from .validate_cmd import validate_config

app = typer.Typer(
    name="heulistic",
    help="Heulistic — fine-tune LLMs without the infrastructure.",
    # Spelled out because people reasonably assume the whole tool needs Docker.
    epilog=(
        "init and validate-config need only Python — no Docker, no stack, no "
        "network. start, train, download, export and status run the local stack "
        "and need Docker — or, after `heulistic login`, train, download and "
        "export run on Heulistic Cloud."
    ),
    no_args_is_help=True,
)

app.command("validate-config")(validate_config)
app.command("status")(status)
app.command("train")(train)
app.command("download")(download)
app.command("export")(export)
app.command("init")(init)


def _was_given(ctx: typer.Context, name: str) -> bool:
    """True when the user supplied this option rather than taking the default."""
    source = ctx.get_parameter_source(name)
    return source is not None and source.name != "DEFAULT"


@app.command()
def start(
    ctx: typer.Context,
    port: int = typer.Option(3000, help="Frontend port"),
    api_port: int = typer.Option(8000, help="API port"),
    hf_token: Optional[str] = typer.Option(
        None, "--hf-token", envvar="HF_TOKEN", help="HuggingFace token for gated models"
    ),
    no_gpu: bool = typer.Option(False, "--no-gpu", help="Disable GPU (CPU-only mode)"),
    gpus_per_job: int = typer.Option(
        0,
        "--gpus-per-job",
        min=0,
        help="Give each job this many GPUs and run several at once. "
             "Default 0: each job gets every GPU, one job at a time.",
    ),
    image_tag: Optional[str] = typer.Option(
        None,
        "--image-tag",
        envvar="HEULISTIC_IMAGE_TAG",
        help="Container image tag to run. Defaults to this CLI's version.",
    ),
):
    """Start Heulistic locally and open it in your browser."""
    from . import local_runner, preflight

    # Defaulted ports step aside for whatever already holds them; a port named
    # on the command line is left alone so the clash is reported instead.
    port, api_port, notes = preflight.resolve_ports(
        port, api_port, _was_given(ctx, "port"), _was_given(ctx, "api_port")
    )
    for note in notes:
        typer.secho(f"  {note}", fg=typer.colors.YELLOW)

    typer.echo("Running pre-flight checks…")
    # preflight returns the *effective* GPU setting — a machine with no driver
    # is downgraded to CPU rather than refused.
    use_gpu = preflight.run(frontend_port=port, api_port=api_port, use_gpu=not no_gpu)

    gpu_count = 0
    if use_gpu and gpus_per_job:
        from . import gpu

        gpu_count = len(gpu.detect_gpus())
        if gpus_per_job > gpu_count:
            typer.secho(
                f"--gpus-per-job {gpus_per_job} needs more GPUs than this machine "
                f"has ({gpu_count} GPU{'s' if gpu_count != 1 else ''}).",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        jobs = gpu_count // gpus_per_job
        typer.echo(f"  {gpu_count} GPUs, {gpus_per_job} per job: up to {jobs} jobs at once.")

    typer.echo("Starting Heulistic locally…")
    local_runner.start(
        frontend_port=port,
        api_port=api_port,
        hf_token=hf_token,
        use_gpu=use_gpu,
        image_tag_override=image_tag,
        gpus_per_job=gpus_per_job if use_gpu else 0,
        gpu_count=gpu_count,
    )


@app.command()
def stop():
    """Stop the local Heulistic stack."""
    from . import local_runner

    local_runner.stop()


@app.command()
def logs(
    follow: bool = typer.Option(False, "-f", "--follow", help="Follow log output"),
):
    """Show logs from the local stack."""
    from . import local_runner

    local_runner.logs(follow=follow)


@app.command()
def login(
    api_url: Optional[str] = typer.Option(
        None, "--api-url", help="Log in to a different Heulistic API (testing, self-hosting)."
    ),
):
    """Log in to Heulistic Cloud with an API key, so commands run there."""
    from . import client, session

    url = (api_url or session.CLOUD_API_URL).rstrip("/")
    if not session.is_safe_url(url):
        typer.secho(f"Refusing to send a key to {url}: use https.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.echo("Create a key at https://heulistic.com/account, then paste it here.")
    key = typer.prompt("API key", hide_input=True).strip()
    if not key.startswith("hk_"):
        typer.secho("That is not a Heulistic API key — keys start with hk_.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    client.use(session.Session(url, key, cloud=True))
    try:
        me = client.me(url)
    except client.ApiError as exc:
        typer.secho(f"Login failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    session.save(key, url)
    typer.secho(f"Logged in to Heulistic Cloud as {me['email']}.", fg=typer.colors.GREEN)
    if me.get("billing_enabled"):
        typer.echo(f"  Credit balance: ${me['credit_balance_cents'] / 100:.2f}")
    typer.echo("  train, download and export now run there; add --local to use your local stack.")
    typer.echo("  heulistic logout to switch back.")


@app.command()
def logout():
    """Forget the Heulistic Cloud API key; commands go back to the local stack."""
    from . import session

    if session.clear():
        typer.echo("Logged out of Heulistic Cloud. Commands use your local stack again.")
    else:
        typer.echo("Not logged in.")


def main():
    app()
