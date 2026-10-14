"""`heulistic status` — is the local stack up, and what is it doing?

Exits 0 only when every service is running, so it can gate a shell script:

    heulistic status && heulistic validate-config config.yaml
"""

from __future__ import annotations

import typer

from . import stack

_STATE_COLOUR = {"running": typer.colors.GREEN, "exited": typer.colors.RED}

# The long-running services that make up a usable stack. Anything absent from
# `docker compose ps` is reported as missing rather than quietly ignored —
# `docker-proxy` sets `restart: unless-stopped`, so it can outlive a `stop` and
# make a dead stack look alive if we only counted what happened to be present.
_CORE_SERVICES = ("db", "minio", "api", "scheduler", "frontend")

_MISSING = "not created"


def _state_colour(service: stack.Service) -> str:
    return _STATE_COLOUR.get(service.state, typer.colors.YELLOW)


def _url_for(service: stack.Service) -> str:
    return f"http://localhost:{service.port}" if service.port else ""


def _print_jobs(api_base: str) -> None:
    counts = stack.job_counts(api_base)
    if counts is None:
        typer.secho(
            "  Could not reach the API for job counts.", fg=typer.colors.YELLOW
        )
        return
    if not counts:
        typer.echo("  No jobs yet.")
        return
    summary = ", ".join(f"{count} {status}" for status, count in sorted(counts.items()))
    typer.echo(f"  Jobs: {summary}")


def status() -> None:
    """Show the state of the local Heulistic stack."""
    present = {service.name: service for service in stack.services()}

    if not any(name in present for name in _CORE_SERVICES):
        if stack.api_url_is_overridden():
            # Pointed at a stack we did not start, so there are no compose
            # services to list — report what we can reach instead.
            typer.echo(f"Using {stack.api_url()} (from {stack.API_URL_ENV})")
            typer.echo()
            _print_jobs(stack.api_url())
            raise typer.Exit(code=0)
        typer.echo("Heulistic is not running.")
        typer.echo("Start it with: heulistic start")
        raise typer.Exit(code=1)

    typer.echo("Heulistic local stack")
    typer.echo()

    extras = [name for name in present if name not in _CORE_SERVICES]
    width = max(len(name) for name in (*_CORE_SERVICES, *extras)) + 2

    for name in (*_CORE_SERVICES, *extras):
        service = present.get(name)
        if service is None:
            typer.echo(
                f"  {name:<{width}}"
                + typer.style(_MISSING, fg=typer.colors.RED)
            )
            continue
        typer.echo(
            f"  {name:<{width}}"
            + typer.style(f"{service.display_state:<20}", fg=_state_colour(service))
            + _url_for(service)
        )

    typer.echo()
    api_base = stack.api_url()
    if api_base:
        _print_jobs(api_base)
    else:
        typer.secho("  API is not published — cannot read job counts.", fg=typer.colors.YELLOW)

    healthy = all(
        name in present and present[name].running for name in _CORE_SERVICES
    )
    if not healthy:
        typer.echo()
        typer.secho(
            "  Some services are not running. Try: heulistic logs",
            fg=typer.colors.YELLOW,
        )
    raise typer.Exit(code=0 if healthy else 1)
