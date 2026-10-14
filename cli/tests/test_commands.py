"""Top-level stack commands: start / stop / logs / status.

These are flat verbs rather than a `local` sub-group — `heulistic start` is
what people expect from a tool that runs a local service, and nothing has
shipped that depended on the old spelling.
"""

import pytest
from typer.testing import CliRunner

from heulistic import local_runner, preflight, stack
from heulistic.main import app

runner = CliRunner()


@pytest.fixture
def spy(monkeypatch):
    """Record calls into the runner instead of touching Docker."""
    calls = {}

    monkeypatch.setattr(preflight, "run", lambda **kw: (calls.setdefault("preflight", kw), True)[1])
    # Pass ports through untouched. Without this the test depends on whether
    # 3000/8000 happen to be free on the machine running it.
    monkeypatch.setattr(
        preflight, "resolve_ports",
        lambda fe, api, fe_explicit, api_explicit: (fe, api, []),
    )
    monkeypatch.setattr(local_runner, "start", lambda **kw: calls.setdefault("start", kw))
    monkeypatch.setattr(local_runner, "stop", lambda: calls.setdefault("stop", True))
    monkeypatch.setattr(local_runner, "logs", lambda **kw: calls.setdefault("logs", kw))
    return calls


@pytest.fixture
def running(monkeypatch):
    """Pretend a healthy stack is up — every core service present."""
    monkeypatch.setattr(
        stack,
        "services",
        lambda: [
            stack.Service("db", "running", health="healthy"),
            stack.Service("minio", "running"),
            stack.Service("api", "running", port=8000),
            stack.Service("scheduler", "running"),
            stack.Service("frontend", "running", port=3000),
        ],
    )
    monkeypatch.setattr(stack, "job_counts", lambda url: {"running": 2, "completed": 1})


# ── start ─────────────────────────────────────────────────────────────────────

def test_start_runs_preflight_then_the_stack(spy):
    result = runner.invoke(app, ["start"])
    assert result.exit_code == 0
    assert spy["preflight"] == {"frontend_port": 3000, "api_port": 8000, "use_gpu": True}
    assert spy["start"]["frontend_port"] == 3000
    assert spy["start"]["use_gpu"] is True


def test_start_passes_ports_and_token(spy):
    runner.invoke(
        app,
        ["start", "--port", "3001", "--api-port", "8001", "--hf-token", "hf_abc"],
    )
    assert spy["start"]["frontend_port"] == 3001
    assert spy["start"]["api_port"] == 8001
    assert spy["start"]["hf_token"] == "hf_abc"


def test_start_honours_no_gpu(spy):
    runner.invoke(app, ["start", "--no-gpu"])
    assert spy["preflight"]["use_gpu"] is False


def test_preflight_may_downgrade_gpu(monkeypatch, spy):
    # preflight returns the *effective* GPU setting, so a machine without a
    # driver silently falls back to CPU.
    monkeypatch.setattr(preflight, "run", lambda **kw: False)
    runner.invoke(app, ["start"])
    assert spy["start"]["use_gpu"] is False


# ── stop / logs ───────────────────────────────────────────────────────────────

def test_stop(spy):
    assert runner.invoke(app, ["stop"]).exit_code == 0
    assert spy["stop"] is True


def test_logs(spy):
    assert runner.invoke(app, ["logs"]).exit_code == 0
    assert spy["logs"] == {"follow": False}


def test_logs_follow(spy):
    runner.invoke(app, ["logs", "-f"])
    assert spy["logs"] == {"follow": True}


# ── The old `local` group is gone ─────────────────────────────────────────────

def test_local_is_no_longer_a_command():
    result = runner.invoke(app, ["local"])
    assert result.exit_code == 2


def test_help_lists_the_flat_verbs():
    output = runner.invoke(app, ["--help"]).stdout
    for command in ("start", "stop", "logs", "status", "validate-config"):
        assert command in output


# ── status ────────────────────────────────────────────────────────────────────

def test_status_when_nothing_is_running(monkeypatch):
    monkeypatch.setattr(stack, "services", lambda: [])
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 1
    assert "not running" in result.stdout.lower()
    assert "heulistic start" in result.stdout


def test_status_lists_services(running):
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "db" in result.stdout
    assert "healthy" in result.stdout
    assert "running" in result.stdout


def test_status_shows_the_urls(running):
    output = runner.invoke(app, ["status"]).stdout
    assert "http://localhost:3000" in output
    assert "http://localhost:8000" in output


def test_status_shows_job_counts(running):
    output = runner.invoke(app, ["status"]).stdout
    assert "2 running" in output
    assert "1 completed" in output


def test_status_when_the_api_is_unreachable(running, monkeypatch):
    monkeypatch.setattr(stack, "job_counts", lambda url: None)
    result = runner.invoke(app, ["status"])
    # Services are up but the API did not answer — worth saying, not a crash.
    assert "could not reach" in result.stdout.lower()
    assert result.exit_code == 0


def test_status_reports_no_jobs_yet(running, monkeypatch):
    monkeypatch.setattr(stack, "job_counts", lambda url: {})
    output = runner.invoke(app, ["status"]).stdout
    assert "no jobs" in output.lower()


def test_a_stray_container_alone_is_not_a_running_stack(monkeypatch):
    """`docker-proxy` sets `restart: unless-stopped`, so it outlives a stop.

    Counting only what `ps` returns reported a dead stack as healthy with
    exit 0 — the core services were absent, not failing.
    """
    monkeypatch.setattr(
        stack, "services", lambda: [stack.Service("docker-proxy", "running")]
    )
    monkeypatch.setattr(stack, "job_counts", lambda url: None)
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 1
    assert "not running" in result.stdout.lower()


def test_status_names_missing_core_services(monkeypatch):
    monkeypatch.setattr(
        stack,
        "services",
        lambda: [
            stack.Service("db", "running", health="healthy"),
            stack.Service("api", "running", port=8000),
        ],
    )
    monkeypatch.setattr(stack, "job_counts", lambda url: {})
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 1
    assert "scheduler" in result.stdout
    assert "not created" in result.stdout


def test_service_names_are_not_run_together(running):
    # `docker-proxy` is exactly as wide as the old fixed column, so its state
    # butted straight against the name.
    output = runner.invoke(app, ["status"]).stdout
    assert "schedulerrunning" not in output
    assert "scheduler " in output


def test_status_flags_a_degraded_stack(monkeypatch):
    monkeypatch.setattr(
        stack,
        "services",
        lambda: [
            stack.Service("db", "running", health="healthy"),
            stack.Service("scheduler", "exited"),
        ],
    )
    monkeypatch.setattr(stack, "job_counts", lambda url: None)
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 1
    assert "scheduler" in result.stdout
    assert "exited" in result.stdout


def test_status_reports_an_overridden_api_url(monkeypatch):
    monkeypatch.setattr(stack, "services", lambda: [])
    monkeypatch.setenv("HEULISTIC_API_URL", "http://localhost:8002")
    monkeypatch.setattr(stack, "job_counts", lambda url: {"completed": 1})
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "http://localhost:8002" in result.stdout
    assert "1 completed" in result.stdout


# ── Automatic port selection ──────────────────────────────────────────────────

def test_start_shifts_a_busy_default_port_and_says_so(spy, monkeypatch):
    monkeypatch.setattr(
        preflight, "resolve_ports",
        lambda fe, api, fe_x, api_x: (3001, 8000, ["Frontend port 3000 is in use — using 3001 instead"]),
    )
    result = runner.invoke(app, ["start"])
    assert result.exit_code == 0
    assert "using 3001" in result.stdout
    assert spy["start"]["frontend_port"] == 3001


def test_start_tells_resolve_ports_which_values_were_explicit(spy, monkeypatch):
    seen = {}

    def _resolve(fe, api, fe_explicit, api_explicit):
        seen.update(fe=fe, api=api, fe_explicit=fe_explicit, api_explicit=api_explicit)
        return fe, api, []

    monkeypatch.setattr(preflight, "resolve_ports", _resolve)

    runner.invoke(app, ["start", "--port", "3005"])
    assert seen["fe_explicit"] is True, "--port was given, so it must not be moved"
    assert seen["api_explicit"] is False, "--api-port was defaulted"

    runner.invoke(app, ["start"])
    assert seen["fe_explicit"] is False
    assert seen["api_explicit"] is False


def test_start_passes_an_image_tag_override(spy):
    runner.invoke(app, ["start", "--image-tag", "latest"])
    assert spy["start"]["image_tag_override"] == "latest"


def test_start_defaults_the_image_tag_to_none(spy):
    # None means "derive it from the installed version".
    runner.invoke(app, ["start"])
    assert spy["start"]["image_tag_override"] is None


def test_help_says_which_commands_need_docker():
    """People reasonably assume the whole tool needs Docker. It does not —
    `pip install` needs nothing but Python, and two commands run offline."""
    output = runner.invoke(app, ["--help"]).stdout
    assert "validate-config" in output and "init" in output
    assert "Docker" in output, "the help must say where Docker is and is not needed"


# ── start --gpus-per-job ──────────────────────────────────────────────────────

def test_start_shares_gpus_between_jobs_when_asked(spy, monkeypatch):
    from heulistic import gpu
    monkeypatch.setattr(gpu, "detect_gpus", lambda: [gpu.DetectedGpu("A10G", 23028)] * 4)
    result = runner.invoke(app, ["start", "--gpus-per-job", "1"])
    assert result.exit_code == 0, result.output
    assert spy["start"]["gpus_per_job"] == 1
    assert spy["start"]["gpu_count"] == 4


def test_start_gives_one_job_every_gpu_by_default(spy):
    runner.invoke(app, ["start"])
    assert spy["start"]["gpus_per_job"] == 0


def test_start_refuses_more_gpus_per_job_than_the_machine_has(spy, monkeypatch):
    from heulistic import gpu
    monkeypatch.setattr(gpu, "detect_gpus", lambda: [gpu.DetectedGpu("A10G", 23028)] * 2)
    result = runner.invoke(app, ["start", "--gpus-per-job", "3"])
    assert result.exit_code != 0
    assert "2 GPU" in result.output
    assert "start" not in spy


def test_gpu_sharing_reaches_the_scheduler():
    from heulistic import local_runner
    env = local_runner._env(3000, 8000, None, True, gpus_per_job=2, gpu_count=4)
    assert env["GPUS_PER_JOB"] == "2"
    assert env["LOCAL_GPU_COUNT"] == "4"
