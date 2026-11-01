"""`heulistic login` / `logout`, and the commands that follow a login."""

import pytest
from typer.testing import CliRunner

from heulistic import client, export_cmd, session, stack
from heulistic.main import app

runner = CliRunner()

ME = {"email": "you@example.com", "credit_balance_cents": 1250, "billing_enabled": True}


@pytest.fixture
def api(monkeypatch):
    """Record which URL and token each API call used."""
    calls = []

    def record(name, result):
        def fn(api_url, *args, **kwargs):
            calls.append((name, api_url, client._headers()["Authorization"]))
            if isinstance(result, Exception):
                raise result
            return result
        return fn

    monkeypatch.setattr(stack, "api_url", lambda: "http://localhost:8000")
    monkeypatch.setattr(client, "me", record("me", ME))
    monkeypatch.setattr(client, "submit", record("submit", {"job_id": "job-1", "message": "Job queued."}))
    monkeypatch.setattr(client, "get_job", record("get_job", {"job_id": "job-1", "status": "completed"}))
    monkeypatch.setattr(client, "download_url", record("download_url", "https://s3/x/model.tar.gz"))
    monkeypatch.setattr(client, "stream_logs", lambda *a, **k: iter([("done", "completed")]))
    return calls


# ── login / logout ────────────────────────────────────────────────────────────


def test_login_checks_the_key_then_saves_it(api):
    result = runner.invoke(app, ["login", "--key", "hk_good"])
    assert result.exit_code == 0, result.output
    assert "you@example.com" in result.output
    assert "$12.50" in result.output
    assert api == [("me", session.CLOUD_API_URL, "Bearer hk_good")]
    assert session.resolve().token == "hk_good"


def test_login_prompts_without_echoing_the_key(api):
    result = runner.invoke(app, ["login"], input="hk_typed\n")
    assert result.exit_code == 0, result.output
    assert "hk_typed" not in result.output
    assert session.resolve().token == "hk_typed"


def test_a_rejected_key_is_not_saved(api, monkeypatch):
    def refuse(api_url):
        raise client.ApiError("Invalid or revoked API key")

    monkeypatch.setattr(client, "me", refuse)
    result = runner.invoke(app, ["login", "--key", "hk_bad"])
    assert result.exit_code == 1
    assert "Invalid or revoked API key" in result.output
    assert session.resolve().cloud is False


def test_something_that_is_not_a_key_is_refused_before_asking_the_server(api):
    result = runner.invoke(app, ["login", "--key", "eyJhbGciOi.jwt"])
    assert result.exit_code == 1
    assert "hk_" in result.output
    assert api == []


def test_login_can_target_another_api(api):
    runner.invoke(app, ["login", "--key", "hk_good", "--api-url", "http://localhost:8000/"])
    assert session.resolve().api_url == "http://localhost:8000"


def test_logout(api):
    runner.invoke(app, ["login", "--key", "hk_good"])
    result = runner.invoke(app, ["logout"])
    assert result.exit_code == 0
    assert session.resolve().cloud is False


# ── Commands after a login ────────────────────────────────────────────────────


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "base_model: Qwen/Qwen2.5-0.5B\nadapter: lora\nlearning_rate: 0.0002\n"
        "micro_batch_size: 2\nnum_epochs: 1\nsequence_len: 512\n"
        "datasets:\n  - path: tatsu-lab/alpaca\n    type: alpaca\n"
    )
    return path


def test_train_goes_to_the_cloud_when_logged_in(api, config):
    session.save("hk_good", session.CLOUD_API_URL)
    result = runner.invoke(app, ["train", str(config), "--detach"])
    assert result.exit_code == 0, result.output
    assert ("submit", session.CLOUD_API_URL, "Bearer hk_good") in api
    assert "Heulistic Cloud" in result.output


def test_local_sends_one_command_to_the_local_stack(api, config):
    session.save("hk_good", session.CLOUD_API_URL)
    result = runner.invoke(app, ["train", str(config), "--detach", "--local"])
    assert result.exit_code == 0, result.output
    assert ("submit", "http://localhost:8000", f"Bearer {client.DEV_TOKEN}") in api


def test_logged_out_train_still_needs_the_local_stack(api, config, monkeypatch):
    monkeypatch.setattr(stack, "api_url", lambda: None)
    result = runner.invoke(app, ["train", str(config), "--detach"])
    assert result.exit_code == 1
    assert "heulistic start" in result.output


def test_download_from_the_cloud(api, monkeypatch, tmp_path):
    session.save("hk_good", session.CLOUD_API_URL)
    from heulistic import download_cmd
    monkeypatch.setattr(download_cmd, "_fetch", lambda url, dest: dest.write_bytes(b"m") or dest)
    result = runner.invoke(app, ["download", "job-1", "-o", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert ("download_url", session.CLOUD_API_URL, "Bearer hk_good") in api


def test_a_cloud_export_skips_the_local_disk_check(api, monkeypatch):
    # The export runs on a cloud machine sized for it; this machine's disk is irrelevant.
    session.save("hk_good", session.CLOUD_API_URL)
    monkeypatch.setattr(client, "get_job", lambda u, j: {
        "job_id": j, "kind": "train", "status": "completed", "exports": [],
        "config_yaml": "base_model: org/m-70b\nadapter: lora\n",
    } if j == "src" else {"job_id": j, "status": "failed", "error_message": "stop here"})
    monkeypatch.setattr(client, "start_export", lambda u, j, q: {"job_id": "exp"})
    monkeypatch.setattr(export_cmd, "_free_disk_gb", lambda: pytest.fail("checked local disk"))
    result = runner.invoke(app, ["export", "src", "-q", "Q4_K_M"])
    assert "stop here" in result.output  # got as far as the export result


def test_status_says_who_you_are_logged_in_as(api, monkeypatch):
    session.save("hk_good", session.CLOUD_API_URL)
    monkeypatch.setattr(stack, "services", lambda: [])
    result = runner.invoke(app, ["status"])
    assert "Heulistic Cloud" in result.output


def test_a_detached_cloud_job_points_at_the_website(api, config):
    # `heulistic logs` and `heulistic status` describe the local stack; a cloud
    # job is followed on its page.
    session.save("hk_good", session.CLOUD_API_URL)
    result = runner.invoke(app, ["train", str(config), "--detach"])
    assert "https://heulistic.com/jobs/job-1" in result.output
    assert "heulistic logs" not in result.output


def test_a_detached_local_job_keeps_the_local_hints(api, config):
    result = runner.invoke(app, ["train", str(config), "--detach"])
    assert "heulistic status" in result.output
