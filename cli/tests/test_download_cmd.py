"""`heulistic download` — get the trained model out.

The model lands in MinIO inside Docker, so without this the output of a local
run is effectively trapped.
"""

import pytest
from typer.testing import CliRunner

from heulistic import client, download_cmd, stack
from heulistic.main import app

runner = CliRunner()


@pytest.fixture
def api(monkeypatch, tmp_path):
    state = {"fetched": None}
    monkeypatch.setattr(stack, "api_url", lambda: "http://localhost:8000")
    monkeypatch.setattr(
        client, "get_job",
        lambda u, j: {"job_id": j, "status": "completed"},
    )
    monkeypatch.setattr(
        client, "download_url",
        lambda u, j: "http://localhost:8001/local-download/models/j/model.tar.gz",
    )

    def _fetch(url, dest):
        state["fetched"] = (url, dest)
        dest.write_bytes(b"tarball")
        return dest

    monkeypatch.setattr(download_cmd, "_fetch", _fetch)
    return state


def run(*args):
    return runner.invoke(app, ["download", *args])


def test_refuses_when_the_stack_is_down(monkeypatch):
    monkeypatch.setattr(stack, "api_url", lambda: None)
    result = run("job-1")
    assert result.exit_code == 1
    assert "heulistic start" in result.stdout


def test_downloads_to_the_working_directory(api, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run("job-1")
    assert result.exit_code == 0
    assert (tmp_path / "job-1-model.tar.gz").exists()


def test_honours_an_explicit_output_path(api, tmp_path):
    dest = tmp_path / "mine.tar.gz"
    result = run("job-1", "--output", str(dest))
    assert result.exit_code == 0
    assert dest.exists()


def test_writes_into_a_directory_if_given_one(api, tmp_path):
    result = run("job-1", "--output", str(tmp_path))
    assert result.exit_code == 0
    assert (tmp_path / "job-1-model.tar.gz").exists()


def test_refuses_to_overwrite_without_force(api, tmp_path):
    dest = tmp_path / "mine.tar.gz"
    dest.write_bytes(b"existing")
    result = run("job-1", "--output", str(dest))
    assert result.exit_code == 1
    assert "--force" in result.stdout
    assert dest.read_bytes() == b"existing"


def test_force_overwrites(api, tmp_path):
    dest = tmp_path / "mine.tar.gz"
    dest.write_bytes(b"existing")
    assert run("job-1", "--output", str(dest), "--force").exit_code == 0
    assert dest.read_bytes() == b"tarball"


def test_unfinished_job_is_rejected_with_its_status(api, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        client, "get_job", lambda u, j: {"job_id": j, "status": "running"}
    )
    result = run("job-1")
    assert result.exit_code == 1
    assert "running" in result.stdout


def test_url_only_prints_without_downloading(api, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run("job-1", "--url-only")
    assert result.exit_code == 0
    assert "local-download/models/j/model.tar.gz" in result.stdout
    assert api["fetched"] is None


def test_api_errors_are_reported(api, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        client, "download_url",
        lambda u, j: (_ for _ in ()).throw(client.ApiError("Model file not found")),
    )
    result = run("job-1")
    assert result.exit_code == 1
    assert "Model file not found" in result.stdout
