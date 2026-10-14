"""`heulistic train` — submit a job and follow it from the terminal.

The point of the command is that you never need the browser: validate, submit,
watch the loss scroll, get told where the model landed.
"""

import pytest
from typer.testing import CliRunner

from heulistic import client, stack
from heulistic.main import app

runner = CliRunner()

CLEAN = """\
base_model: mistralai/Mistral-7B-v0.1
learning_rate: 0.0002
adapter: qlora
load_in_4bit: true
sequence_len: 2048
micro_batch_size: 1
num_epochs: 1
save_steps: 100
bf16: auto
gradient_checkpointing: true
datasets:
  - path: tatsu-lab/alpaca
    type: alpaca
"""


@pytest.fixture
def config(tmp_path):
    def _write(text=CLEAN, name="config.yaml"):
        p = tmp_path / name
        p.write_text(text)
        return str(p)

    return _write


@pytest.fixture
def api(monkeypatch):
    """A reachable stack with a scriptable API."""
    state = {"submitted": None, "job": {"job_id": "job-1", "status": "pending"}}

    monkeypatch.setattr(stack, "api_url", lambda: "http://localhost:8000")

    def _submit(api_url, cfg, dataset=None, hf_token=None):
        state["submitted"] = {"config": cfg, "dataset": dataset, "hf_token": hf_token}
        return state["job"]

    monkeypatch.setattr(client, "submit", _submit)
    monkeypatch.setattr(client, "stream_logs", lambda u, j: iter([("done", "completed")]))
    monkeypatch.setattr(
        client, "get_job",
        lambda u, j: {"job_id": j, "status": "completed", "costs": {"actual": None}},
    )
    return state


def run(*args):
    return runner.invoke(app, ["train", *args])


# ── Preconditions ─────────────────────────────────────────────────────────────

def test_refuses_when_the_stack_is_down(config, monkeypatch):
    monkeypatch.setattr(stack, "api_url", lambda: None)
    result = run(config())
    assert result.exit_code == 1
    assert "heulistic start" in result.stdout


def test_invalid_config_is_caught_before_submitting(config, api):
    result = run(config(CLEAN.replace("num_epochs: 1", "num_epochs: 0")))
    assert result.exit_code == 1
    assert "num_epochs" in result.stdout
    assert api["submitted"] is None, "must not reach the API"


def test_validation_can_be_skipped(config, api):
    result = run(config(CLEAN.replace("num_epochs: 1", "num_epochs: 0")), "--no-validate")
    assert api["submitted"] is not None
    assert result.exit_code == 0


def test_warnings_do_not_block_submission(config, api):
    run(config(CLEAN + "totally_unknown_field: 1\n"))
    assert api["submitted"] is not None


def test_missing_config_file_is_a_usage_error(tmp_path):
    assert run(str(tmp_path / "nope.yaml")).exit_code == 2


# ── Submission ────────────────────────────────────────────────────────────────

def test_submits_and_reports_the_job_id(config, api):
    result = run(config())
    assert result.exit_code == 0
    assert "job-1" in result.stdout


def test_passes_dataset_and_token(config, api, tmp_path):
    ds = tmp_path / "data.jsonl"
    ds.write_text('{"a": 1}\n')
    run(config(), "--dataset", str(ds), "--hf-token", "hf_abc")
    assert api["submitted"]["dataset"].name == "data.jsonl"
    assert api["submitted"]["hf_token"] == "hf_abc"


def test_detach_skips_log_streaming(config, api, monkeypatch):
    streamed = []
    monkeypatch.setattr(
        client, "stream_logs",
        lambda u, j: streamed.append(j) or iter([]),
    )
    result = run(config(), "--detach")
    assert result.exit_code == 0
    assert streamed == []
    assert "heulistic logs" in result.stdout or "job-1" in result.stdout


def test_api_errors_are_reported_cleanly(config, api, monkeypatch):
    def _boom(*a, **kw):
        raise client.ApiError("Insufficient credits")

    monkeypatch.setattr(client, "submit", _boom)
    result = run(config())
    assert result.exit_code == 1
    assert "Insufficient credits" in result.stdout


# ── Following the run ─────────────────────────────────────────────────────────

def test_streams_log_lines(config, api, monkeypatch):
    monkeypatch.setattr(
        client, "stream_logs",
        lambda u, j: iter([("line", "step 1 loss 2.1"), ("line", "step 2 loss 1.8"),
                           ("done", "completed")]),
    )
    out = run(config()).stdout
    assert "step 1 loss 2.1" in out
    assert "step 2 loss 1.8" in out


def test_successful_run_exits_zero_and_points_at_the_model(config, api, monkeypatch):
    monkeypatch.setattr(
        client, "get_job",
        lambda u, j: {"job_id": j, "status": "completed", "costs": {"actual": None}},
    )
    result = run(config())
    assert result.exit_code == 0
    assert "completed" in result.stdout.lower()
    assert "heulistic download" in result.stdout


def test_failed_run_exits_nonzero_with_the_error(config, api, monkeypatch):
    monkeypatch.setattr(client, "stream_logs", lambda u, j: iter([("done", "failed")]))
    monkeypatch.setattr(
        client, "get_job",
        lambda u, j: {"job_id": j, "status": "failed",
                      "error_message": "CUDA out of memory", "costs": {"actual": None}},
    )
    result = run(config())
    assert result.exit_code == 1
    assert "CUDA out of memory" in result.stdout


def test_interrupting_the_stream_leaves_the_job_running(config, api, monkeypatch):
    def _interrupt(u, j):
        raise KeyboardInterrupt
        yield  # pragma: no cover

    monkeypatch.setattr(client, "stream_logs", _interrupt)
    result = run(config())
    # Ctrl-C detaches from the logs; it must not look like a failed run, and it
    # must tell the user the job is still going.
    assert result.exit_code == 0
    assert "still running" in result.stdout.lower()
