"""`heulistic validate-config`.

The command must work with nothing running — no Docker, no stack, no network.
Its value is instant feedback on a file, so anything that would make it depend
on a live service belongs elsewhere.

Exit codes: 0 clean, 1 problems found, 2 bad invocation (typer's default).
"""

import json
import os

import pytest
from typer.testing import CliRunner

from heulistic import gpu
from heulistic.main import app

runner = CliRunner()

CLEAN = """\
base_model: mistralai/Mistral-7B-v0.1
learning_rate: 0.0002
adapter: qlora
load_in_4bit: true
sequence_len: 2048
micro_batch_size: 1
gradient_accumulation_steps: 4
num_epochs: 3
save_steps: 100
bf16: auto
gradient_checkpointing: true
datasets:
  - path: tatsu-lab/alpaca
    type: alpaca
"""

WARNING_ONLY = CLEAN.replace("sequence_len: 2048", "sequence_len: 16384")
HAS_ERROR = CLEAN.replace("num_epochs: 3", "num_epochs: 0")


@pytest.fixture
def config(tmp_path):
    def _write(text, name="config.yaml"):
        path = tmp_path / name
        path.write_text(text)
        return str(path)

    return _write


def run(*args):
    return runner.invoke(app, ["validate-config", *args])


# ── Exit codes ────────────────────────────────────────────────────────────────

def test_clean_config_exits_zero(config):
    result = run(config(CLEAN))
    assert result.exit_code == 0
    assert "No problems" in result.stdout


def test_errors_exit_nonzero(config):
    result = run(config(HAS_ERROR))
    assert result.exit_code == 1
    assert "num_epochs" in result.stdout


def test_warnings_alone_still_exit_zero(config):
    result = run(config(WARNING_ONLY))
    assert result.exit_code == 0
    assert "sequence_len" in result.stdout


def test_strict_turns_warnings_into_failure(config):
    result = run(config(WARNING_ONLY), "--strict")
    assert result.exit_code == 1


def test_strict_still_passes_a_clean_config(config):
    # CLEAN emits neither warnings nor info, so --strict must not trip on it.
    assert run(config(CLEAN), "--strict").exit_code == 0


def test_missing_file_is_a_usage_error(tmp_path):
    result = run(str(tmp_path / "nope.yaml"))
    assert result.exit_code == 2


def test_unreadable_yaml_reports_the_syntax_error(config):
    result = run(config("base_model: [unclosed\n"))
    assert result.exit_code == 1
    assert "yaml" in result.stdout.lower()


# ── Output ────────────────────────────────────────────────────────────────────

def test_output_shows_level_field_and_line(config):
    result = run(config(CLEAN + "bogus_field: 1\n"))
    assert "warning" in result.stdout
    assert "bogus_field" in result.stdout
    assert "15" in result.stdout  # the line the unknown field sits on


def test_summary_counts_are_reported(config):
    result = run(config(HAS_ERROR))
    assert "1 error" in result.stdout


def test_quiet_hides_warnings_and_hints(config):
    noisy = "base_model: org/m-7b\nlearning_rate: 0.0002\nadapter: qlora\n"
    full = run(config(noisy))
    quiet = run(config(noisy), "--quiet")
    assert "info" in full.stdout
    assert "info" not in quiet.stdout


def test_json_output_is_machine_readable(config):
    result = run(config(HAS_ERROR), "--json")
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["counts"]["error"] >= 1
    codes = {d["code"] for d in payload["diagnostics"]}
    assert "epochs.invalid" in codes
    assert all({"level", "code", "message"} <= set(d) for d in payload["diagnostics"])


def test_json_output_on_a_clean_config(config):
    payload = json.loads(run(config(CLEAN), "--json").stdout)
    assert payload["ok"] is True
    assert payload["diagnostics"] == []


# ── Hardware checks ───────────────────────────────────────────────────────────

def test_no_hardware_flag_means_no_vram_check(config):
    # A 70B full fine-tune needs ~1.3 TB, but without a target we say nothing.
    big = "base_model: org/m-70b\nlearning_rate: 0.0002\n"
    payload = json.loads(run(config(big), "--json").stdout)
    assert not [d for d in payload["diagnostics"] if d["code"].startswith("vram.")]


def test_explicit_vram_enables_the_fit_check(config):
    big = "base_model: org/m-70b\nlearning_rate: 0.0002\n"
    result = run(config(big), "--vram", "8")
    assert result.exit_code == 1
    assert "VRAM" in result.stdout


def test_vram_and_gpus_combine(config):
    big = "base_model: org/m-7b\nlearning_rate: 0.0002\nadapter: lora\n"
    payload = json.loads(
        run(config(big), "--vram", "88", "--gpus", "4", "--json").stdout
    )
    assert "hardware.multi_gpu" in {d["code"] for d in payload["diagnostics"]}


def test_gpu_flag_uses_the_detected_card(config, monkeypatch):
    monkeypatch.setattr(gpu, "_nvidia_smi_output", lambda: "NVIDIA A10G, 23028\n")
    big = "base_model: org/m-70b\nlearning_rate: 0.0002\n"
    result = run(config(big), "--gpu")
    assert result.exit_code == 1
    assert "A10G" in result.stdout


def test_gpu_flag_without_a_gpu_warns_but_still_validates(config, monkeypatch):
    monkeypatch.setattr(gpu, "_nvidia_smi_output", lambda: None)
    result = run(config(CLEAN), "--gpu")
    assert result.exit_code == 0
    assert "no nvidia gpu" in result.stdout.lower()
    assert "No problems" in result.stdout


def test_gpu_and_vram_together_is_rejected(config):
    # Usage errors belong on stderr so that `--json` output stays parseable
    # when a caller pipes stdout.
    result = run(config(CLEAN), "--gpu", "--vram", "8")
    assert result.exit_code == 2
    assert "--gpu" in result.stderr and "--vram" in result.stderr
    assert result.stdout == ""


def test_target_is_named_in_the_header(config, monkeypatch):
    monkeypatch.setattr(gpu, "_nvidia_smi_output", lambda: "NVIDIA A10G, 23028\n")
    result = run(config(CLEAN), "--gpu")
    assert "A10G" in result.stdout


# ── Module entry point ────────────────────────────────────────────────────────

def test_python_m_heulistic_runs_the_cli(config, tmp_path):
    """`python -m heulistic` must work, not silently no-op.

    Before `__main__.py` existed this exited 0 with no output, which looks
    exactly like a clean config.
    """
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(repo / "cli"),
    }
    result = subprocess.run(
        [sys.executable, "-m", "heulistic", "validate-config", config(HAS_ERROR)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 1
    assert "num_epochs" in result.stdout
