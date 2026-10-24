"""`heulistic init` — write a starter config that actually fits the machine.

Both reference tools hand you something that runs immediately. Since we have
the Axolotl schema, a VRAM estimator and GPU detection, the generated config
can be sized to the card the user actually has instead of a generic sample.
"""

import pytest
import yaml
from typer.testing import CliRunner

from heulistic import gpu
from heulistic.main import app
from heulistic.axolotl import validate
from heulistic.axolotl.vram import Hardware, vram_required_gb

runner = CliRunner()


@pytest.fixture
def card(monkeypatch):
    def _set(vram_gb=None):
        if vram_gb is None:
            monkeypatch.setattr(gpu, "detect_hardware", lambda: None)
        else:
            monkeypatch.setattr(
                gpu, "detect_hardware",
                lambda: Hardware(total_vram_gb=vram_gb, gpu_count=1, label=f"{vram_gb}GB card"),
            )

    return _set


def run(*args):
    return runner.invoke(app, ["init", *args])


def written(tmp_path, name="config.yaml"):
    return yaml.safe_load((tmp_path / name).read_text())


# ── Output ────────────────────────────────────────────────────────────────────

def test_writes_a_config(card, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    card(24)
    assert run().exit_code == 0
    assert (tmp_path / "config.yaml").exists()


def test_generated_config_passes_our_own_validator(card, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    card(24)
    run()
    diagnostics = validate((tmp_path / "config.yaml").read_text())
    assert [d for d in diagnostics if d.level == "error"] == []


def test_generated_config_has_no_warnings_either(card, tmp_path, monkeypatch):
    # A starter config that immediately warns would undercut the tool.
    monkeypatch.chdir(tmp_path)
    card(24)
    run()
    diagnostics = validate((tmp_path / "config.yaml").read_text())
    assert [d for d in diagnostics if d.level == "warning"] == []


def test_refuses_to_overwrite_without_force(card, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    card(24)
    (tmp_path / "config.yaml").write_text("existing: true\n")
    result = run()
    assert result.exit_code == 1
    assert "--force" in result.stdout
    assert (tmp_path / "config.yaml").read_text() == "existing: true\n"


def test_force_overwrites(card, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    card(24)
    (tmp_path / "config.yaml").write_text("existing: true\n")
    assert run("--force").exit_code == 0
    assert "existing" not in (tmp_path / "config.yaml").read_text()


def test_custom_output_path(card, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    card(24)
    assert run("--output", "my.yaml").exit_code == 0
    assert (tmp_path / "my.yaml").exists()


# ── Sizing ────────────────────────────────────────────────────────────────────

def test_small_card_gets_a_config_that_fits(card, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    card(8)
    run()
    cfg = written(tmp_path)
    assert vram_required_gb(cfg) < 8, "the suggested model must fit the card"
    assert cfg["adapter"] == "qlora"
    assert cfg["load_in_4bit"] is True


def test_a_tiny_card_gets_a_smaller_model_than_a_roomy_one(card, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    card(24)
    run()
    roomy = written(tmp_path)
    card(4)
    run("--force")
    tiny = written(tmp_path)
    assert vram_required_gb(tiny) < vram_required_gb(roomy)
    assert tiny["base_model"] != roomy["base_model"]


def test_the_ladder_is_capped_for_a_starter_config(card, tmp_path, monkeypatch):
    """A huge card should not be handed a 70B first run.

    The point of `init` is a run that finishes and works; maximising GPU use
    would mean a multi-hour first experience and a very large download.
    """
    monkeypatch.chdir(tmp_path)
    card(200)
    run()
    assert vram_required_gb(written(tmp_path)) < 10


def test_no_gpu_still_produces_a_valid_config(card, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    card(None)
    result = run()
    assert result.exit_code == 0
    assert [d for d in validate((tmp_path / "config.yaml").read_text())
            if d.level == "error"] == []
    assert "no nvidia gpu" in result.stdout.lower()


def test_explicit_model_is_respected(card, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    card(24)
    run("--model", "Qwen/Qwen2.5-0.5B")
    assert written(tmp_path)["base_model"] == "Qwen/Qwen2.5-0.5B"


def test_warns_when_the_chosen_model_will_not_fit(card, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    card(8)
    result = run("--model", "meta-llama/Meta-Llama-3-70B")
    assert result.exit_code == 0
    assert "vram" in result.stdout.lower()


def test_reports_the_detected_card(card, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    card(24)
    assert "24GB card" in run().stdout


def test_next_step_is_printed(card, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    card(24)
    out = run().stdout
    assert "heulistic train" in out


def test_the_ladder_contains_no_gated_models(card, tmp_path, monkeypatch):
    """A starter config must be trainable without a HuggingFace token.

    `init` first shipped meta-llama models, which are gated: the API's own
    pre-flight check rejected the very first `heulistic train` with "add a
    HuggingFace token with approved access". Found by running the flow
    end-to-end against a live stack.
    """
    from heulistic import init_cmd

    gated_orgs = ("meta-llama/", "mistralai/", "google/")
    assert all(
        not m.startswith(gated_orgs) for m in init_cmd._LADDER
    ), init_cmd._LADDER
    assert not init_cmd._FALLBACK_MODEL.startswith(gated_orgs)

    # ...and whatever gets written must be one of them.
    monkeypatch.chdir(tmp_path)
    card(24)
    run()
    assert written(tmp_path)["base_model"] in init_cmd._LADDER


# ── Several GPUs ──────────────────────────────────────────────────────────────
# `init` writes QLoRA, which is never sharded: each card holds the whole model,
# so the budget is one card's memory, not the sum of them.

@pytest.fixture
def cards(monkeypatch):
    def _set(count, per_card_gb):
        monkeypatch.setattr(
            gpu, "detect_hardware",
            lambda: Hardware(total_vram_gb=count * per_card_gb, gpu_count=count,
                             label=f"{count}x {per_card_gb}GB card"),
        )

    return _set


def test_several_small_cards_are_sized_by_one_card(cards, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cards(4, 2)
    run()
    assert vram_required_gb(written(tmp_path)) < 2


def test_warns_when_the_model_fits_the_sum_but_not_one_card(cards, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cards(4, 2)
    result = run("--model", "Qwen/Qwen2.5-7B")  # ~4 GB: under 8 total, over 2 per card
    assert "vram" in result.stdout.lower()
    assert "per gpu" in result.stdout.lower()
