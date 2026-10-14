"""VRAM estimation.

This is the single heuristic for "will this config fit?". It is a faithful port
of `api/app/services/cost.py::_vram_required_gb`, which drives cloud instance
selection and therefore billing. The expected numbers below are computed from
that formula by hand so a drift in either copy shows up here.

Superseded heuristic: an earlier browser-side estimate, which used weights x1.5 (PEFT) / x4 (full) with a 0.7 gradient-checkpointing discount,
which disagreed wildly at scale — ~157 GB vs ~1344 GB for a 70B full fine-tune.
The cost.py formula wins because money depends on it.
"""

import pytest

from heulistic_axolotl.vram import (
    Hardware,
    parse_param_billions,
    training_bits,
    vram_required_gb,
)


# ── Parameter count ───────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "model,expected",
    [
        ("mistralai/Mistral-7B-v0.1", 7.0),
        ("meta-llama/Llama-3.2-1B", 1.0),
        ("meta-llama/Meta-Llama-3-70B", 70.0),
        ("Qwen/Qwen2.5-0.5B", 0.5),
        ("microsoft/phi-2", 7.0),          # unparseable → conservative default
        ("", 7.0),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 1.1),
    ],
)
def test_parse_param_billions(model, expected):
    assert parse_param_billions(model) == expected


def test_digits_adjacent_to_b_do_not_false_match():
    # The lookarounds exist so "bf16"-style tokens and version numbers such as
    # "v0.1b2" cannot be read as a parameter count.
    assert parse_param_billions("org/model-3b") == 3.0
    assert parse_param_billions("org/model-13b") == 13.0


# ── Training precision ────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "config,expected_bits",
    [
        ({}, 16),
        ({"load_in_4bit": True}, 4),
        ({"load_in_8bit": True}, 8),
        ({"adapter": "qlora"}, 4),          # qlora implies 4-bit
        ({"adapter": "QLoRA"}, 4),          # case-insensitive
        ({"adapter": "lora"}, 16),
        ({"adapter": "lora", "load_in_4bit": True}, 4),
    ],
)
def test_training_bits(config, expected_bits):
    assert training_bits(config) == expected_bits


# ── Required VRAM ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "config,expected_gb",
    [
        # PEFT: P x (bits/8) x 1.2
        ({"base_model": "org/m-7b", "adapter": "qlora"}, 4.2),
        ({"base_model": "org/m-7b", "adapter": "lora"}, 16.8),
        ({"base_model": "org/m-13b", "adapter": "lora", "load_in_8bit": True}, 15.6),
        ({"base_model": "org/m-70b", "adapter": "qlora"}, 42.0),
        # Full fine-tune: P x 16 x 1.2
        ({"base_model": "org/m-7b"}, 134.4),
        ({"base_model": "org/m-1b"}, 19.2),
    ],
)
def test_vram_required_gb(config, expected_gb):
    assert vram_required_gb(config) == pytest.approx(expected_gb)


def test_full_finetune_dwarfs_peft():
    peft = vram_required_gb({"base_model": "org/m-70b", "adapter": "qlora"})
    full = vram_required_gb({"base_model": "org/m-70b"})
    assert full > peft * 30


# ── Hardware targets ──────────────────────────────────────────────────────────

def test_hardware_totals():
    assert Hardware(total_vram_gb=22, gpu_count=1).total_vram_gb == 22
    multi = Hardware(total_vram_gb=88, gpu_count=4, label="g5.12xlarge")
    assert multi.gpu_count == 4
    assert multi.label == "g5.12xlarge"


def test_hardware_fit_thresholds():
    # Over capacity is a hard no; above 75% is tight. The hard line used to be
    # 95%, while instance selection used 100% — so the API rented hardware the
    # UI then refused to submit to. The estimate already carries a 20%
    # overhead factor, so the extra 5% margin was double-counting.
    hw = Hardware(total_vram_gb=100, gpu_count=1)
    assert hw.verdict_for(101) == "over"
    assert hw.verdict_for(100) == "tight"
    assert hw.verdict_for(96) == "tight"
    assert hw.verdict_for(80) == "tight"
    assert hw.verdict_for(50) == "fits"


def test_hardware_rejects_nonsense():
    with pytest.raises(ValueError):
        Hardware(total_vram_gb=0, gpu_count=1)
    with pytest.raises(ValueError):
        Hardware(total_vram_gb=16, gpu_count=0)
