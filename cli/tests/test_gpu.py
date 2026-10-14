"""Local GPU detection.

`nvidia-smi` is the only detection route. Everything here must degrade quietly
when it is missing — a laptop with no NVIDIA card is an ordinary place to run
`heulistic validate-config`, not an error.
"""

import pytest

from heulistic import gpu


@pytest.fixture
def smi(monkeypatch):
    """Replace the nvidia-smi call with canned output."""

    def _set(output):
        monkeypatch.setattr(gpu, "_nvidia_smi_output", lambda: output)

    return _set


# ── Parsing ───────────────────────────────────────────────────────────────────

def test_single_gpu(smi):
    smi("NVIDIA GeForce RTX 4060 Laptop GPU, 8188\n")
    cards = gpu.detect_gpus()
    assert len(cards) == 1
    assert cards[0].name == "NVIDIA GeForce RTX 4060 Laptop GPU"
    assert cards[0].vram_mb == 8188


def test_multiple_gpus(smi):
    smi("NVIDIA A10G, 23028\nNVIDIA A10G, 23028\n")
    assert [c.vram_mb for c in gpu.detect_gpus()] == [23028, 23028]


def test_blank_and_malformed_lines_are_skipped(smi):
    smi("NVIDIA A10G, 23028\n\ngarbage-without-a-comma\nNVIDIA A10G, notanumber\n")
    cards = gpu.detect_gpus()
    assert len(cards) == 1
    assert cards[0].name == "NVIDIA A10G"


def test_no_nvidia_smi(smi):
    smi(None)
    assert gpu.detect_gpus() == []


def test_empty_output(smi):
    smi("")
    assert gpu.detect_gpus() == []


# ── Conversion to a validation target ─────────────────────────────────────────

def test_hardware_from_one_card(smi):
    smi("NVIDIA GeForce RTX 4060 Laptop GPU, 8188\n")
    hw = gpu.detect_hardware()
    assert hw is not None
    assert hw.gpu_count == 1
    assert hw.total_vram_gb == pytest.approx(7.99, abs=0.01)
    assert hw.label == "NVIDIA GeForce RTX 4060 Laptop GPU"


def test_hardware_sums_vram_across_identical_cards(smi):
    smi("NVIDIA A10G, 23028\nNVIDIA A10G, 23028\n")
    hw = gpu.detect_hardware()
    assert hw.gpu_count == 2
    assert hw.total_vram_gb == pytest.approx(44.98, abs=0.01)
    assert hw.label == "2x NVIDIA A10G"


def test_hardware_labels_a_mixed_rig_generically(smi):
    smi("NVIDIA A10G, 23028\nNVIDIA GeForce RTX 3090, 24576\n")
    hw = gpu.detect_hardware()
    assert hw.gpu_count == 2
    assert "2 GPUs" in hw.label


def test_hardware_is_none_without_a_gpu(smi):
    smi(None)
    assert gpu.detect_hardware() is None


def test_zero_vram_is_not_a_usable_target(smi):
    # Hardware() rejects non-positive VRAM; detection must not raise.
    smi("Weird Device, 0\n")
    assert gpu.detect_hardware() is None


def test_a_mixed_rig_is_sized_by_its_smallest_card(smi):
    """Every card holds a full copy under data parallelism, and ZeRO-3 gives
    each an equal share, so the smallest card is the one that runs out — the
    same rule the trainer uses when it plans the job."""
    smi("NVIDIA GeForce RTX 3070, 8192\nNVIDIA GeForce RTX 3090, 24576\n")
    hw = gpu.detect_hardware()
    assert hw.gpu_count == 2
    assert hw.per_gpu_vram_gb == pytest.approx(8.0, abs=0.01)
