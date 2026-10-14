"""Running an exported model: GGUF sizes and the memory they need.

Training fit is vram.py's job; this is the other end — after export, which
quantization will run on the machine the user actually has. Bits per weight are
llama.cpp's effective figures, including the per-block scales.
"""

import pytest

from heulistic_axolotl.inference import (
    GGUF_QUANTS,
    gguf_size_gb,
    inference_memory_gb,
    recommend_quant,
)


def test_offered_quants_run_from_smallest_to_best():
    assert list(GGUF_QUANTS) == ["Q4_K_M", "Q5_K_M", "Q8_0"]


@pytest.mark.parametrize("quant,size", [("Q4_K_M", 4.3), ("Q5_K_M", 5.0), ("Q8_0", 7.4)])
def test_file_size_for_a_7b_model(quant, size):
    assert gguf_size_gb(7, quant) == pytest.approx(size, abs=0.1)


def test_running_needs_more_than_the_file():
    # Context (KV cache) and runtime buffers come on top of the weights.
    assert inference_memory_gb(7, "Q4_K_M") > gguf_size_gb(7, "Q4_K_M")


def test_a_7b_q4_runs_in_8_gb():
    # The common hobbyist case: a 7–8B model on an 8 GB card or laptop.
    assert inference_memory_gb(7, "Q4_K_M") < 8


def test_unknown_quant_is_an_error():
    with pytest.raises(ValueError, match="Q4_K_M"):
        gguf_size_gb(7, "Q3_FOO")


class TestRecommend:
    """The best-quality quant whose running footprint fits in 90% of memory.

    7B footprints: Q4_K_M ~5.4 GB, Q5_K_M ~6.2 GB, Q8_0 ~9.1 GB.
    """

    @pytest.mark.parametrize("available,expected", [
        (24, "Q8_0"),    # a 24 GB card runs the near-lossless one
        (12, "Q8_0"),    # 10.8 GB usable
        (8, "Q5_K_M"),   # 7.2 GB usable: Q8_0 does not fit, Q5_K_M does
        (6.5, "Q4_K_M"), # 5.85 GB usable
    ])
    def test_best_quality_that_fits(self, available, expected):
        assert recommend_quant(7, available_gb=available) == expected

    def test_leaves_headroom_rather_than_filling_memory(self):
        # Another app, the desktop, the browser — memory is never all free.
        need = inference_memory_gb(7, "Q8_0")
        assert recommend_quant(7, available_gb=need) != "Q8_0"

    def test_nothing_fits(self):
        assert recommend_quant(7, available_gb=5) is None
        assert recommend_quant(70, available_gb=24) is None

    def test_without_a_known_machine_the_default_is_q4(self):
        assert recommend_quant(7, available_gb=None) == "Q4_K_M"


# ── Exporting ─────────────────────────────────────────────────────────────────

from heulistic_axolotl.inference import export_scratch_gb  # noqa: E402


def test_export_scratch_covers_merged_weights_and_the_full_precision_gguf():
    # At the peak both exist, each the size of the fp16 model (2 bytes/param).
    assert export_scratch_gb(70) > 2 * (70 * 2)
    assert export_scratch_gb(1) < 20
