"""Running an exported model: GGUF file sizes and the memory they need.

The counterpart to vram.py, which asks whether training fits. This asks which
quantization of the finished model will run on the user's own machine.

Like vram.py it is a heuristic, not a simulator. llama.cpp's footprint is the
weights plus the KV cache and compute buffers, which grow with context length
and vary by architecture; a flat 15% plus 0.5 GB covers a 4K context on the
common 1–13B models without pretending to more precision than that.
"""

from __future__ import annotations

__all__ = [
    "GGUF_QUANTS", "export_scratch_gb", "gguf_size_gb", "inference_memory_gb", "recommend_quant",
]

# Effective bits per weight, block scales included, smallest file first. Only
# these three are offered: Q4_K_M is the usual default, Q5_K_M a step up, and
# Q8_0 is close to lossless. Below Q4 quality falls off fast for small models.
GGUF_QUANTS: dict[str, float] = {
    "Q4_K_M": 4.89,
    "Q5_K_M": 5.70,
    "Q8_0": 8.50,
}

_RUNTIME_FACTOR = 1.15
_RUNTIME_OVERHEAD_GB = 0.5
# Memory is never all free — the OS, a browser, the desktop.
_USABLE_FRACTION = 0.9
_DEFAULT_QUANT = "Q4_K_M"


def gguf_size_gb(params_b: float, quant: str) -> float:
    """Size of the GGUF file, in GB."""
    try:
        bits = GGUF_QUANTS[quant]
    except KeyError:
        raise ValueError(
            f"Unknown quantization {quant!r}; choose one of {', '.join(GGUF_QUANTS)}"
        ) from None
    return params_b * bits / 8


def inference_memory_gb(params_b: float, quant: str) -> float:
    """Memory needed to run the model at a 4K context — VRAM on a GPU, RAM
    (unified memory on a Mac) otherwise."""
    return gguf_size_gb(params_b, quant) * _RUNTIME_FACTOR + _RUNTIME_OVERHEAD_GB


def recommend_quant(params_b: float, available_gb: float | None) -> str | None:
    """The best-quality quantization that runs in `available_gb`, or None.

    Without a known machine the default is Q4_K_M, the format most local
    runtimes assume.
    """
    if available_gb is None:
        return _DEFAULT_QUANT
    usable = available_gb * _USABLE_FRACTION
    fitting = [q for q in GGUF_QUANTS if inference_memory_gb(params_b, q) <= usable]
    return fitting[-1] if fitting else None


def export_scratch_gb(params_b: float) -> float:
    """Disk an export needs at its peak, in GB, beyond the base model download.

    Merging writes the fp16 weights; converting writes a full-precision GGUF
    from them; the two exist together until the quantized file is made from
    the second. 10% headroom on each, the quantized file, and 10 GB slack.
    """
    fp16 = params_b * 2
    return 2 * fp16 * 1.1 + fp16 / 2 + 10
