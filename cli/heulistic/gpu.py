"""Local GPU detection via nvidia-smi.

Used to check whether a config fits the machine it is about to run on. Every
path here degrades to "no GPU found" rather than raising — running
`validate-config` on a laptop without an NVIDIA card is ordinary, not an error.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

from . import axolotl

__all__ = ["DetectedGpu", "detect_gpus", "detect_hardware"]

_MB_PER_GB = 1024
_QUERY = ["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]


@dataclass(frozen=True)
class DetectedGpu:
    name: str
    vram_mb: int


def _nvidia_smi_output() -> str | None:
    """Raw `nvidia-smi` CSV, or None when unavailable.

    Isolated so tests can supply canned output without mocking subprocess.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", *_QUERY],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def detect_gpus() -> list[DetectedGpu]:
    """Every NVIDIA GPU on this machine. Empty when none can be found."""
    output = _nvidia_smi_output()
    if not output:
        return []

    cards: list[DetectedGpu] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue
        name, _, memory = line.rpartition(",")
        try:
            vram_mb = int(memory.strip())
        except ValueError:
            continue  # a driver error lands in this column instead of a number
        cards.append(DetectedGpu(name=name.strip(), vram_mb=vram_mb))
    return cards


def _label(cards: list[DetectedGpu]) -> str:
    names = {c.name for c in cards}
    if len(cards) == 1:
        return cards[0].name
    if len(names) == 1:
        return f"{len(cards)}x {cards[0].name}"
    return f"{len(cards)} GPUs"


def detect_hardware() -> "Hardware | None":  # noqa: F821 - lazily imported
    """The local GPUs as a validation target, or None if there are none usable."""
    cards = detect_gpus()
    if not cards:
        return None
    Hardware = axolotl.Hardware
    # Sized by the smallest card: under data parallelism every card holds the
    # whole model, and ZeRO-3 gives each an equal share, so that card is the
    # one that runs out. The trainer plans with the same rule.
    smallest_gb = min(c.vram_mb for c in cards) / _MB_PER_GB
    try:
        return Hardware(
            total_vram_gb=round(smallest_gb * len(cards), 2),
            gpu_count=len(cards),
            label=_label(cards),
        )
    except ValueError:
        # A device reporting zero memory is not something we can size against.
        return None
