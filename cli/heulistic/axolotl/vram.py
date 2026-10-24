"""VRAM estimation and hardware targets.

A faithful port of `api/app/services/cost.py::_vram_required_gb`. That formula
already picks cloud instances and therefore sets prices, so it is the one
heuristic of record — the frontend's competing estimate is retired rather than
reconciled.

The estimate is a conservative upper bound, not a simulator: it ignores
activation memory growth with sequence length and assumes gradient
checkpointing on full fine-tunes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "Hardware",
    "ParallelismPlan",
    "parse_param_billions",
    "plan_parallelism",
    "training_bits",
    "vram_required_gb",
]

# A parameter count is digits followed by "b", not adjacent to other digits —
# so "Llama-3.2-1B" reads as 1B and version fragments do not false-match.
_PARAMS_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)b(?!\d)")

_DEFAULT_PARAMS_B = 7.0
_PEFT_ADAPTERS = ("lora", "qlora")

# Bytes per parameter of overhead beyond the weights themselves — adapter,
# optimiser state and activations for PEFT; the full optimiser for the rest.
_OVERHEAD_FACTOR = 1.2
_FULL_FINETUNE_BYTES_PER_PARAM = 16.0

# Fractions of capacity. Over the line is a hard no; above the tight line is a
# warning. There is no extra margin below 100%: the 1.2 overhead factor above
# is the margin, and a second one made the validator reject jobs that instance
# selection had already placed.
_OVER_THRESHOLD = 1.0
_TIGHT_THRESHOLD = 0.75


def parse_param_billions(base_model: str) -> float:
    """Parameter count in billions, read from the model name.

    Falls back to 7B when the name says nothing — deliberately conservative,
    since under-estimating produces an OOM hours into a paid training run.
    """
    match = _PARAMS_RE.search((base_model or "").lower())
    return float(match.group(1)) if match else _DEFAULT_PARAMS_B


def training_bits(config: dict) -> int:
    """Effective bit-width of the base model weights during training."""
    adapter = str(config.get("adapter") or "").lower()
    if config.get("load_in_4bit") or adapter == "qlora":
        return 4
    if config.get("load_in_8bit"):
        return 8
    return 16


def vram_required_gb(config: dict) -> float:
    """Estimated VRAM needed to train this config, in GB.

    PEFT (LoRA/QLoRA)  base weights + adapter + optimiser + activations
                       = P x (bits/8) x 1.2
    Full fine-tune     weights(2) + grads(2) + fp32 master(4) + Adam m,v(8)
                       = P x 16 x 1.2
    """
    params_b = parse_param_billions(config.get("base_model") or "")
    adapter = str(config.get("adapter") or "").lower()

    if adapter in _PEFT_ADAPTERS:
        bytes_per_param = training_bits(config) / 8
    else:
        bytes_per_param = _FULL_FINETUNE_BYTES_PER_PARAM

    return params_b * bytes_per_param * _OVERHEAD_FACTOR


@dataclass(frozen=True)
class Hardware:
    """What the job will actually run on.

    Deliberately catalogue-free: the caller supplies the numbers. The CLI reads
    them from the local GPU, the API from its EC2 instance catalogue. Baking an
    AWS instance table into an Axolotl package would tie the open schema layer
    to one deployment's hardware.
    """

    total_vram_gb: float
    gpu_count: int = 1
    label: str = ""

    def __post_init__(self) -> None:
        if self.total_vram_gb <= 0:
            raise ValueError("total_vram_gb must be positive")
        if self.gpu_count < 1:
            raise ValueError("gpu_count must be at least 1")

    @property
    def per_gpu_vram_gb(self) -> float:
        """VRAM on one card — the figure that matters under data parallelism,
        where each GPU holds a full copy of the model."""
        return self.total_vram_gb / self.gpu_count

    @property
    def display_name(self) -> str:
        return self.label or f"{self.total_vram_gb:g} GB"

    def verdict_for(self, required_gb: float) -> str:
        """One of "fits", "tight", or "over" for a given VRAM requirement."""
        if required_gb > self.total_vram_gb * _OVER_THRESHOLD:
            return "over"
        if required_gb > self.total_vram_gb * _TIGHT_THRESHOLD:
            return "tight"
        return "fits"


# ── Parallelism ───────────────────────────────────────────────────────────────


def _verdict(required_gb: float, capacity_gb: float) -> str:
    if required_gb > capacity_gb * _OVER_THRESHOLD:
        return "over"
    if required_gb > capacity_gb * _TIGHT_THRESHOLD:
        return "tight"
    return "fits"


def _user_sharding(config: dict) -> str | None:
    """The sharding the user configured, if any.

    FSDP and ZeRO-3 shard the weights themselves. ZeRO-1/2 shard only optimiser
    state and gradients, which this estimate does not separate out, so those
    are reported as plain "deepspeed" and treated conservatively.
    """
    if config.get("fsdp") or config.get("fsdp_config"):
        return "fsdp"
    deepspeed = config.get("deepspeed")
    if deepspeed:
        if isinstance(deepspeed, str) and "zero3" in deepspeed.lower():
            return "zero3"
        return "deepspeed"
    return None


@dataclass(frozen=True)
class ParallelismPlan:
    """How a job will use the GPUs on one machine.

    `per_gpu_gb` is the estimated peak on each card under `strategy`. `fits`
    is what instance selection reads and `verdict` what the validator reads;
    `fits` is derived from `verdict`, so the two cannot disagree about whether
    a job can run. `auto_sharded` means ZeRO-3 was chosen on the
    user's behalf, and `exact` is False when the user's own sharding config
    could not be modelled.
    """

    strategy: str  # "single" | "ddp" | "zero3" | "fsdp" | "deepspeed"
    per_gpu_gb: float
    fits: bool
    verdict: str
    auto_sharded: bool = False
    exact: bool = True
    reason: str = ""


def plan_parallelism(config: dict, hardware: Hardware) -> ParallelismPlan:
    """Decide how to spread a job over `hardware`, and whether it fits.

    The one function the validator, instance selection and the trainer all
    call, so that what gets checked, what gets rented and what gets launched
    cannot disagree.
    """
    full = vram_required_gb(config)
    gpus = hardware.gpu_count
    capacity = hardware.per_gpu_vram_gb

    def plan(strategy: str, per_gpu: float, **extra) -> ParallelismPlan:
        verdict = _verdict(per_gpu, capacity)
        return ParallelismPlan(
            strategy=strategy,
            per_gpu_gb=per_gpu,
            fits=verdict != "over",
            verdict=verdict,
            **extra,
        )

    if gpus == 1:
        return plan("single", full, reason="one GPU")

    user = _user_sharding(config)
    if user in ("fsdp", "zero3"):
        return plan(user, full / gpus, reason=f"configured {user} across {gpus} GPUs")
    if user == "deepspeed":
        return plan(
            "deepspeed",
            full,
            exact=False,
            reason="ZeRO-1/2 keep weights replicated; fit is estimated as unsharded",
        )

    if full <= capacity:
        return plan("ddp", full, reason=f"fits one GPU; replicated across {gpus}")

    if training_bits(config) < 16:
        # Auto-sharding a quantized model would usually mean QLoRA under
        # ZeRO-3, which is unreliable — better to say it does not fit.
        return plan(
            "ddp",
            full,
            reason="quantized models are not auto-sharded; this needs a GPU "
            "large enough to hold it on its own",
        )

    return plan(
        "zero3",
        full / gpus,
        auto_sharded=True,
        reason=f"too large for one GPU; sharded with ZeRO-3 across {gpus}",
    )
