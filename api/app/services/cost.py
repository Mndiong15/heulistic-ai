import math
import re
from decimal import Decimal

import yaml

from heulistic_axolotl import (
    Hardware,
    export_scratch_gb,
    parse_param_billions,
    plan_parallelism,
    training_bits,
    vram_required_gb,
)

from .instances import get_catalog, get_rate, valid_instance_types

MARKUP = Decimal("2.2")
BUFFER_FACTOR = Decimal("1.5")  # 50% buffer held at submission, refunded on settlement


# The VRAM model lives in heulistic-axolotl, shared with the validator and the
# trainer. These names are kept for existing callers and tests; they used to be
# a private copy of the same formula, which is how selection and validation
# came to disagree.
_parse_param_billions = parse_param_billions
_training_bits = training_bits
_vram_required_gb = vram_required_gb


def hardware_for(instance_type: str) -> Hardware:
    """A catalogue instance as a validation and planning target."""
    return _hardware(instance_type, get_catalog()[instance_type])


def _hardware(instance_type: str, info: dict) -> Hardware:
    return Hardware(
        total_vram_gb=info["vram_gb"],
        gpu_count=info.get("gpus", 1),
        label=instance_type,
    )


def select_instance(config: dict) -> str:
    """Cheapest instance this job can actually run on.

    Fit is judged per GPU, using the same plan the trainer executes: a job that
    does not fit one card is placed only where ZeRO-3 sharding makes it fit.
    This used to compare against total VRAM, which under DDP — one full model
    copy per card — routed quantized jobs to multi-GPU boxes where they OOMed
    on every card.
    """
    catalog = get_catalog()
    plans = [
        (itype, info, plan_parallelism(config, _hardware(itype, info)))
        for itype, info in sorted(catalog.items(), key=lambda kv: kv[1]["on_demand_rate"])
    ]
    for itype, _, plan in plans:
        if plan.fits:
            return itype

    # Nothing fits. Take the box the job comes closest to fitting, cheapest
    # among equals. Simply taking the biggest box meant renting four GPUs for a
    # quantized model that cannot be sharded across them.
    def shortfall(entry):
        itype, info, plan = entry
        return (plan.per_gpu_gb / _hardware(itype, info).per_gpu_vram_gb, info["on_demand_rate"])

    return min(plans, key=shortfall)[0]


def _param_tier(base_model: str) -> str:
    """Return model size tier for TPS calibration. Checks largest first to avoid false matches."""
    name = base_model.lower()
    if re.search(r'(?<!\d)(65|70|72)b(?!\d)', name):
        return "xlarge"
    if re.search(r'(?<!\d)(20|30|34|40)b(?!\d)', name):
        return "large"
    if re.search(r'(?<!\d)(11|12|13|14)b(?!\d)', name):
        return "medium"
    if re.search(r'(?<!\d)(7|8|9)b(?!\d)', name):
        return "small"
    if re.search(r'(?<!\d)(0\.5|1|1\.1|1\.5|2|3|3\.8|4)b(?!\d)', name):
        return "tiny"
    return "small"  # default: assume 7B-class


def _estimate_tps(base_model: str, is_peft: bool) -> int:
    """Tokens per second per GPU on A10G baseline, scaled by model parameter count."""
    tier = _param_tier(base_model)
    if is_peft:
        return {"tiny": 20_000, "small": 2_000, "medium": 800, "large": 200, "xlarge": 80}[tier]
    else:
        return {"tiny": 8_000, "small": 700, "medium": 280, "large": 70, "xlarge": 30}[tier]


# Fraction of linear speed-up each extra GPU delivers, by parallelism strategy.
# Every multi-GPU instance offered (g4dn/g5/g6/g6e .12xlarge) links its GPUs
# over PCIe, not NVLink. DDP all-reduces gradients once per step, which is cheap
# for LoRA; ZeRO-1/2 add optimiser-state traffic; ZeRO-3 and FSDP gather the
# weights in every forward and backward pass. These are deliberately cautious
# guesses, not measurements — the trainer now reports tokens/sec summed over
# GPUs, so they can be calibrated from real jobs. Erring slow only raises the
# credit hold; settlement bills actual time either way.
_SCALING_EFFICIENCY = {
    "single": 1.0,
    "ddp": 0.9,
    "deepspeed": 0.8,
    "zero3": 0.6,
    "fsdp": 0.6,
}


def _scaling_efficiency(strategy: str) -> float:
    return _SCALING_EFFICIENCY.get(strategy, _SCALING_EFFICIENCY["zero3"])


def _estimate_num_samples(dataset_size_mb: int, sequence_len: int) -> int:
    """50K tokens/MB divided by sequence_len gives approximate sample count."""
    return max(1, (max(dataset_size_mb, 1) * 50_000) // sequence_len)


def _framework_tax_seconds(config: dict, num_raw_samples: int, total_steps: int) -> float:
    tax = 0.0
    val_set_size = float(config.get("val_set_size") or 0.0)
    num_epochs = float(config.get("num_epochs") or 1.0)

    if val_set_size > 0.0:
        val_samples = math.ceil(num_raw_samples * val_set_size)
        if config.get("eval_steps"):
            eval_runs = total_steps // int(config["eval_steps"])
        elif config.get("evals_per_epoch"):
            eval_runs = int(config["evals_per_epoch"]) * math.ceil(num_epochs)
        else:
            eval_runs = math.ceil(num_epochs)
        tax += eval_runs * val_samples * 0.005  # 5 ms/sample inference

    adapter = (config.get("adapter") or "").lower()
    disk_write_secs = 10 if adapter in ("lora", "qlora") else 60
    if config.get("save_steps"):
        save_runs = total_steps // int(config["save_steps"])
    elif config.get("saves_per_epoch"):
        save_runs = int(config["saves_per_epoch"]) * math.ceil(num_epochs)
    else:
        save_runs = math.ceil(num_epochs)
    tax += save_runs * disk_write_secs

    return tax


def estimate_duration_minutes(
    config: dict,
    dataset_size_mb: int,
    instance_type: str = "g5.xlarge",
    *,
    num_train_samples: int | None = None,
) -> int:
    sequence_len = int(config.get("sequence_len") or 512)
    num_epochs   = float(config.get("num_epochs") or 1.0)
    micro_batch  = max(1, int(config.get("micro_batch_size") or 1))
    grad_accum   = max(1, int(config.get("gradient_accumulation_steps") or 1))
    val_set_size = float(config.get("val_set_size") or 0.0)
    adapter      = (config.get("adapter") or "").lower()
    base_model   = config.get("base_model") or ""
    is_peft      = adapter in ("lora", "qlora")
    info         = get_catalog().get(instance_type)
    num_gpus     = max(1, info.get("gpus", 1)) if info else 1
    strategy     = (
        plan_parallelism(config, _hardware(instance_type, info)).strategy
        if num_gpus > 1 else "single"
    )

    # Phase 1: total training sequences
    # num_train_samples overrides MB-based estimation when we have exact row counts (e.g. from HF API)
    if num_train_samples is not None:
        num_raw_samples = num_train_samples
    else:
        num_raw_samples = _estimate_num_samples(dataset_size_mb, sequence_len)
    train_samples   = math.ceil(num_raw_samples * (1.0 - val_set_size))
    total_sequences = train_samples

    # Phase 2: total GPU forward/backward passes
    global_batch_size = micro_batch * grad_accum * num_gpus
    steps_per_epoch   = math.ceil(total_sequences / global_batch_size)
    total_steps       = int(steps_per_epoch * num_epochs)
    max_steps         = config.get("max_steps")
    if max_steps is not None and int(max_steps) < total_steps:
        total_steps = int(max_steps)
    total_gpu_iters = total_steps * grad_accum

    # TPS scaled by model size; num_gpus already reduces total_steps via
    # global_batch_size, which assumes perfect scaling — the efficiency undoes
    # that optimism for the communication each strategy adds.
    tps_per_gpu      = _estimate_tps(base_model, is_peft) * _scaling_efficiency(strategy)
    training_seconds = total_gpu_iters * (micro_batch * sequence_len) / tps_per_gpu

    # Phase 3: eval + checkpoint overhead
    tax_seconds = _framework_tax_seconds(config, num_raw_samples, total_steps)

    # 5-min cold start: instance boot + EFS mount + model pull
    total_seconds = training_seconds + tax_seconds + 300
    minutes = max(10, int(total_seconds / 60))
    if will_evaluate(config):
        minutes += evaluation_minutes(config)
    return minutes


# ── Before/after comparison ───────────────────────────────────────────────────


def will_evaluate(config: dict) -> bool:
    """Whether training will end with the base-vs-fine-tuned comparison.

    It runs when rows are held out: the user's own val_set_size or
    test_datasets, or — when val_set_size is not set — the trainer's automatic
    hold-out, which needs 100+ rows. The size is not always known here, so an
    unset val_set_size counts as yes: the estimate errs long, never short.
    """
    if config.get("rl"):
        return False
    if config.get("test_datasets"):
        return True
    return "val_set_size" not in config or bool(config.get("val_set_size"))


def evaluation_minutes(config: dict) -> int:
    """Loading the model twice, scoring up to 50 rows, three short answers each."""
    return math.ceil(2 + 0.3 * parse_param_billions(config.get("base_model") or ""))


# ── Export ────────────────────────────────────────────────────────────────────
# Merging and quantizing barely use the GPU, so exports run on the cheapest
# instance that still uses the training image and AMI. More vCPUs for the big
# models, where quantizing on four cores would take hours.

_EXPORT_SMALL = "g4dn.xlarge"
_EXPORT_LARGE = "g4dn.4xlarge"
_LARGE_MODEL_B = 30
# Root volume floor: the provisioner's default, which holds the image.
_MIN_DISK_GB = 200
_IMAGE_AND_OS_GB = 60


def export_instance(config: dict) -> str:
    params_b = parse_param_billions(config.get("base_model") or "")
    return _EXPORT_LARGE if params_b > _LARGE_MODEL_B else _EXPORT_SMALL


def export_minutes(config: dict) -> int:
    """Boot and image pull, then download, merge, convert and quantize — all
    roughly linear in model size."""
    return math.ceil(5 + 1.2 * parse_param_billions(config.get("base_model") or ""))


def export_disk_gb(config: dict) -> int:
    params_b = parse_param_billions(config.get("base_model") or "")
    return max(_MIN_DISK_GB, math.ceil(_IMAGE_AND_OS_GB + export_scratch_gb(params_b)))


def export_estimate(config: dict) -> dict:
    instance_type = export_instance(config)
    minutes = export_minutes(config)
    return {
        "instance_type": instance_type,
        "estimated_minutes": minutes,
        "estimated_cost": estimate_cost(instance_type, minutes),
        "disk_gb": export_disk_gb(config),
    }


def estimate_cost(instance_type: str, duration_minutes: int) -> Decimal:
    rate = get_rate(instance_type)
    raw = Decimal(str(round(rate * duration_minutes / 60, 2)))
    return (raw * MARKUP).quantize(Decimal("0.01"))


def parse_and_estimate(
    config_yaml: str,
    dataset_size_mb: int = 0,
    instance_type: str | None = None,
    *,
    num_train_samples: int | None = None,
) -> dict:
    config = yaml.safe_load(config_yaml)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")

    if instance_type and instance_type in valid_instance_types():
        selected = instance_type
    else:
        selected = select_instance(config)

    duration = estimate_duration_minutes(
        config, dataset_size_mb, instance_type=selected, num_train_samples=num_train_samples
    )
    cost = estimate_cost(selected, duration)
    hourly_rate = (Decimal(str(get_rate(selected))) * MARKUP).quantize(Decimal("0.001"))

    return {
        "instance_type": selected,
        "estimated_duration_minutes": duration,
        "estimated_cost": cost,
        "hourly_rate": hourly_rate,
    }
