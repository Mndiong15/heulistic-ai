"""Validate an Axolotl config.

Three sources of rules, in order of confidence:

1. **The generated schema** — unknown fields, wrong types, missing required
   fields. Derived from Axolotl's own reference, so it cannot drift from
   upstream by hand.
2. **Coherence and sanity rules** — the worthwhile half of the frontend's
   `validate-config.ts`, now schema-backed instead of hand-listed.
3. **Heulistic platform limits** — single-node only, and an optional check
   that the job fits the hardware it will run on.

Severity follows the contract the frontend already renders: `error` should
block submission, `warning` is probably wrong, `info` is a suggestion.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

import yaml

from .schema import SCHEMA, get_field, get_model, known_field_names
from .vram import Hardware, plan_parallelism, vram_required_gb

__all__ = ["Diagnostic", "validate"]

_LEVEL_ORDER = {"error": 0, "warning": 1, "info": 2}

# Old or wrong names people actually write, and what they meant.
_ALIASES: dict[str, str] = {
    "dataset_format": "datasets[].type",
    "dataset_type": "datasets[].type",
    "lora_modules": "lora_target_modules",
    "epochs": "num_epochs",
    "batch": "micro_batch_size",
    "lr": "learning_rate",
    "model": "base_model",
    "tokenizer": "tokenizer_type",
}

# Prompt formats Axolotl ships. `datasets[].type` is typed `str |
# UserDefinedPrompterType`, so the schema cannot supply this list.
_KNOWN_DATASET_TYPES = frozenset({
    "alpaca", "sharegpt", "chat_template", "completion", "oasst", "gpteacher",
    "input_output", "json", "custom", "llama3_chat", "mistral_v1", "chatml",
    "jinja_template", "reflection", "pygmalion",
})

_LORA_FIELDS = (
    "lora_r", "lora_alpha", "lora_dropout", "lora_target_linear",
    "lora_target_modules", "peft_use_dora", "peft_use_rslora",
)

_PLACEHOLDER = "YOUR_MODEL_HERE"
_MAX_SEQUENCE_LEN = 131072
_LARGE_SEQUENCE_LEN = 8192
_MAX_SANE_EPOCHS = 100
_MAX_SANE_LR = 0.01
_LARGE_LORA_R = 256


@dataclass(frozen=True)
class Diagnostic:
    """One finding.

    `code` is a stable identifier for the rule that fired — `vram.over`,
    `field.unknown`, and so on. Callers should branch and filter on it rather
    than matching message text, which is free to change.

    `field` and `line` are absent when a finding is not attributable to one
    setting.
    """

    level: str
    message: str
    field: str | None = None
    line: int | None = None
    code: str = ""


def _line_map(config_yaml: str) -> dict[str, int]:
    """Map top-level keys to 1-based line numbers, for pointing at the source."""
    try:
        node = yaml.compose(config_yaml)
    except yaml.YAMLError:
        return {}
    if not isinstance(node, yaml.MappingNode):
        return {}
    return {
        str(key.value): key.start_mark.line + 1
        for key, _ in node.value
        if isinstance(key, yaml.ScalarNode)
    }


def _as_number(value) -> float | None:
    """Coerce to float, tolerating the quoted scientific form Axolotl allows."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _dataset_entry_fields() -> frozenset[str]:
    """Every key valid on a `datasets[]` entry.

    Taken from the union of models the `datasets` field references, so DPO and
    KTO entries are not flagged for lacking SFT fields.
    """
    names: set[str] = set()
    datasets = get_field("datasets")
    for model_name in datasets.type.named if datasets else ():
        model = get_model(model_name)
        if model:
            names.update(model.fields)
    return frozenset(names)


def validate(config_yaml: str, target: Hardware | None = None) -> list[Diagnostic]:
    """Validate an Axolotl config.

    `target` enables the hardware fit check — the local GPU for the CLI, the
    selected instance for the cloud. Without one, only the config itself is
    judged.
    """
    out: list[Diagnostic] = []

    try:
        parsed = yaml.safe_load(config_yaml)
    except yaml.YAMLError as exc:
        return [Diagnostic("error", f"YAML syntax error: {exc}", code="yaml.syntax")]

    if parsed is None:
        return [Diagnostic("error", "Config is empty", code="config.empty")]
    if not isinstance(parsed, dict):
        return [Diagnostic("error", "YAML did not parse to a mapping of settings",
                           code="config.not_mapping")]

    lines = _line_map(config_yaml)

    def add(level: str, message: str, field: str | None = None, code: str = "") -> None:
        out.append(Diagnostic(level, message, field, lines.get(field or ""), code))

    missing_required = _check_required(parsed, add)
    _check_known_and_typed(parsed, missing_required, add)
    _check_base_model(parsed, add)
    _check_adapter(parsed, add)
    _check_datasets(parsed, add)
    _check_numbers(parsed, add)
    _check_hints(parsed, add)
    _check_platform(parsed, add)
    if target is not None:
        _check_hardware(parsed, target, add)

    # Severity first, then source order. Findings with no line — a missing
    # required field, a dataset entry — sort to the end of their group rather
    # than above line 1, which reads as though they came first in the file.
    out.sort(
        key=lambda d: (_LEVEL_ORDER[d.level], 0 if d.line else 1, d.line or 0, d.field or "")
    )
    return out


# ── Rule groups ───────────────────────────────────────────────────────────────

def _check_required(config: dict, add) -> set[str]:
    missing = set()
    for name, field in SCHEMA.fields.items():
        if not field.required:
            continue
        if name not in config or config[name] is None or config[name] == "":
            add("error", f"'{name}' is required", name, code="field.required")
            missing.add(name)
    return missing


def _check_known_and_typed(config: dict, missing_required: set[str], add) -> None:
    known = known_field_names()
    for key, value in config.items():
        if key in _ALIASES:
            add("warning", f"Unknown field '{key}' — did you mean '{_ALIASES[key]}'?", key, code="field.alias")
            continue
        if key not in known:
            close = difflib.get_close_matches(key, known, n=1, cutoff=0.8)
            hint = f" — did you mean '{close[0]}'?" if close else " — check spelling or the Axolotl docs"
            add("warning", f"Unrecognised field '{key}'{hint}", key, code="field.unknown")
            continue
        if key in missing_required:
            continue  # already reported as missing; a type error would echo it
        field = SCHEMA.fields[key]
        if not field.type.accepts(value):
            add(
                "error",
                f"'{key}' expects {field.raw_type}, got {type(value).__name__}",
                key,
            code="type.mismatch",
        )


def _check_base_model(config: dict, add) -> None:
    raw = config.get("base_model")
    if not isinstance(raw, str) or not raw.strip():
        return
    model = raw.strip()
    if _PLACEHOLDER in model:
        add(
            "error",
            f"Replace '{_PLACEHOLDER}' with a real model ID "
            "(e.g. meta-llama/Llama-3.2-1B)",
            "base_model",
            code="model.placeholder",
        )
        return
    if any(c.isspace() for c in model):
        add(
            "error",
            f"base_model '{model}' contains whitespace — use the HuggingFace "
            "repo ID format (e.g. meta-llama/Llama-3.2-1B)",
            "base_model",
            code="model.whitespace",
        )
    elif "/" not in model:
        add(
            "warning",
            f"base_model '{model}' has no '/' — most models use the 'org/name' "
            "format (e.g. mistralai/Mistral-7B-v0.1)",
            "base_model",
            code="model.no_slash",
        )


def _check_adapter(config: dict, add) -> None:
    adapter = str(config.get("adapter") or "").lower()

    if adapter == "qlora" and not config.get("load_in_4bit"):
        add("warning", "QLoRA normally requires 'load_in_4bit: true'", "load_in_4bit", code="adapter.qlora_4bit")

    if not adapter and any(f in config for f in _LORA_FIELDS):
        add(
            "warning",
            "LoRA settings are present but 'adapter' is not set — "
            "add 'adapter: lora' or 'adapter: qlora'",
            "adapter",
            code="adapter.missing",
        )

    lora_r = config.get("lora_r")
    if isinstance(lora_r, int) and not isinstance(lora_r, bool) and lora_r > _LARGE_LORA_R:
        add(
            "warning",
            f"lora_r {lora_r} is very high — this markedly increases VRAM use "
            "and adapter size",
            "lora_r",
            code="lora.r_large",
        )


def _check_datasets(config: dict, add) -> None:
    datasets = config.get("datasets")
    has_any = bool(datasets) or bool(config.get("pretraining_dataset"))
    if not has_any:
        add(
            "warning",
            "No 'datasets' specified — add at least one dataset entry",
            "datasets",
            code="dataset.missing",
        )
        return
    if not isinstance(datasets, list):
        return  # a type error was already raised

    valid_keys = _dataset_entry_fields()
    for index, entry in enumerate(datasets):
        label = f"datasets[{index}]"
        if not isinstance(entry, dict):
            add("error", f"{label}: each dataset entry must be a mapping", label, code="dataset.entry_not_mapping")
            continue

        if not entry.get("path") and not entry.get("name"):
            add("error", f"{label}: missing required 'path'", label, code="dataset.no_path")

        entry_type = entry.get("type")
        if entry_type is None:
            add(
                "warning",
                f"{label}: 'type' is not set — Axolotl may assume the wrong "
                "prompt format",
                label,
            code="dataset.no_type",
        )
        elif isinstance(entry_type, str) and entry_type not in _KNOWN_DATASET_TYPES:
            add(
                "warning",
                f"{label}: unrecognised dataset type '{entry_type}' — "
                "check the Axolotl docs",
                label,
            code="dataset.unknown_type",
        )

        for key in entry:
            if key not in valid_keys:
                add(
                    "warning",
                    f"{label}: unrecognised key '{key}' for a dataset entry",
                    f"{label}.{key}",
            code="dataset.unknown_key",
        )


def _check_numbers(config: dict, add) -> None:
    if "num_epochs" not in config and "max_steps" not in config:
        add(
            "warning",
            "Neither 'num_epochs' nor 'max_steps' is set — Axolotl defaults to "
            "1 epoch",
            "num_epochs",
            code="epochs.missing",
        )
    else:
        epochs = _as_number(config.get("num_epochs"))
        if epochs is not None:
            if epochs <= 0:
                add("error", "num_epochs must be greater than 0", "num_epochs", code="epochs.invalid")
            elif epochs > _MAX_SANE_EPOCHS:
                add(
                    "warning",
                    f"num_epochs {epochs:g} is very high — confirm this is "
                    "intentional",
                    "num_epochs",
            code="epochs.high",
        )

    if "micro_batch_size" not in config and "batch_size" not in config:
        add(
            "warning",
            "'micro_batch_size' is not set — Axolotl defaults to 1",
            "micro_batch_size",
            code="batch.missing",
        )
    else:
        micro = _as_number(config.get("micro_batch_size"))
        if micro is not None and micro <= 0:
            add(
                "error",
                "micro_batch_size must be a positive integer",
                "micro_batch_size",
            code="batch.invalid",
        )

    grad_accum = _as_number(config.get("gradient_accumulation_steps"))
    if grad_accum is not None and grad_accum <= 0:
        add(
            "error",
            "gradient_accumulation_steps must be a positive integer",
            "gradient_accumulation_steps",
            code="grad_accum.invalid",
        )

    if "sequence_len" not in config:
        add(
            "warning",
            "'sequence_len' is not set — Axolotl defaults to 512; set it "
            "explicitly to avoid surprises",
            "sequence_len",
            code="seqlen.missing",
        )
    else:
        seq_len = _as_number(config.get("sequence_len"))
        if seq_len is not None:
            if seq_len <= 0:
                add("error", "sequence_len must be a positive integer", "sequence_len", code="seqlen.invalid")
            elif seq_len > _MAX_SEQUENCE_LEN:
                add(
                    "error",
                    f"sequence_len {seq_len:g} exceeds what most models support "
                    f"({_MAX_SEQUENCE_LEN})",
                    "sequence_len",
            code="seqlen.too_large",
        )
            elif seq_len > _LARGE_SEQUENCE_LEN:
                add(
                    "warning",
                    f"sequence_len {seq_len:g} is very large — check the model "
                    "and hardware can carry it",
                    "sequence_len",
            code="seqlen.large",
        )

    learning_rate = _as_number(config.get("learning_rate"))
    if learning_rate is not None:
        if learning_rate <= 0:
            add("error", "learning_rate must be positive", "learning_rate", code="lr.invalid")
        elif learning_rate > _MAX_SANE_LR:
            add(
                "warning",
                f"learning_rate {learning_rate:g} looks very high — the usual "
                "range is 1e-5 to 5e-4",
                "learning_rate",
            code="lr.high",
        )


def _check_hints(config: dict, add) -> None:
    if "save_steps" not in config and "save_strategy" not in config:
        add(
            "info",
            "'save_steps' is not set — Axolotl saves a checkpoint each epoch by "
            "default",
            "save_steps",
            code="save.unset",
        )
    elif _as_number(config.get("save_steps")) == 0:
        add(
            "warning",
            "save_steps 0 disables checkpointing — a failed run could not be "
            "resumed",
            "save_steps",
            code="save.disabled",
        )

    if "bf16" not in config:
        add(
            "info",
            "'bf16' is not set — it defaults to 'auto', which suits modern GPUs",
            "bf16",
            code="bf16.unset",
        )
    if "gradient_checkpointing" not in config:
        add(
            "info",
            "'gradient_checkpointing' is not set — enabling it trades speed for "
            "a large VRAM saving",
            "gradient_checkpointing",
            code="gc.unset",
        )


def _check_platform(config: dict, add) -> None:
    """Heulistic runs single-node, on one machine with up to 8 GPUs.

    Only genuine multi-*node* settings are rejected. `world_size`,
    `tensor_parallel_size` and friends are valid on a single multi-GPU box and
    must stay allowed — rejecting them would break legitimate 4-GPU configs.
    """
    if config.get("use_ray") is True:
        add(
            "error",
            "'use_ray' needs a multi-node Ray cluster, which is not supported — "
            "training runs on a single node (up to 8 GPUs)",
            "use_ray",
            code="platform.ray",
        )
    workers = _as_number(config.get("ray_num_workers"))
    if workers is not None and workers > 1:
        add(
            "error",
            "'ray_num_workers' above 1 needs a multi-node Ray cluster, which is "
            "not supported",
            "ray_num_workers",
            code="platform.ray_workers",
        )


def _check_hardware(config: dict, target: Hardware, add) -> None:
    """Judge fit per GPU, using the same plan the trainer will execute.

    This used to compare the requirement against total VRAM, which is wrong
    under data parallelism: each GPU holds a full copy of the model, so 4x22 GB
    is not 88 GB of room. It also suggested ZeRO-2 for large models, which
    shards optimiser state but not weights and so cannot rescue a model that
    does not fit one card.
    """
    plan = plan_parallelism(config, target)
    gpus = target.gpu_count
    capacity = target.per_gpu_vram_gb
    per_gpu = plan.per_gpu_gb
    name = target.display_name

    if plan.verdict == "over":
        if not plan.exact:
            add(
                "warning",
                f"Your DeepSpeed config keeps weights replicated (ZeRO-1/2), so this "
                f"cannot be checked precisely: roughly {per_gpu:.0f} GB per GPU "
                f"unsharded, against {capacity:g} GB on each of {gpus} GPUs. It may "
                "still fit. A ZeRO-3 config would shard the weights too.",
                code="vram.unverified",
            )
        elif gpus == 1:
            add(
                "error",
                f"Needs roughly {per_gpu:.0f} GB of VRAM but {name} has "
                f"{capacity:g} GB. Use a smaller model, switch to 'adapter: qlora' "
                "with 'load_in_4bit: true', or pick larger hardware.",
                code="vram.over",
            )
        else:
            if plan.strategy in ("zero3", "fsdp"):
                tail = (
                    f"That is already sharded across all {gpus}; it needs more or "
                    "larger GPUs."
                )
            else:
                tail = (
                    "Quantized models are not sharded automatically, so this needs "
                    "one GPU large enough to hold it on its own."
                )
            add(
                "error",
                f"Needs roughly {per_gpu:.0f} GB of VRAM per GPU but each of the {gpus} "
                f"GPUs on {name} has {capacity:g} GB. {tail}",
                code="vram.over",
            )
    elif plan.verdict == "tight":
        per = " per GPU" if gpus > 1 else ""
        add(
            "warning",
            f"VRAM will be tight — roughly {per_gpu:.0f} GB{per} needed of "
            f"{capacity:g} GB on {name}. Set 'gradient_checkpointing: true' to "
            "reduce usage.",
            code="vram.tight",
        )

    if plan.auto_sharded:
        add(
            "info",
            f"Too large for one GPU (roughly {vram_required_gb(config):.0f} GB). It "
            f"will be sharded with DeepSpeed ZeRO-3 across all {gpus} GPUs — about "
            f"{per_gpu:.0f} GB each.",
            code="hardware.auto_sharded",
        )
    elif plan.strategy == "ddp" and gpus > 1 and plan.fits:
        add(
            "info",
            f"Training will use DDP across {gpus} GPUs — each holds a full copy of "
            f"the model, about {per_gpu:.0f} GB of the {capacity:g} GB on each card.",
            code="hardware.multi_gpu",
        )
