"""Export a trained model to run locally: merge → GGUF → Ollama Modelfile.

An export job runs in the training image with JOB_KIND=export. It fetches the
source job's model, merges a LoRA/QLoRA adapter into the base weights
(`axolotl merge-lora`, which works shard by shard and so needs little RAM),
converts and quantizes to GGUF through llama.cpp (`axolotl export`), and
writes a Modelfile whose prompt template matches the one the model was
trained on. A mismatched template still runs — it just answers worse, which
is the failure this module exists to prevent.
"""

import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

import yaml

LLAMA_CPP_DIR = Path("/opt/llama.cpp")

_PEFT = ("lora", "qlora")
# Settings for a training run that a single-process merge must not inherit.
_DROP = (
    "deepspeed", "fsdp", "fsdp_config", "fsdp_version", "resume_from_checkpoint",
    "use_wandb", "wandb_project", "wandb_entity", "wandb_watch", "mlflow_experiment_name",
)


# ── Prompt format ─────────────────────────────────────────────────────────────

_ALPACA_SYSTEM = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request."
)

# Ollama Go templates reproducing each training format. Alpaca matches
# Axolotl's AlpacaPrompter (instruct style, no input) character for character.
_TEMPLATES = {
    "alpaca": (
        'TEMPLATE """{{ if .System }}{{ .System }}\n\n{{ end }}'
        '### Instruction:\n{{ .Prompt }}\n\n### Response:\n"""\n'
        f'SYSTEM """{_ALPACA_SYSTEM}"""\n'
        'PARAMETER stop "### Instruction:"\n'
    ),
    "chatml": (
        'TEMPLATE """{{- range .Messages }}<|im_start|>{{ .Role }}\n'
        '{{ .Content }}<|im_end|>\n{{ end }}<|im_start|>assistant\n"""\n'
        'PARAMETER stop "<|im_end|>"\n'
    ),
    "llama3": (
        'TEMPLATE """{{- range .Messages }}<|start_header_id|>{{ .Role }}<|end_header_id|>\n\n'
        '{{ .Content }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>\n\n"""\n'
        'PARAMETER stop "<|eot_id|>"\n'
    ),
    "raw": 'TEMPLATE """{{ .Prompt }}"""\n',
}
_RAW_TYPES = ("completion", "input_output", "pretrain")
_CHAT_TYPES = ("chat_template", "sharegpt")


def prompt_format(cfg: dict) -> Optional[str]:
    """How the model saw its prompts in training: a key of _TEMPLATES, or None
    when the tokenizer's own chat template was used — which is saved with the
    model, carried into the GGUF, and read from there by Ollama."""
    datasets = cfg.get("datasets") or []
    first = datasets[0] if datasets and isinstance(datasets[0], dict) else {}
    # The trainer gives an uploaded dataset with no type `type: alpaca`.
    ds_type = str(first.get("type") or "alpaca")

    base_type, _, style = ds_type.partition(":")
    if style:
        return "chatml" if style == "chatml" else None
    if base_type in _RAW_TYPES:
        return "raw"
    if base_type in _CHAT_TYPES:
        template = cfg.get("chat_template") or first.get("chat_template")
        return template if template in ("chatml", "llama3") else None
    if base_type == "alpaca":
        return "alpaca"
    return None


def ollama_modelfile(cfg: dict, gguf_file: str) -> str:
    lines = [f"FROM ./{gguf_file}", ""]
    fmt = prompt_format(cfg)
    if fmt:
        lines.append(_TEMPLATES[fmt])
    else:
        lines.append("# Chat template: read from the GGUF — the one the model trained with.\n")
    if cfg.get("sequence_len"):
        lines.append(f"PARAMETER num_ctx {int(cfg['sequence_len'])}\n")
    return "\n".join(lines).rstrip("\n") + "\n"


def gguf_name(base_model: str, job_id: str, quant: str) -> str:
    stem = re.sub(r"[^a-z0-9.-]+", "-", (base_model or "model").split("/")[-1].lower()).strip("-")
    return f"{stem or 'model'}-{job_id[:8]}-{quant}.gguf"


# ── Pipeline ──────────────────────────────────────────────────────────────────


def disk_needed_gb(params_b: float) -> float:
    """Scratch disk an export needs at its peak; shared with the API and CLI."""
    from heulistic_axolotl import export_scratch_gb

    return export_scratch_gb(params_b)


def export_config(cfg: dict, model_dir: Path, work_dir: Path, quant: str) -> dict:
    """The Axolotl config for merging and exporting the trained model."""
    out = {k: v for k, v in cfg.items() if k not in _DROP}
    out["output_dir"] = str(model_dir)
    if str(cfg.get("adapter") or "").lower() in _PEFT:
        out["lora_model_dir"] = str(model_dir)
    out["export"] = {
        "format": "gguf",
        # Keep the model's own precision for the intermediate file; quantizing
        # from it is what matters.
        "outtype": "auto",
        "quantize": [quant],
        "outfile": str(work_dir / "out-{ftype}.gguf"),
        "llama_cpp_dir": str(LLAMA_CPP_DIR),
    }
    return out


def _step(run: Callable, cmd: list[str], name: str) -> None:
    if run(cmd, check=False).returncode != 0:
        raise RuntimeError(f"`axolotl {name}` failed — see the log above for the cause")


def run_export(
    cfg: dict,
    model_dir: Path,
    work_dir: Path,
    quant: str,
    run: Callable = subprocess.run,
) -> Path:
    """Merge (if there is an adapter), convert, quantize; return the GGUF.

    Intermediates are deleted as soon as they are spent: each is as large as
    the model, and the instance disk is sized for the peak, not the sum.
    """
    cfg_path = work_dir / "export.yaml"
    cfg_path.write_text(yaml.safe_dump(export_config(cfg, model_dir, work_dir, quant)))

    has_adapter = (model_dir / "adapter_config.json").exists()
    source = model_dir
    if has_adapter:
        _step(run, ["axolotl", "merge-lora", str(cfg_path)], "merge-lora")
        source = model_dir / "merged"

    _step(run, ["axolotl", "export", str(cfg_path), "--model-dir", str(source)], "export")

    gguf = work_dir / f"out-{quant}.gguf"
    if not gguf.exists():
        raise RuntimeError(f"`axolotl export` finished but wrote no {quant} file")
    for leftover in work_dir.glob("out-*.gguf"):
        if leftover != gguf:
            leftover.unlink()
    if has_adapter:
        shutil.rmtree(source, ignore_errors=True)
    return gguf
