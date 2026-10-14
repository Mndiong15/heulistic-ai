"""Before/after evaluation: the base model against the fine-tuned one.

Runs after training, in its own process so the training run's GPU memory is
gone. Both models are scored on the rows Axolotl held out (its seeded
val_set_size split) — never seen in training — using Axolotl's own dataset and
model loading, so prompts are formatted and the model is quantized exactly as
they were during training.

Usage: python model_eval.py --config CONFIG --output-dir DIR --report OUT.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

_MAX_EXAMPLES = 50
_SAMPLES = 3
_MAX_NEW_TOKENS = 200
_MAX_TEXT_CHARS = 2000
_IGNORE = -100

# Single-process evaluation: no sharding, no resuming, and no trained adapter
# on the base side.
_DROP = (
    "deepspeed", "fsdp", "fsdp_config", "fsdp_version",
    "resume_from_checkpoint", "lora_model_dir",
)
_PEFT = ("lora", "qlora")


def split_prompt(input_ids: list[int], labels: list[int]) -> tuple[list[int], list[int]]:
    """Split a tokenized example into the prompt and the answer it trained on.

    Axolotl masks the prompt with -100 labels, so the prompt is everything
    before the first trained token. Completion data trains on every token;
    there the model is asked to continue from the first quarter.
    """
    first = next((i for i, label in enumerate(labels) if label != _IGNORE), None)
    if not first:  # None (nothing trained) or 0 (everything trained)
        cut = max(1, len(input_ids) // 4)
        return list(input_ids[:cut]), list(input_ids[cut:])
    reference = [t for t, label in zip(input_ids[first:], labels[first:]) if label != _IGNORE]
    return list(input_ids[:first]), reference


def eval_configs(cfg: dict, output_dir: str) -> tuple[dict, dict]:
    """Configs that load the base model and the fine-tuned one.

    For LoRA the base is the config with a fresh adapter: its B matrices start
    at zero, so it computes exactly the base model, under the same
    quantization. A full fine-tune compares the original weights with the
    trained ones in the output directory.
    """
    base = {k: v for k, v in cfg.items() if k not in _DROP}
    tuned = dict(base)
    if str(cfg.get("adapter") or "").lower() in _PEFT:
        tuned["lora_model_dir"] = output_dir
    else:
        tuned["base_model"] = output_dir
    return base, tuned


def summarize(
    base_losses: list[tuple[float, int]],
    tuned_losses: list[tuple[float, int]],
    samples: list[dict],
) -> dict:
    """The report: token-weighted loss and perplexity for each model."""

    def scores(losses):
        tokens = sum(n for _, n in losses)
        loss = sum(value * n for value, n in losses) / tokens if tokens else float("nan")
        return {"loss": loss, "perplexity": math.exp(loss) if loss < 50 else float("inf")}

    return {
        "examples": len(base_losses),
        "base": scores(base_losses),
        "tuned": scores(tuned_losses),
        "samples": samples,
    }


def _clip(text: str) -> str:
    return text if len(text) <= _MAX_TEXT_CHARS else text[:_MAX_TEXT_CHARS] + "…"


def run(config_path: Path, output_dir: Path, report_path: Path) -> dict:
    import torch
    from axolotl.cli.config import load_cfg
    from axolotl.common.datasets import load_datasets
    from axolotl.train import setup_model_and_tokenizer

    raw = yaml.safe_load(config_path.read_text())
    paths = {}
    for name, cfg in zip(("base", "tuned"), eval_configs(raw, str(output_dir))):
        paths[name] = config_path.parent / f"eval_{name}.yaml"
        paths[name].write_text(yaml.safe_dump(cfg))

    eval_dataset = load_datasets(cfg=load_cfg(str(paths["base"]))).eval_dataset
    if eval_dataset is None or len(eval_dataset) == 0:
        report = {"skipped": "No held-out examples to evaluate on."}
        report_path.write_text(json.dumps(report))
        return report
    examples = [eval_dataset[i] for i in range(min(_MAX_EXAMPLES, len(eval_dataset)))]

    losses: dict[str, list[tuple[float, int]]] = {}
    outputs: dict[str, list[str]] = {}
    prompts: list[tuple[list[int], list[int]]] = [
        split_prompt(ex["input_ids"], ex["labels"]) for ex in examples[:_SAMPLES]
    ]
    tokenizer = None
    for name in ("base", "tuned"):
        model, tokenizer, _, _ = setup_model_and_tokenizer(load_cfg(str(paths[name])))
        model.eval()
        device = next(model.parameters()).device
        losses[name], outputs[name] = [], []
        with torch.no_grad():
            for ex in examples:
                ids = torch.tensor([ex["input_ids"]], device=device)
                labels = torch.tensor([ex["labels"]], device=device)
                # Tokens scored: every label after the first, as the loss shifts by one.
                scored = int((labels[0, 1:] != _IGNORE).sum())
                if scored == 0:
                    continue
                loss = model(input_ids=ids, labels=labels).loss.item()
                losses[name].append((loss, scored))
            for prompt, _ in prompts:
                ids = torch.tensor([prompt], device=device)
                generated = model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    max_new_tokens=_MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
                outputs[name].append(
                    tokenizer.decode(generated[0][len(prompt):], skip_special_tokens=True)
                )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    samples = [
        {
            "prompt": _clip(tokenizer.decode(prompt, skip_special_tokens=True)),
            "reference": _clip(tokenizer.decode(reference, skip_special_tokens=True)),
            "base": _clip(outputs["base"][i]),
            "tuned": _clip(outputs["tuned"][i]),
        }
        for i, (prompt, reference) in enumerate(prompts)
    ]
    report = summarize(losses["base"], losses["tuned"], samples)
    report_path.write_text(json.dumps(report))
    return report


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.config, args.output_dir, args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
