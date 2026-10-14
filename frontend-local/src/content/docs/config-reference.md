---
title: Config Reference
description: Axolotl YAML configuration fields supported by Heulistic.
order: 3
---

## Overview

Heulistic uses [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) as its training framework. Your config is a standard Axolotl YAML file. This page covers the fields most relevant to Heulistic jobs.

## Required Fields

| Field | Description |
|---|---|
| `base_model` | HuggingFace model ID (e.g. `meta-llama/Llama-3.2-3B`) |
| `learning_rate` | Initial learning rate (e.g. `2e-4`) |

## Common Fields

| Field | Default | Description |
|---|---|---|
| `num_epochs` | — | Number of training epochs |
| `micro_batch_size` | — | Batch size per GPU |
| `gradient_accumulation_steps` | `1` | Steps to accumulate before optimizer update |
| `sequence_len` | `2048` | Max token sequence length |
| `output_dir` | auto-set | Set automatically by Heulistic — do not override |
| `save_steps` | `100` | Checkpoint every N steps (Heulistic uploads to S3) |

## LoRA / QLoRA

For parameter-efficient fine-tuning:

```yaml
adapter: lora          # or qlora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - v_proj

# QLoRA only — quantize the base model to 4-bit
load_in_4bit: true
```

Use `adapter: qlora` + `load_in_4bit: true` to fit large models (13B+) on a single GPU with less memory.

## Datasets

### Uploading a file

Upload a JSONL file on the Train page. Heulistic sets the path automatically. In your config, specify the format:

```yaml
datasets:
  - type: alpaca    # or sharegpt, completion, input_output
```

Supported dataset types and their required fields:

| Type | Required keys |
|---|---|
| `alpaca` | `instruction`, `output` |
| `input_output` | `input`, `output` |
| `completion` | `text` |
| `sharegpt` | `conversations` |

### Using a HuggingFace dataset

```yaml
datasets:
  - path: tatsu-lab/alpaca
    type: alpaca
```

No file upload needed. For gated or private datasets, provide your HuggingFace token.

## Full Fine-Tune Example

```yaml
base_model: mistralai/Mistral-7B-v0.1
model_type: MistralForCausalLM
tokenizer_type: LlamaTokenizer

datasets:
  - type: alpaca

sequence_len: 4096
micro_batch_size: 2
gradient_accumulation_steps: 4
num_epochs: 3
learning_rate: 2e-5
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
```

## LoRA Example (Llama 3)

```yaml
base_model: meta-llama/Llama-3.2-3B-Instruct

datasets:
  - type: alpaca

adapter: lora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05

sequence_len: 2048
micro_batch_size: 4
num_epochs: 3
learning_rate: 2e-4
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
```

## Unsupported Fields

Heulistic runs single-node jobs. The following fields are rejected at submission:

- `world_size` > 1
- `tensor_parallel_size` > 1
- `use_ray: true`
- `ray_num_workers` > 1

WandB and MLflow logging are disabled automatically — metrics are streamed to the Heulistic dashboard instead.
