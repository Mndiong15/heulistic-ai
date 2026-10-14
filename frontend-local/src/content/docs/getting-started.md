---
title: Getting Started
description: Fine-tune your first model on your own GPU, from the terminal or the browser.
order: 1
---

## What is Heulistic?

Heulistic runs fine-tuning jobs on your own hardware. You bring an Axolotl
config and a dataset; it handles queueing, GPU scheduling, checkpointing, live
metrics and model delivery — locally, with no cloud account and nothing to pay.

## Prerequisites

Depends what you want to do. **Installing needs nothing but Python**, and two
commands work with no Docker, no stack and no network:

| | Requires |
|---|---|
| `heulistic init`, `heulistic validate-config` | Python 3.10+ |
| `heulistic start` and training | Docker, plus an NVIDIA GPU and the Container Toolkit |

`heulistic start` checks for Docker, the driver and the Container Toolkit, and
tells you which is missing. Without a GPU it falls back to CPU, which is fine
for checking that a config and dataset work end to end but far too slow for
real training.

A HuggingFace token is only needed if your base model is gated (Llama, Gemma,
…). `heulistic init` picks an ungated model, so the first run needs none.

## From the terminal

```bash
pip install heulistic

heulistic init                        # a starter config sized to your GPU
heulistic validate-config config.yaml # check it before spending GPU time
heulistic start                       # bring the stack up
heulistic train config.yaml           # submit and watch the loss scroll
heulistic download <job-id>           # collect the model
heulistic export <job-id>             # run it with Ollama
```

`heulistic init` inspects your GPU and picks a base model that fits it, so the
first run does not end in an out-of-memory error twenty minutes in.

`heulistic validate-config` works offline — no Docker, no stack, no network. It
checks your config against Axolotl's own schema: unknown fields, wrong types,
missing required settings, and whether the job will fit your card.

## From the browser

`heulistic start` opens the dashboard automatically. Upload a config on
**Train**, optionally attach a JSONL dataset, and submit. **My Jobs** shows
live logs streamed from the training container, loss and GPU charts, and a
download button when the run finishes.

## Bringing your own dataset

Either point `datasets` at a HuggingFace dataset:

```yaml
datasets:
  - path: tatsu-lab/alpaca
    type: alpaca
```

…or pass a local JSONL file, one JSON object per line:

```bash
heulistic train config.yaml --dataset ./my-data.jsonl
```

## Managing the stack

```bash
heulistic status        # what is running, and what your jobs are doing
heulistic logs -f       # follow the stack's own logs
heulistic stop          # shut everything down
```

Jobs run one at a time, because they share your GPU. Submit as many as you
like — they queue.

## Need more GPU than you have?

[Heulistic Cloud](https://heulistic.com) runs the same configs on managed
GPUs, billed by the second.
