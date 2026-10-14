---
title: CLI Reference
description: Every heulistic command and option, for running jobs without the browser.
order: 2
---

## Overview

The whole workflow runs from the terminal — the dashboard is optional.

```bash
heulistic init                        # a starter config sized to your GPU
heulistic validate-config config.yaml # check it before spending GPU time
heulistic start                       # bring the stack up
heulistic train config.yaml           # submit and follow the logs
heulistic download <job-id>           # collect the model
heulistic export <job-id>             # GGUF + Ollama Modelfile, ready to run
```

Commands that talk to the API (`train`, `download`, `export`, `status`) need the stack
running. `init` and `validate-config` work offline — no Docker, no network.

## heulistic init

Write a starter Axolotl config, sized to the GPU it finds.

| Option | |
|---|---|
| `--output`, `-o PATH` | Where to write it. Default `config.yaml` |
| `--model`, `-m TEXT` | Use this base model instead of the auto-picked one |
| `--force`, `-f` | Overwrite an existing file |

It reads your card's VRAM and picks the largest model from a modest ladder
that fits, so a first run does not end in an out-of-memory error. Models are
ungated, so no HuggingFace token is needed. The ladder stops at 8B
deliberately: a starter run should finish quickly. For something bigger, pass
`--model`.

With no GPU it writes a small config and says so. The generated file passes
`validate-config` with no findings.

## heulistic validate-config

Check a config against Axolotl's schema. Offline, and the single most useful
command here — it catches in milliseconds what would otherwise fail twenty
minutes into a run.

| Option | |
|---|---|
| `--gpu` | Also check the config fits the local GPU |
| `--vram FLOAT` | Check against this much VRAM (GB, total across all GPUs) instead of detecting |
| `--gpus INTEGER` | GPU count to pair with `--vram`. Default 1 |
| `--strict` | Treat warnings as failures |
| `--quiet` | Errors only |
| `--json` | Machine-readable output |

```bash
heulistic validate-config config.yaml
heulistic validate-config config.yaml --gpu
heulistic validate-config config.yaml --vram 24
heulistic validate-config config.yaml --vram 88 --gpus 4
heulistic validate-config config.yaml --strict   # for CI
```

Findings come at three levels. **error** blocks submission, **warning** is
probably wrong, **info** is a suggestion:

```
Checking config.yaml

  error   line 3   num_epochs        num_epochs must be greater than 0
  error   line 5   micro_batch_size  'micro_batch_size' expects int | None, got str
  warning line 2   lr                Unknown field 'lr' — did you mean 'learning_rate'?
  info             bf16              'bf16' is not set — it defaults to 'auto'

  2 errors, 1 warning, 1 suggestion
```

Field names, types, defaults and required-ness are generated from Axolotl's own
[config reference](https://docs.axolotl.ai/docs/config-reference.html), so the
checks follow upstream rather than a hand-written list. The same rules run at
submission, so anything that passes here passes there.

`--json` emits `{ok, counts, diagnostics}`, where each diagnostic carries a
stable `code` (`epochs.invalid`, `vram.over`, `field.unknown`) — branch on
those rather than on the message text.

## heulistic train

Submit a job and follow it.

| Option | |
|---|---|
| `--dataset FILE` | JSONL dataset to upload. Omit if the config names a HuggingFace dataset |
| `--hf-token TEXT` | Token for gated base models. Also read from `HF_TOKEN` |
| `--detach`, `-d` | Submit and return immediately |
| `--no-validate` | Skip the pre-flight check |

```bash
heulistic train config.yaml
heulistic train config.yaml --dataset ./data.jsonl
heulistic train config.yaml --detach
```

The config is validated locally first, so a typo costs nothing. Logs stream
until the run ends:

```
Submitted 99783c6e-e7d8-4159-9815-287e8ddb0c04

  [axolotl] step 10  loss 1.842
  ...

Job completed.
  get the model: heulistic download 99783c6e-e7d8-4159-9815-287e8ddb0c04
```

**Ctrl-C detaches from the logs without stopping the job** — it keeps running,
and `heulistic status` will show it.

## heulistic download

Fetch the trained model for a completed job.

| Option | |
|---|---|
| `--output`, `-o PATH` | File or directory to write to |
| `--force`, `-f` | Overwrite an existing file |
| `--url-only` | Print the URL instead of downloading |

```bash
heulistic download <job-id>              # ./<job-id>-model.tar.gz
heulistic download <job-id> -o model.tgz
heulistic download <job-id> --url-only
```

Only completed jobs have a model; anything else is rejected with its current
status.

## heulistic export

Turn a completed job into a model you can run with Ollama or llama.cpp: the
trained adapter is merged into the base model, quantized to GGUF, and written
next to an Ollama Modelfile whose prompt template matches the one the model
was trained with. It runs in the local stack and costs nothing.

| Option | |
|---|---|
| `--quant`, `-q` | `Q4_K_M`, `Q5_K_M` or `Q8_0`. Default: the best quality that fits this machine |
| `--name`, `-n` | Ollama model name. Default `heulistic-<first 8 of the job id>` |
| `--output`, `-o DIR` | Where to write the GGUF and Modelfile. Default `./<name>` |
| `--no-ollama` | Do not run `ollama create`, even if Ollama is installed |

```bash
heulistic export <job-id>                # pick a format, export, add it to Ollama
heulistic export <job-id> -q Q8_0        # near-lossless, twice the size of Q4_K_M
ollama run heulistic-<id>
```

Without `--quant`, the format is chosen from your GPU's memory, or your RAM
when there is no NVIDIA GPU (Macs run GGUF from unified memory). If Ollama is
installed the model is added to it; otherwise the two commands to do that are
printed.

Exporting briefly needs two full-precision copies of the model on disk — about
5.4 GB per billion parameters plus 10 GB: ~15 GB for a 1B model, ~48 GB for 7B. If Docker's disk
is too small, the command says so before starting; large models are what
[Heulistic Cloud](https://heulistic.com) is for. Exporting the same format twice
reuses the first export.

## Managing the stack

### heulistic start

| Option | |
|---|---|
| `--port INTEGER` | Frontend port. Default 3000 |
| `--api-port INTEGER` | API port. Default 8000 |
| `--hf-token TEXT` | Token for gated models. Also read from `HF_TOKEN` |
| `--no-gpu` | CPU-only mode |
| `--gpus-per-job INTEGER` | Give each job this many GPUs and run several jobs at once. Default 0: every GPU to one job at a time |
| `--image-tag TEXT` | Image tag to run. Defaults to this CLI's version |

Pre-flight checks Docker and the ports, then pulls images and opens your
browser. A **defaulted** port that is busy steps aside to the next free one and
says so. A port you **name** is never moved — you get an error instead, since
starting somewhere other than where you asked would be worse.

If no NVIDIA driver is found it falls back to CPU with a warning. Training on
CPU works but is very slow.

Container images are **pinned to the CLI's own version**, so `heulistic 0.1.3`
runs `v0.1.3` images and nothing else. That makes a version mismatch an
explicit pull failure rather than a silent one — a stale cached image once
masked a release that had never been published at all. To run the newest
published set instead:

```bash
heulistic start --image-tag latest
export HEULISTIC_IMAGE_TAG=latest     # or set it once
```

### heulistic status

What is running, and what your jobs are doing.

```
Heulistic local stack

  db          running (healthy)
  minio       running
  api         running             http://localhost:8000
  scheduler   running
  frontend    running             http://localhost:3000

  Jobs: 1 completed, 2 running
```

Exits `0` only when every core service is up, so it can gate a script:
`heulistic status && heulistic train config.yaml`. A service listed as
`not created` means the stack is partly down — try `heulistic logs`.

### heulistic stop

Shuts the stack down. Needs nothing else installed, so it still works when the
rest of the install is broken.

### heulistic logs

| Option | |
|---|---|
| `--follow`, `-f` | Follow output |

Logs from the stack's own services. For a *job's* training logs, use
`heulistic train` (which streams them) or open the job in the dashboard.

## Exit codes

| Code | |
|---|---|
| `0` | Success — or, for `status`, everything healthy |
| `1` | Problems found: validation errors, a failed job, a stack that is down |
| `2` | Bad invocation — unreadable file, conflicting options |

`validate-config` returns `1` when it finds errors, so it drops into CI without
extra scripting. Add `--strict` to fail on warnings too.

## Pointing the CLI elsewhere

`HEULISTIC_API_URL` overrides API discovery, for a stack on a non-default port
or another machine:

```bash
HEULISTIC_API_URL=http://localhost:8002 heulistic status
```

Without it, the CLI finds the API through the stack it started itself.
