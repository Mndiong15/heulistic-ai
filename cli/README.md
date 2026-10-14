# heulistic

Fine-tune LLMs without the infrastructure.

Run the full Heulistic stack locally on your own hardware — no cloud account, no billing, no setup beyond Docker.

## Installation

```bash
pip install heulistic
```

Python 3.10+. **Docker is not needed to install, and not needed for every
command** — only for the ones that run the training stack.

| | Needs Docker |
|---|---|
| `heulistic init` — write a starter config, sized to your GPU | no |
| `heulistic validate-config` — check a config against Axolotl's schema | no |
| `heulistic start` / `train` / `download` / `status` | yes |

So you can install it purely to check configs — offline, with nothing else
running. Training needs Docker, plus an NVIDIA GPU and the Container Toolkit
for anything beyond a smoke test; `heulistic start` checks for all three and
says what is missing.

## Quick start

```bash
pip install heulistic

heulistic init                        # starter config, sized to your GPU
heulistic validate-config config.yaml # check it — offline, no Docker needed
heulistic start                       # bring the local stack up
heulistic train config.yaml           # submit and follow the logs
heulistic download <job-id>           # collect the trained model
```

## Training from the terminal

```bash
heulistic train config.yaml
heulistic train config.yaml --dataset ./data.jsonl
heulistic train config.yaml --hf-token hf_...   # gated base models
heulistic train config.yaml --detach            # submit without following
heulistic train config.yaml --no-validate       # skip the pre-flight check
```

The config is validated locally first, so a typo costs nothing. Logs stream
until the run ends; Ctrl-C detaches without stopping the job. On success you
are told where to get the model.

```bash
heulistic download <job-id>              # ./<job-id>-model.tar.gz
heulistic download <job-id> -o model.tgz
heulistic download <job-id> --url-only
```

## Starting from a working config

```bash
heulistic init                    # writes config.yaml
heulistic init -m Qwen/Qwen2.5-0.5B
heulistic init -o experiments/run1.yaml
```

`init` reads your GPU and picks a base model that fits it, so your first run
does not OOM twenty minutes in. The generated file passes `validate-config`
with no findings.

## Checking a config

Validate an Axolotl config before spending a GPU hour on it. Works offline —
no Docker, no running stack, no account.

```bash
heulistic validate-config config.yaml
```

```
Checking config.yaml

  error   line 3   num_epochs        num_epochs must be greater than 0
  error   line 5   micro_batch_size  'micro_batch_size' expects int | None, got str
  warning line 2   lr                Unknown field 'lr' — did you mean 'learning_rate'?
  info             bf16              'bf16' is not set — it defaults to 'auto'

  2 errors, 1 warning, 1 suggestion
```

Field names, types, defaults and required-ness come from Axolotl's own
[config reference](https://docs.axolotl.ai/docs/config-reference.html), so the
checks track upstream rather than a hand-written list.

```bash
# Check it also fits your GPU
heulistic validate-config config.yaml --gpu

# ...or a card you don't have in front of you
heulistic validate-config config.yaml --vram 24
heulistic validate-config config.yaml --vram 88 --gpus 4

# Fail on warnings too, for CI
heulistic validate-config config.yaml --strict

# Errors only, or JSON for tooling
heulistic validate-config config.yaml --quiet
heulistic validate-config config.yaml --json
```

Exit codes: `0` clean, `1` problems found, `2` bad invocation.

## Running the local stack

```bash
# Start the stack (opens your browser automatically)
heulistic start

# GPU-less mode (CPU only — training will be slow)
heulistic start --no-gpu

# Several jobs at once on a multi-GPU machine (one card each)
heulistic start --gpus-per-job 1

# Use a HuggingFace token for gated models
heulistic start --hf-token hf_...

# Custom ports
heulistic start --port 3001 --api-port 8001

# Images are pinned to the CLI version; override when you need to
heulistic start --image-tag latest

# What's running, and what are my jobs doing?
heulistic status

# View logs
heulistic logs
heulistic logs -f   # follow

# Stop
heulistic stop
```

`heulistic status` exits `0` only when every service is up, so it can gate a
script:

```
Heulistic local stack

  db          running (healthy)
  minio       running
  api         running             http://localhost:8000
  scheduler   running
  frontend    running             http://localhost:3000

  Jobs: 1 completed, 2 running
```

## Managed Cloud

Need more GPU or don't want to manage infrastructure? [Heulistic Cloud](https://heulistic.com) handles provisioning, scaling, and cost — pay only for what you use.
