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
| `heulistic start` / `train` / `export` / `download` / `status` | yes |

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
heulistic export <job-id>             # GGUF + Ollama Modelfile, ready to run
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

## Did it help? Before vs after

When a training job finishes, Heulistic scores the base model and your
fine-tune on rows held out of training (5% of the dataset, at most 50, for
datasets of 100+ rows) and shows the loss for each plus three answers side by
side in the dashboard. Set `val_set_size: 0` in your config to keep every row
for training, or set your own `val_set_size` to choose the split.

## Running it with Ollama

```bash
heulistic export <job-id>                # best format for this machine, added to Ollama
heulistic export <job-id> -q Q8_0        # choose: Q4_K_M, Q5_K_M or Q8_0
heulistic export <job-id> --no-ollama    # just write the files
ollama run heulistic-<first 8 of the job id>
```

Export merges the trained adapter into the base model, quantizes it to GGUF,
and writes an Ollama Modelfile whose prompt template matches the one the model
was trained with — so it answers the way it did in training. Without `-q`, the
format is chosen from your GPU's memory, or your RAM when there is no NVIDIA
GPU. It runs in the local stack and needs free disk for two full-precision
copies of the model while it works (about 48 GB for a 7B model); the command
checks first.

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

## Heulistic Cloud

Need more GPU or don't want to manage infrastructure? [Heulistic Cloud](https://heulistic.com) handles provisioning, scaling, and cost — pay only for what you use.

Once you have an account, the same commands run your jobs there:

```bash
heulistic login                          # paste an API key from heulistic.com/account
heulistic train config.yaml              # now runs on Heulistic Cloud
heulistic download <job-id>
heulistic export <job-id>                # a cloud export; the GGUF downloads here

heulistic train config.yaml --local      # one command on your local stack instead
heulistic logout                         # back to local for everything
```

On a server or in CI, set `HEULISTIC_API_KEY` instead of logging in. The key
is saved in `~/.heulistic/credentials.json`, readable only by you; revoke it at
heulistic.com/account.
