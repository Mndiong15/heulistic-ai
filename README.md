# Heulistic

**Fine-tune LLMs with [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) on your own GPU — then run the result in Ollama.**

Heulistic wraps Axolotl in a small local stack: it checks your config before you spend a GPU hour on it, runs the job in Docker with live logs and metrics, tells you whether the fine-tune actually beat the base model, and exports it to a GGUF you can `ollama run`.

```bash
pip install heulistic
heulistic init                        # a starter config sized to your GPU
heulistic validate-config config.yaml # catch mistakes offline, before training
heulistic start                       # bring up the local stack and dashboard
heulistic train config.yaml           # train and follow the logs
heulistic export <job-id>             # GGUF + Modelfile, added to Ollama
```

Free, local, no account.

## Why

Axolotl is powerful, but a typo in a YAML file or a model that doesn't fit your card usually shows up twenty minutes into a run. Heulistic moves those failures to before the run, and handles the parts around training:

- **Config checks against Axolotl's own schema.** Field names, types, defaults and required settings come from Axolotl's config reference — unknown fields get a "did you mean", wrong types are caught, and `--gpu` checks the job fits your VRAM. Works offline, no Docker, no GPU.
- **A starter config that fits.** `heulistic init` reads your GPU and picks a base model that will train on it.
- **One command to train.** Jobs queue and run in Docker; logs stream to your terminal, and the local dashboard charts loss, learning rate and GPU use.
- **Multiple GPUs, done right.** A model that fits one card trains data-parallel; one that doesn't is sharded with DeepSpeed ZeRO-3 automatically. `--gpus-per-job` runs several smaller jobs side by side instead.
- **Before vs after.** A few rows are held out of training, and both the base model and your fine-tune are scored on them, with three answers side by side — so you know whether it helped.
- **Run it locally.** Export merges the adapter into the base model, quantizes to GGUF (Q4_K_M, Q5_K_M or Q8_0) and writes an Ollama Modelfile with the same prompt format the model was trained on — a mismatched template is the most common reason a fine-tune "doesn't work" in Ollama.

## Requirements

| To | You need |
|---|---|
| Check configs (`init`, `validate-config`) | Python 3.10+ |
| Train, export, run the dashboard | Python 3.10+, Docker, and for real training an NVIDIA GPU with the [Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) |

`heulistic start` checks for Docker, the driver and the toolkit, and says what is missing. Without a GPU it falls back to CPU — fine for checking a config and dataset end to end, far too slow for real training.

## Quick start

```bash
pip install heulistic

heulistic init                        # writes config.yaml for a model that fits your GPU
heulistic validate-config config.yaml --gpu
heulistic start                       # opens the dashboard at http://localhost:3000
heulistic train config.yaml --dataset ./data.jsonl
```

When the job finishes, the dashboard shows the before/after comparison. To run the model:

```bash
heulistic export <job-id>             # picks the best format for this machine
ollama run heulistic-<job-id-prefix>
```

Datasets are JSONL, one example per line — or point `datasets:` in the config at a Hugging Face dataset. Gated base models (Llama, Gemma…) need `--hf-token`.

## Commands

| Command | |
|---|---|
| `heulistic init` | Write a starter config sized to your GPU (`-m` to choose the model) |
| `heulistic validate-config FILE` | Check a config; `--gpu` or `--vram 24` to check it fits, `--strict` / `--json` for CI |
| `heulistic start` | Start the local stack; `--gpus-per-job N` to run jobs side by side, `--no-gpu` for CPU |
| `heulistic train FILE` | Submit a job and follow it; `--dataset`, `--hf-token`, `--detach` |
| `heulistic status` | What's running, and how the jobs are doing |
| `heulistic logs [-f]` | Stack logs |
| `heulistic export JOB` | GGUF + Modelfile, registered with Ollama if installed; `-q Q8_0` to choose |
| `heulistic download JOB` | The raw trained model (adapter or full weights) |
| `heulistic stop` | Stop the stack |
| `heulistic login` / `logout` | Use Heulistic Cloud from the CLI (API key from heulistic.com/account); `--local` on any command uses the local stack |

Every command takes `--help`. The full reference is in [cli/README.md](cli/README.md) and in the dashboard's Docs page.

## How it works

`heulistic start` runs a small stack in Docker:

| Service | Does |
|---|---|
| `frontend` | The dashboard — submit jobs, watch them, compare and export |
| `api` | Validates and queues jobs, streams logs and metrics |
| `scheduler` | Starts jobs as GPUs free up, one container per job |
| `training` | Axolotl + PyTorch; runs the job, the before/after evaluation, and exports |
| `db`, `minio` | Postgres for jobs, MinIO for configs, datasets and models |

Everything stays on your machine. Images come from Docker Hub as `heulistic/*`, pinned to the CLI's version.

## Repository layout

```
cli/             the heulistic package on PyPI: the CLI, and heulistic.axolotl
                 (Axolotl config schema, validation, VRAM and GPU planning)
api/             FastAPI service
scheduler/       job dispatcher
training/        the training container: Axolotl, evaluation, GGUF export
frontend-local/  the dashboard (Next.js)
```

## Development

```bash
docker compose up --build        # the whole stack from source

pip install -e ./cli pytest      # the heulistic package, used by api/ and training/ too
(cd cli && pytest)               # CLI and config schema

docker compose up -d db          # the API tests need Postgres
(cd api && pip install -r requirements.txt && pytest)
(cd scheduler && pip install -r requirements.txt && pytest)
(cd training && pip install -r requirements.txt && pytest)
(cd frontend-local && npm ci && npm test)
```

Bug reports and pull requests are welcome. Please include the config (with secrets removed) and the output of `heulistic validate-config` for config-related issues.

## Heulistic Cloud

Need bigger GPUs than you have? [Heulistic Cloud](https://heulistic.com) runs the same jobs on managed cloud GPUs — pay for what you use. Once you have an account, the same commands run your jobs there:

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
heulistic.com/account. Keys last a year and can commit up to $100 in 24 hours;
jobs bigger than that go through the website.

## License

MIT — see [cli/LICENSE](cli/LICENSE).
