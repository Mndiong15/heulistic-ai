---
title: FAQ
description: Frequently asked questions about Heulistic.
order: 5
---

## General

### What models can I fine-tune?

Any model on HuggingFace Hub that Axolotl supports — Llama, Mistral, Gemma, Qwen, Phi, Falcon, and more. Gated models (like Llama 3) require a HuggingFace token with approved access.

### What format does my dataset need to be in?

JSONL (one JSON object per line). The required keys depend on your dataset type:

- `alpaca` → `instruction`, `output` (and optionally `input`)
- `sharegpt` → `conversations` (list of `{from, value}` objects)
- `completion` → `text`
- `input_output` → `input`, `output`

Minimum 10 examples required; maximum 50 GB. Datasets on the Hugging Face Hub have no size limit.

### Can I use a HuggingFace dataset instead of uploading a file?

Yes. In your config, set `datasets[0].path` to the HuggingFace dataset ID and leave the file upload blank. For gated or private datasets, provide your HF token.

### How long does training take?

It depends on model size, dataset size, number of epochs, and instance type. Rough estimates:

- 7B LoRA, 1K examples, 3 epochs on g5.xlarge: ~15–30 min
- 7B full fine-tune, 10K examples, 3 epochs on g5.2xlarge: ~2–4 hrs
- 13B QLoRA, 5K examples, 3 epochs on g6e.xlarge: ~45–90 min

The Train page shows an estimate before you submit.

---

## Jobs & Training

### My job has been pending for a long time. What's happening?

Locally, jobs run one at a time by default, because each one is given every GPU in the machine. Others wait in the queue — the Jobs page and `heulistic status` show the backlog. Resumed jobs go first.

### Can I run multiple jobs at once?

Yes, on a machine with several GPUs: start with `heulistic start --gpus-per-job 1` and each job gets its own card, so a 4-GPU machine runs four at once. `--gpus-per-job 2` runs two jobs on two cards each. See [Hardware](/docs/hardware).

### Did fine-tuning actually help?

Every job ends with a before/after comparison. A few rows of your data — 5%, at
most 50, and only when there are at least 100 — are held out of training. Both
the base model and the fine-tune are scored on them, and the job page shows
the loss for each plus three answers side by side. Set `val_set_size: 0` in
your config to keep every row for training and skip it, or set your own
`val_set_size` to choose the split.

### How do I run my model with Ollama?

Click **Export** on a completed job, or run `heulistic export <job-id>`. You get
a GGUF (Q4_K_M, Q5_K_M or Q8_0) and a Modelfile that uses the prompt format the
model was trained with — then `ollama create` and `ollama run`. The job page
shows how much memory each format needs to run.

### What happens if my job fails?

The error message is shown on the job page, pulled directly from the Axolotl traceback. Common causes:

- **OOM** — reduce batch size, enable gradient checkpointing, or use a larger instance
- **Gated model** — your HF token doesn't have access; request access at huggingface.co
- **Invalid config** — check the [Config Reference](/docs/config-reference) for required fields

If a checkpoint was saved before the failure, you can resume from it.

### What are checkpoints?

Heulistic saves a checkpoint every 100 steps (configurable via `save_steps` in your config). Up to 3 checkpoints are kept in S3 per job. If your job fails, you can resume from the latest checkpoint instead of starting over.

### Does Heulistic support multi-GPU training?

Yes, on one machine. The trainer uses every GPU it can see. A model that fits on one card trains data-parallel (each GPU holds a full copy, so VRAM does not add up across cards). An unquantized model too big for one card is sharded across the GPUs with DeepSpeed ZeRO-3 automatically; QLoRA and 8-bit models are not. Your own `deepspeed` or `fsdp` settings are respected. See [Hardware](/docs/hardware).

Multi-*node* training (several machines) is not supported: `use_ray` and `ray_num_workers` above 1 are rejected at submission.

---

## Data & Privacy

### Is my dataset stored permanently?

No. Uploaded datasets are deleted after training completes (or fails). Trained models are retained for **7 days**, then automatically deleted by S3 lifecycle policy.

### Is my HuggingFace token stored?

Your token is stored in AWS Secrets Manager for the duration of the job. It is permanently deleted once the job completes or fails.

### Who can see my jobs?

Only you. Jobs are scoped to your account. Admins can view all jobs for platform management purposes.

---
