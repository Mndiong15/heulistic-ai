---
title: GPU & Memory
description: Whether a config fits your card, and what to do when it does not.
order: 4
---

## Will it fit?

Ask before you train:

```bash
heulistic validate-config config.yaml --gpu
```

```
Checking config.yaml against NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)

  error   Needs roughly 17 GB of VRAM but NVIDIA GeForce RTX 4060 Laptop GPU
          has 8 GB. Use a smaller model, switch to 'adapter: qlora' with
          'load_in_4bit: true', or pick larger hardware.
```

No GPU in the machine you are planning on? Give the numbers directly:

```bash
heulistic validate-config config.yaml --vram 24
heulistic validate-config config.yaml --vram 88 --gpus 4
```

## How the estimate works

Two cases, driven by the parameter count in the model name:

- **LoRA / QLoRA** — `params × (bits ÷ 8) × 1.2`
- **Full fine-tune** — `params × 16 × 1.2`

Bit width comes from the config: 4 with `load_in_4bit` or `adapter: qlora`,
8 with `load_in_8bit`, otherwise 16. The `1.2` covers the adapter, optimiser
state and activations.

| Model | QLoRA (4-bit) | LoRA (16-bit) | Full fine-tune |
|---|---|---|---|
| 0.5B | 0.3 GB | 1.2 GB | 9.6 GB |
| 1B | 0.6 GB | 2.4 GB | 19 GB |
| 3B | 1.8 GB | 7.2 GB | 58 GB |
| 7B | 4.2 GB | 17 GB | 134 GB |
| 8B | 4.8 GB | 19 GB | 154 GB |
| 13B | 7.8 GB | 31 GB | 250 GB |
| 70B | 42 GB | 168 GB | 1.3 TB |

The jump to full fine-tuning is the thing people underestimate: the same 7B
model needs **4 GB as QLoRA and 134 GB without an adapter.** If you have one
consumer card, QLoRA is not a compromise, it is the only option.

A config using more than 75% of your VRAM is flagged as tight, and one needing
more than the card has as an error. With several GPUs, both are judged per card.

**The estimate is a floor, not a ceiling.** It does not model activation memory,
which grows with `sequence_len` and `micro_batch_size` — so a config that
"fits" at 90% may still OOM at long sequence lengths. That headroom is why
`heulistic init` sizes to 75% of your card rather than 100%.

## When you hit CUDA out of memory

Roughly in order of effect:

1. **`micro_batch_size: 1`** and raise `gradient_accumulation_steps` to keep the
   effective batch size. Costs nothing in quality.
2. **`gradient_checkpointing: true`** — recomputes activations instead of
   storing them. Meaningful saving for perhaps 20-30% slower steps.
3. **`adapter: qlora` with `load_in_4bit: true`** — the big one, as the table
   above shows.
4. **Lower `sequence_len`.** Activation memory scales with it, so 2048 → 1024
   is a large saving. Only do this if your samples actually fit.
5. **A smaller base model.** A well-tuned 3B often beats a half-trained 7B.

`validate-config` suggests these too, so you can check a change before
re-running:

```bash
heulistic validate-config config.yaml --gpu   # after each edit
```

## Multiple GPUs

Several GPUs in one machine are supported — the trainer detects them and runs
`accelerate launch` across all of them. How the model is spread depends on
whether it fits on one card. VRAM does not pool: two 24 GB cards are not one
48 GB card.

- **Fits on one GPU** — data parallel (DDP). Every GPU holds a full copy of
  the model and trains on a different slice of each batch. More GPUs means
  faster training, not room for a bigger model.
- **Too big for one GPU, unquantized** (LoRA bf16, full fine-tune) — sharded
  with DeepSpeed ZeRO-3, which splits weights, gradients and optimizer state
  across the cards. The trainer adds `deepspeed: zero3.json` for you and logs
  that it did. Validation checks the per-GPU share against your smallest card.
- **Too big for one GPU, quantized** (QLoRA, 8-bit) — not sharded
  automatically; QLoRA under ZeRO-3 is unreliable. It needs a card that holds
  it alone, so validation reports it as not fitting.

If your config sets `deepspeed` or `fsdp`, that is used as written and nothing
is added. Axolotl's bundled configs can be referenced by their repo path, e.g.
`deepspeed: deepspeed_configs/zero2.json`. With ZeRO-1/2 the weights are still
copied to every GPU, so validation can only estimate the fit and says so.
Settings like `world_size`, `tensor_parallel_size`, `dp_shard_size` and
`context_parallel_size` are passed through to Axolotl unchanged.

Logged tokens per second counts all GPUs together.

What is **not** supported is multi-*node* training: `use_ray` and
`ray_num_workers > 1` need a cluster of machines, and are rejected at
submission.

## Running several jobs at once

By default, jobs run one at a time and each is given every GPU in the machine —
the fastest choice for a single job. Submit as many as you like; they queue,
and `heulistic status` shows the backlog.

For a sweep on a multi-GPU machine, give each job a share of the cards instead:

```bash
heulistic start --gpus-per-job 1   # 4 GPUs: four jobs at once, one card each
heulistic start --gpus-per-job 2   # 4 GPUs: two jobs at once, two cards each
```

Each job is pinned to cards no other job holds, and the rest wait their turn.
A job only sees its own cards, so check that it fits them:
`heulistic validate-config config.yaml --vram 24` for one 24 GB card.

## No GPU at all

Everything runs CPU-only — `heulistic start` detects the absence of a driver
and falls back automatically, and `--no-gpu` forces it.

This is genuinely useful for checking that a config, dataset and the pipeline
all work end to end. It is not useful for real training: expect CPU to be
slower by a factor of hundreds. Validate on CPU, train on a GPU.

## Driver visible, Docker blind

If `nvidia-smi` works but training cannot see the GPU, the NVIDIA Container
Toolkit is missing — Docker needs it to pass devices through.
`heulistic start` detects this exact case and prints the install commands for
your platform.
