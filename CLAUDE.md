# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Heulistic** — fine-tune large language models with Axolotl on your own GPUs. Users bring an Axolotl YAML config and a dataset; Heulistic validates it, queues and runs the job in Docker, streams logs and metrics, compares the result with the base model, and exports it to GGUF for Ollama.

This repository is the open-source local product: the `heulistic` CLI, the shared validator, the API, scheduler and training container, and the local dashboard. The hosted service (Heulistic Cloud) runs the same API, scheduler and trainer on AWS; its cloud-only parts — Cognito auth, Stripe billing and the EC2 provisioner — are in this code but switched off when `LOCAL_MODE=true`. The cloud website and infrastructure are not in this repository.

## Two repositories

This repository is Heulistic Cloud. The open-source repository is generated from it:

```bash
python scripts/export_public.py   # into ../heulistic; then review, commit, push there
```

It is this repository minus the cloud-only paths listed in `PRIVATE_PATHS` in that script — `frontend/` (cloud website), `infra/` (Terraform), `studio/` (blog CMS), `netlify.toml` and `.github/workflows/cloud.yml` (cloud checks and the production deploy). Everything else is shared, so keep cloud-only things in those paths: a workflow, compose file or Dockerfile outside them that refers to one of them makes the export fail. Those directories have their own CLAUDE.md.

**Cloud-only features inside shared code** (something the managed platform offers that the free version does not) go in a `cloud_only/` folder inside the service — for the API, `api/app/cloud_only/`, registered in its `register(app)`, with tests in `api/tests/cloud_only/`. The export deletes every `cloud_only/` folder. Shared code must not import from one: the API reaches it only through `load_cloud_extensions` in `api/app/extensions.py`, which does nothing when the folder is absent, and the export fails if anything else imports it.

## Tech Stack

| Layer | Technology |
|---|---|
| Dashboard | Next.js 14 (App Router), TypeScript, Tailwind CSS, Recharts — ships as `heulistic/frontend-local` |
| API | FastAPI (Python 3.12), SQLAlchemy 2 (async), Alembic, PostgreSQL 15 |
| Auth | AWS Cognito (JWT-based); local mode uses a fixed test user |
| Storage | AWS S3 in production; MinIO in local mode |
| Caching | AWS EFS (`/efs/datasets/`, `/efs/models/`) — shared across training containers |
| Compute | AWS EC2 GPU (g4dn / g5 / g6 / g6e families) in production; Docker sibling containers locally |
| Training | Axolotl + PyTorch — LoRA, QLoRA, full fine-tune |
| Billing | Stripe (credit packages + off-session overage charges) |
| Container | Docker (images on Docker Hub as `heulistic/*`) |
| CLI | `heulistic` Python package (`/cli`) |
| Config validation | `heulistic.axolotl`, inside the `heulistic` package — schema generated from Axolotl's config reference; shared by the CLI, API and trainer |

## Repository Layout

```
/api              FastAPI service
/scheduler        Job dispatcher + watchdog
/training         Axolotl training container entrypoint
/frontend-local   Local dashboard (Next.js)
/cli              The `heulistic` package on PyPI: the CLI, and `heulistic.axolotl` (Axolotl schema,
                  validation, VRAM/GPU planning) that the API and trainer install from here
/scripts          Release checks and the open-source export
```

## Architecture

Four services, run together by `docker-compose.yml` (or `heulistic start`):

1. **Dashboard** (`/frontend-local`) — Next.js. Pages: `/train` (job submit), `/jobs` (list), `/jobs/[id]` (monitoring, before/after report, export), `/docs`.

2. **API** (`/api`) — FastAPI. Handles all user-facing and internal endpoints, billing settlement, SSE log streaming, and S3 presigned URLs. Runs Alembic migrations on startup.

3. **Scheduler** (`/scheduler`) — Python process on EC2 t3.small. Polls DB every 10 s; dispatches eligible jobs; runs watchdog every N ticks to terminate orphaned instances. Two provisioner backends: `provisioner.py` (EC2) and `local_provisioner.py` (Docker via socket proxy).

4. **Training Container** (`/training`) — Ephemeral Docker image per job. Downloads config + dataset from S3/EFS, patches the Axolotl config, runs training, streams logs + metrics to API, uploads model and checkpoints to S3, then exits.

### Job Lifecycle

```
Submit → API validates + uploads to S3 + reserves credits → Scheduler claims + provisions instance →
Training boots + fetches config/dataset → Axolotl runs → metrics/logs POSTed to API →
Checkpoint saved to S3 every N steps → Model tar'd + uploaded to S3 →
complete POST → API settles billing (refund or overage charge) → User downloads via presigned URL
```

### Job Statuses

`pending` → `provisioning` → `running` ↔ `checkpointing` → `completed` / `failed` / `cancelled`

### Data Paths

- Configs: `s3://<bucket>/configs/{job_id}/config.yaml`
- Datasets: `s3://<bucket>/temp-datasets/{user_id}/{timestamp}/data.jsonl`
- Models: `s3://<bucket>/models/{job_id}/model.tar.gz`
- Checkpoints: `s3://<bucket>/checkpoints/{job_id}/step-{n}/checkpoint.tar.gz`
- EFS datasets cache: `/efs/datasets/{sha256_prefix}/data.jsonl`
- EFS HuggingFace cache: `/efs/models/`

## API Endpoints

**Public (frontend → API) — require Cognito JWT:**
- `POST /api/jobs` — submit job (multipart: config YAML, optional dataset, optional HF token, optional instance_type)
- `GET /api/jobs` — list jobs for current user
- `GET /api/jobs/:id` — job detail
- `GET /api/jobs/:id/logs` — SSE log stream
- `GET /api/jobs/:id/metrics` — training metrics
- `DELETE /api/jobs/:id` — cancel job (settles billing immediately)
- `POST /api/jobs/:id/resume` — resume a failed job from its last checkpoint
- `GET /api/jobs/:id/download` — presigned S3 URL for completed model (the GGUF, for an export job)
- `GET /api/jobs/:id/export-options` — GGUF formats with file size, memory to run, cost
- `POST /api/jobs/:id/export` — queue an export job (merge → GGUF → Ollama Modelfile) of a completed training job
- `POST /api/estimate` — cost estimate from config YAML
- `POST /api/validate` — validate an Axolotl config; returns diagnostics (always 200)
- `GET /api/credits` — credit balance + transaction history
- `POST /api/credits/checkout` — create Stripe checkout session
- `GET /api/instances` — available instance types + rates
- `POST /api/stripe/webhook` — Stripe webhook receiver
- `GET /api/admin/...` — admin-only endpoints (scheduler heartbeat, job list)

**Internal (training container → API) — require per-job `X-Internal-Token`:**
- `POST /internal/jobs/:id/heartbeat` — marks job running on first call, refreshes liveness
- `POST /internal/jobs/:id/logs` — append log lines
- `POST /internal/jobs/:id/metrics` — append training metrics
- `PATCH /internal/jobs/:id/checkpoint` — record latest checkpoint S3 path + step
- `POST /internal/jobs/:id/evaluation` — before/after report written at the end of training
- `POST /internal/jobs/:id/complete` — mark completed or failed; triggers billing settlement (an export also sends its Modelfile)

**Local mode only:**
- `GET /local/jobs/:id/download` — direct S3/MinIO download (no presigned URL needed)

## Database Schema

Managed by Alembic (12 migrations). Key tables:

- **`users`** — Cognito sub as PK, email, `credit_balance_cents`, Stripe fields (`customer_id`, `subscription_id`, `payment_method_id`, `payment_past_due`)
- **`jobs`** — UUID PK, `session_id` (= user id), status, config YAML + S3 paths, instance info, cost fields, timestamps, checkpoint fields, `internal_token` (SHA-256 hash); `kind` (`train` | `export`), `source_job_id` + `export_quant` + `modelfile` for exports, `disk_gb` (instance volume), `evaluation` (JSONB before/after report)
- **`metrics`** — per-step: loss, learning_rate, gpu_utilization, gpu_memory_used_mb, tokens_per_second
- **`logs`** — append-only log lines per job
- **`credit_transactions`** — ledger of all credit movements (top-up, reservation, refund, overage)
- **`scheduler_heartbeat`** — singleton row (id=1) written every scheduler tick

## Key Business Rules

- Auth: AWS Cognito JWT required on all public endpoints; `ADMIN_EMAILS` env var grants admin access
- Billing: 1 credit = $0.01; job submission reserves 1.5× estimated cost; difference refunded/charged at completion
- Overage: if balance goes negative at completion or cancellation, immediately charges the saved Stripe card off-session
- Local mode: `LOCAL_MODE=true` disables billing, Cognito, and S3 (uses MinIO); safeguard prevents use with a live Stripe key
- Dataset limit: 50 GB (`MAX_DATASET_SIZE_MB`). Cloud browsers upload straight to S3 via presigned multipart (`/api/uploads`) and submit a `dataset_key`; files attached to the job stream through the API (capped at 5 GB in cloud, `MAX_PROXIED_UPLOAD_MB`). The API only reads the first 1 MB: minimum 10 examples, first 100 lines validated as JSON. Datasets live on the training instance's disk, not EFS
- Concurrency: submit never rejects for capacity — jobs queue. Anti-abuse caps at submit: 50 unfinished jobs/user, 100 submissions/hour/user. The scheduler (`scheduler/dispatch.py`) starts jobs while their instances fit the EC2 vCPU quota (`VCPU_QUOTA`, 64), in fair order (each user's k-th job before anyone's k+1-th), one user capped at `USER_VCPU_SHARE` (0.5) of the quota while others wait, and no small job jumps a big one waiting for room. Local mode counts jobs instead: one at a time with every GPU, or GPU count ÷ `heulistic start --gpus-per-job N` with each job pinned to its own cards
- Instance types: g4dn, g5, g6, g6e families; auto-selected from config or user-specified
- Checkpoint retention: 3 checkpoints kept in S3 per job (older ones pruned)
- Resume: creates a new job record pointing at the source job's last checkpoint; given queue priority (position 0)
- After training: the trainer holds out up to 50 rows (5%, datasets ≥ 100 rows, unless `val_set_size` is set) and runs `training/model_eval.py` to compare base vs fine-tune; it never fails the job. Export is a separate job (`JOB_KIND=export`, `training/export.py`): `axolotl merge-lora` → `axolotl export` (llama.cpp, pinned in the training image) → GGUF + Ollama Modelfile. Cloud exports run on g4dn with disk sized by `heulistic.axolotl.export_scratch_gb`; local exports use no GPU and take no scheduler slot
- Theming: colours are CSS-variable tokens (`ink`, `paper`, `line`, `inverse`, `danger`…) in each app's `globals.css`/`tailwind.config.ts`; `.dark` on `<html>` switches them. Use tokens, not hex — dark surfaces (`bg-inverse`, code, the log terminal) keep white text
- HF token: validated against HF API on submit; stored in AWS Secrets Manager; deleted after job completes/fails

## Billing Flow

1. Submit: estimate cost → reserve `ceil(estimate × 1.5)` credits → deduct from balance
2. Complete: actual cost = elapsed_hours × instance_rate × markup → settle (refund or charge diff)
3. Overage: if balance < 0 after settlement, charge saved Stripe card via off-session PaymentIntent
4. Top-up: Stripe Checkout session (packages: $50 + 10% bonus, $200 + 20% bonus; or custom amount)

## Local Mode Stack

`docker-compose.yml` in project root runs the full stack locally:
- `postgres:15`, `minio/minio` + `minio-init` (bucket setup)
- `api` (with `LOCAL_MODE=true`), `scheduler` (uses `local_provisioner.py`)
- `docker-proxy` (tecnativa/docker-socket-proxy) — lets scheduler run training containers safely
- `frontend` (built from `./frontend-local`)

The `heulistic` CLI manages this:
```bash
heulistic start            # pull images, start stack, open browser
heulistic stop             # docker compose down
heulistic logs [-f]        # tail compose logs
```

## Development Commands

```bash
# API
cd api && uvicorn app.main:app --reload      # dev server (port 8000)
cd api && pytest                              # all tests
cd api && pytest tests/test_jobs.py -k name  # single test

# Dashboard
cd frontend-local && NEXT_PUBLIC_LOCAL_MODE=true npm run dev   # port 3000
cd frontend-local && npm run build
cd frontend-local && npm run lint
cd frontend-local && npm run test             # vitest unit tests

# Scheduler
cd scheduler && python main.py

# Full local stack
docker compose up --build                    # or: heulistic start
```

## Testing

- **API**: pytest against a real **PostgreSQL** (asyncpg) — there is no SQLite path. `conftest.py` skips the DB-backed tests when Postgres is unreachable, so a green run with most tests skipped means the database was not up. Start one with `docker compose up -d db`. Tests in `api/tests/`: `test_jobs.py`, `test_internal.py`, `test_credits.py`, `test_webhooks.py`, `test_cost.py`, `test_estimate.py`, `test_instances.py`, `test_admin.py`, `test_validate.py`.
- **Config schema** (`heulistic.axolotl`): pytest in `cli/tests/axolotl/`. Refreshing it for a new Axolotl release: `cli/reference/README.md`.
- **CLI**: pytest in `cli/tests/` (this includes `cli/tests/axolotl/`).
- **Scheduler**: pytest in `scheduler/tests/`: `test_main.py`, `test_provisioner.py`, `test_local_provisioner.py`, `test_watchdog.py`.
- **Training**: pytest in `training/tests/`: `test_train_utils.py`.
- **Dashboard**: vitest in `frontend-local/src/lib/*.test.ts`.

## Environment Variables

Key vars for the API (see `api/.env.example`):
- `DATABASE_URL` — Postgres connection string
- `S3_BUCKET`, `AWS_REGION`
- `LOCAL_MODE` — `true` for local dev (disables billing + Cognito)
- `S3_ENDPOINT_URL`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` — MinIO config (local mode)
- `COGNITO_USER_POOL_ID`, `COGNITO_CLIENT_ID`, `COGNITO_REGION`
- `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`
- `BILLING_ENABLED` — `false` to disable billing on non-local cloud deployments
- `ADMIN_EMAILS` — comma-separated list of admin email addresses
- `FRONTEND_URL` — used for CORS and Stripe redirect URLs
- `SENTRY_DSN`
