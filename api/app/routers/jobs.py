import asyncio
import json
import logging
import math

import uuid

logger = logging.getLogger(__name__)
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import httpx
import yaml
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from heulistic_axolotl import (
    GGUF_QUANTS,
    gguf_size_gb,
    inference_memory_gb,
    parse_param_billions,
    recommend_quant,
    validate,
)

from ..config import settings
from ..dependencies import get_current_user, get_db
from ..models.db import CreditTransaction, Job, Log, Metric, User
from ..schemas.jobs import (
    DownloadResponse,
    ExportOption,
    ExportOptionsResponse,
    ExportRequest,
    ExportSummary,
    JobCosts,
    JobDetail,
    JobListItem,
    JobSubmitResponse,
    MetricResponse,
)

from ..services import cost as cost_service
from ..services.cost import BUFFER_FACTOR
from ..services import s3 as s3_service
from ..services import secrets as secrets_service
from ..services.instances import get_rate, valid_instance_types
from ..services.stripe_service import charge_overage
from .uploads import require_own_dataset_key

router = APIRouter(prefix="/jobs", tags=["jobs"])

_DATASET_TYPE_REQUIRED_FIELDS: dict[str, set[str]] = {
    "alpaca":       {"instruction", "output"},
    "input_output": {"input", "output"},
    "completion":   {"text"},
    "sharegpt":     {"conversations"},
}
_MIN_EXAMPLES = 10
_MB = 1024 * 1024
# Only this much of a dataset is ever read by the API. It covers the 100 lines
# that are checked; the trainer reads the rest.
_DATASET_HEAD_BYTES = _MB
# Axolotl configs are a few KB; anything near this is not one.
_MAX_CONFIG_BYTES = _MB


def _validate_dataset(head: bytes, parsed_config: dict, *, complete: bool = True) -> None:
    """Validate a dataset from its first bytes.

    `complete` says whether `head` is the whole file. When it is not, the last
    line may be cut off and is dropped, and too few lines is not an error —
    a megabyte holding under ten rows just means the rows are long.
    """
    if complete and not head.strip():
        raise HTTPException(status_code=422, detail="Dataset file is empty")

    text = head.decode(errors="replace")
    if not complete:
        text = text[: text.rfind("\n") + 1]
    lines = [l for l in text.splitlines() if l.strip()]

    if complete and len(lines) < _MIN_EXAMPLES:
        raise HTTPException(
            status_code=422,
            detail=f"Dataset has only {len(lines)} example(s) — at least {_MIN_EXAMPLES} required",
        )

    for i, line in enumerate(lines[:100], 1):
        try:
            json.loads(line)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=422,
                detail=f"Dataset line {i} is not valid JSON",
            )

    dataset_type = ""
    datasets_cfg = parsed_config.get("datasets") or []
    if datasets_cfg and isinstance(datasets_cfg, list):
        dataset_type = str((datasets_cfg[0] or {}).get("type", "")).strip()

    required = _DATASET_TYPE_REQUIRED_FIELDS.get(dataset_type, set())
    if required:
        try:
            first = json.loads(lines[0])
        except Exception:
            return
        missing = required - set(first.keys())
        if missing:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Dataset type '{dataset_type}' requires fields {sorted(required)} — "
                    f"first row is missing: {sorted(missing)}"
                ),
            )


def _validate_config(
    config_yaml: str,
    has_dataset_upload: bool,
    *,
    check_hardware: bool = False,
    instance_type: str | None = None,
) -> None:
    """Reject a config the shared validator considers broken.

    Rules live in `heulistic-axolotl`, whose schema is generated from Axolotl's
    own config reference — so field names, types, defaults and required-ness
    track upstream instead of a hand-maintained list here. The CLI's
    `heulistic validate-config` runs exactly the same rules, so what passes
    locally passes on submit.

    Only `error` blocks. Warnings and hints are advisory and reach users
    through the UI and the CLI rather than a 422.

    With `check_hardware`, the config is also checked against the GPUs of
    `instance_type`. When none was chosen, the one selection would rent is
    used — and if even that does not fit, no instance does.
    """
    target = None
    auto_selected = False
    if not check_hardware:
        instance_type = None
    elif instance_type is None:
        parsed = yaml.safe_load(config_yaml)
        if isinstance(parsed, dict):
            instance_type = cost_service.select_instance(parsed)
            auto_selected = True
    if instance_type is not None:
        target = cost_service.hardware_for(instance_type)

    problems = []
    for d in validate(config_yaml, target=target):
        if d.level != "error":
            continue
        message = d.message
        if auto_selected and d.code == "vram.over":
            message = (
                f"No available instance has enough VRAM for this job. The closest "
                f"is {instance_type}: {message}"
            )
        problems.append(f"{d.field}: {message}" if d.field else message)

    # Dataset presence is decided here rather than in the shared rules, because
    # only the API knows whether a file came with the request. The shared
    # validator can only see the config, so it downgrades this to a warning.
    if not has_dataset_upload:
        parsed = yaml.safe_load(config_yaml)
        if isinstance(parsed, dict) and not (
            parsed.get("datasets") or parsed.get("pretraining_dataset")
        ):
            problems.append(
                "No dataset provided — upload a dataset file or specify "
                "'datasets' in the config"
            )

    # Reported together: someone fixing a config one 422 at a time learns
    # nothing about the other three things that are also wrong.
    if problems:
        raise HTTPException(
            status_code=422,
            detail="Config validation failed: " + "; ".join(problems),
        )


async def _get_job_or_404(
    db: AsyncSession, job_id: uuid.UUID, user_id: str, is_admin: bool = False
) -> Job:
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not is_admin and job.session_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return job


_UNFINISHED = ("pending", "provisioning", "running", "checkpointing")


async def _check_rate_limits(db: AsyncSession, user_id: str) -> None:
    """Anti-abuse limits only. Capacity is not checked here: the scheduler
    queues jobs until the GPU quota and the user's running limit allow them."""
    result = await db.execute(
        select(func.count()).where(
            Job.session_id == user_id,
            Job.status.in_(_UNFINISHED),
        )
    )
    if result.scalar_one() >= settings.max_unfinished_jobs_per_user:
        raise HTTPException(
            status_code=429,
            detail=(
                f"You have {settings.max_unfinished_jobs_per_user} unfinished jobs — "
                "wait for some to finish or cancel them before submitting more"
            ),
        )

    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
    result = await db.execute(
        select(func.count()).where(
            Job.session_id == user_id,
            Job.created_at >= since,
        )
    )
    if result.scalar_one() >= settings.max_submissions_per_hour:
        raise HTTPException(
            status_code=429,
            detail=f"Max {settings.max_submissions_per_hour} submissions per hour",
        )


def _too_large(limit_mb: int, hint: str = "") -> HTTPException:
    size = f"{limit_mb // 1024} GB" if limit_mb >= 1024 else f"{limit_mb} MB"
    return HTTPException(status_code=413, detail=f"Dataset exceeds the {size} limit{hint}")


async def _store_attached_dataset(
    dataset: UploadFile, parsed_config: dict, user_id: str
) -> tuple[str, int]:
    """Validate a dataset sent with the job and stream it to S3.

    It is never read whole: that is what took the API down on large files.
    Starlette has already spooled it to disk, so only the head is read into
    memory and the rest goes to S3 as a stream.
    """
    f = dataset.file
    f.seek(0, 2)
    size = f.tell()
    f.seek(0)

    limit_mb = min(settings.max_dataset_size_mb, settings.max_proxied_upload_mb)
    if size > limit_mb * _MB:
        hint = "" if settings.local_mode else " for files attached to the job — larger ones upload directly"
        raise _too_large(limit_mb, hint)

    head = await dataset.read(_DATASET_HEAD_BYTES)
    _validate_dataset(head, parsed_config, complete=len(head) >= size)
    f.seek(0)

    key = f"{s3_service.user_dataset_prefix(user_id)}{uuid.uuid4().hex}/data.jsonl"
    try:
        path = await asyncio.to_thread(s3_service.upload_fileobj, f, key)
    except (ClientError, BotoCoreError) as exc:
        raise HTTPException(status_code=503, detail="Storage service unavailable") from exc
    return path, size // _MB


async def _check_uploaded_dataset(
    key: str, parsed_config: dict, user_id: str
) -> tuple[str, int]:
    """Validate a dataset the browser already uploaded to S3, by its first bytes."""
    require_own_dataset_key(key, user_id)
    try:
        size = await asyncio.to_thread(s3_service.object_size, key)
        if size is None:
            raise HTTPException(
                status_code=422,
                detail="Uploaded dataset not found — the upload may not have finished, "
                       "or it expired. Upload it again.",
            )
        if size > settings.max_dataset_size_mb * _MB:
            raise _too_large(settings.max_dataset_size_mb)
        head = await asyncio.to_thread(s3_service.read_head, key, _DATASET_HEAD_BYTES)
    except (ClientError, BotoCoreError) as exc:
        raise HTTPException(status_code=503, detail="Storage service unavailable") from exc
    _validate_dataset(head, parsed_config, complete=len(head) >= size)
    return s3_service.s3_path_for(key), size // _MB


@router.post("", response_model=JobSubmitResponse, status_code=201)
async def submit_job(
    config: UploadFile = File(...),
    dataset: UploadFile | None = File(default=None),
    hf_token: str | None = Form(default=None),
    instance_type: str | None = Form(default=None),
    dataset_key: str | None = Form(default=None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if dataset is not None and dataset_key:
        raise HTTPException(
            status_code=422, detail="Send either a dataset file or a dataset_key, not both"
        )
    if settings.local_mode:
        instance_type = "local-gpu"
    elif instance_type and instance_type not in valid_instance_types():
        raise HTTPException(
            status_code=422,
            detail=f"Invalid instance_type '{instance_type}'. "
                   f"Valid options: {sorted(valid_instance_types())}",
        )

    if not settings.local_mode:
        await _check_rate_limits(db, user.id)

    config_bytes = await config.read(_MAX_CONFIG_BYTES + 1)
    if len(config_bytes) > _MAX_CONFIG_BYTES:
        raise HTTPException(status_code=413, detail="Config file is larger than 1 MB")
    try:
        parsed_config = yaml.safe_load(config_bytes)
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid YAML: {exc}")
    if not isinstance(parsed_config, dict) or "base_model" not in parsed_config:
        raise HTTPException(status_code=422, detail="Config must include 'base_model'")

    # Local mode runs on whatever GPU the machine has, which the API cannot see;
    # the trainer plans for it, and `heulistic validate-config` checks it.
    _validate_config(
        config_bytes.decode(),
        has_dataset_upload=dataset is not None or bool(dataset_key),
        check_hardware=not settings.local_mode,
        instance_type=instance_type,
    )

    job_id = uuid.uuid4()
    dataset_s3_path = None
    dataset_size_mb = 0

    if dataset is not None:
        dataset_s3_path, dataset_size_mb = await _store_attached_dataset(
            dataset, parsed_config, user.id
        )
    elif dataset_key:
        dataset_s3_path, dataset_size_mb = await _check_uploaded_dataset(
            dataset_key, parsed_config, user.id
        )

    config_key = f"configs/{job_id}/config.yaml"
    try:
        config_s3_path = s3_service.upload_bytes(config_bytes, config_key, "application/x-yaml")
    except (ClientError, BotoCoreError) as exc:
        raise HTTPException(status_code=503, detail="Storage service unavailable") from exc

    estimate = cost_service.parse_and_estimate(config_bytes.decode(), dataset_size_mb, instance_type)

    # Credit balance check — require 1.5× the estimate as a buffer against overruns
    estimated_cost_cents = int(float(estimate["estimated_cost"]) * 100)
    held_cents = math.ceil(estimated_cost_cents * float(BUFFER_FACTOR))
    if settings.billing_enabled:
        # Re-fetch user with a lock to get current balance
        result = await db.execute(select(User).where(User.id == user.id).with_for_update())
        user = result.scalar_one()
        if user.credit_balance_cents < held_cents:
            raise HTTPException(
                status_code=402,
                detail=(
                    f"Insufficient credits. Estimated cost is ${estimate['estimated_cost']:.2f} — "
                    f"we hold ${held_cents / 100:.2f} (50% buffer) and refund unused credits at completion. "
                    f"You have ${user.credit_balance_cents / 100:.2f}. "
                    "Visit /credits to top up."
                ),
            )

    if hf_token:
        hf_token = hf_token.strip()

    base_model = parsed_config.get("base_model", "")

    if hf_token or base_model:
        try:
            async with httpx.AsyncClient(timeout=5.0) as hf_client:
                if hf_token:
                    r = await hf_client.get(
                        "https://huggingface.co/api/whoami-v2",
                        headers={"Authorization": f"Bearer {hf_token}"},
                    )
                    if r.status_code >= 400:
                        raise HTTPException(
                            status_code=422,
                            detail="Invalid HuggingFace token — verify it at huggingface.co/settings/tokens",
                        )

                if base_model:
                    # HEAD the actual download gate — /api/models returns 200 for gated
                    # models even without access; the 403 only surfaces on file download
                    auth_headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
                    model_r = await hf_client.head(
                        f"https://huggingface.co/{base_model}/resolve/main/config.json",
                        headers=auth_headers,
                        follow_redirects=True,
                    )
                    if model_r.status_code == 404:
                        raise HTTPException(
                            status_code=422,
                            detail=f"Model '{base_model}' not found on HuggingFace",
                        )
                    if model_r.status_code in (401, 403):
                        if hf_token:
                            raise HTTPException(
                                status_code=422,
                                detail=f"Token does not have access to '{base_model}' — request access at huggingface.co/{base_model}",
                            )
                        else:
                            raise HTTPException(
                                status_code=422,
                                detail=f"'{base_model}' is a gated model — add a HuggingFace token with approved access to submit this job",
                            )
        except HTTPException:
            raise
        except Exception:
            pass  # Network error — don't block submission on HF API outage

    hf_token_secret_arn = None
    if hf_token:
        try:
            hf_token_secret_arn = secrets_service.store_hf_token(str(job_id), hf_token)
        except (ClientError, BotoCoreError) as exc:
            raise HTTPException(status_code=503, detail="Failed to store credentials") from exc

    # Store held_cents / 100 as estimated_cost so settlement math (held - actual) is correct
    held_cost = Decimal(str(held_cents)) / 100

    job = Job(
        id=job_id,
        session_id=user.id,
        status="pending",
        config_yaml=config_bytes.decode(),
        config_name=parsed_config.get("base_model", "unknown"),
        config_s3_path=config_s3_path,
        dataset_s3_path=dataset_s3_path,
        dataset_size_mb=dataset_size_mb or None,
        instance_type=estimate["instance_type"],
        estimated_cost=held_cost,
        estimated_duration_minutes=estimate["estimated_duration_minutes"],
        hf_token_secret_arn=hf_token_secret_arn,
    )
    db.add(job)
    # Flush job first so the FK on credit_transactions.job_id can resolve
    await db.flush()

    if settings.billing_enabled:
        user.credit_balance_cents -= held_cents
        db.add(CreditTransaction(
            user_id=user.id,
            amount_cents=-held_cents,
            description=f"Reserved ${estimate['estimated_cost']:.2f} est. + 50% buffer",
            job_id=job_id,
        ))

    await db.commit()
    await db.refresh(job)

    if settings.billing_enabled:
        message = (
            f"Job queued. ${held_cents / 100:.2f} reserved "
            f"(${estimate['estimated_cost']:.2f} estimate + 50% buffer). "
            "Unused credits are refunded automatically at completion."
        )
    else:
        message = "Job queued."

    return JobSubmitResponse(
        job_id=job.id,
        status=job.status,
        estimated_cost=job.estimated_cost,
        estimated_duration_minutes=job.estimated_duration_minutes,
        instance_type=job.instance_type,
        created_at=job.created_at,
        message=message,
    )


@router.get("", response_model=list[JobListItem])
async def list_jobs(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Job)
        .where(Job.session_id == user.id)
        .order_by(Job.created_at.desc())
    )
    return [
        JobListItem(
            job_id=j.id,
            status=j.status,
            kind=j.kind,
            config_name=j.config_name,
            instance_type=j.instance_type,
            created_at=j.created_at,
            started_at=j.started_at,
            completed_at=j.completed_at,
            costs=JobCosts(estimated=j.estimated_cost, actual=j.actual_cost),
            queue_position=j.queue_position,
        )
        for j in result.scalars().all()
    ]


@router.get("/{job_id}", response_model=JobDetail)
async def get_job(
    job_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job = await _get_job_or_404(db, job_id, user.id, is_admin=settings.is_admin(user.email))
    hourly_rate = None
    if job.instance_type:
        try:
            hourly_rate = (Decimal(str(get_rate(job.instance_type))) * cost_service.MARKUP).quantize(Decimal("0.001"))
        except KeyError:
            pass
    return JobDetail(
        job_id=job.id,
        status=job.status,
        config_name=job.config_name,
        config_yaml=job.config_yaml,
        instance_type=job.instance_type,
        dataset_size_mb=job.dataset_size_mb,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
        estimated_duration_minutes=job.estimated_duration_minutes,
        hourly_rate=hourly_rate,
        costs=JobCosts(estimated=job.estimated_cost, actual=job.actual_cost),
        queue_position=job.queue_position,
        provisioned_at=job.provisioned_at,
        checkpointed_at=job.checkpointed_at,
        last_checkpoint_step=job.last_checkpoint_step,
        last_checkpoint_s3_path=job.last_checkpoint_s3_path,
        resume_from_checkpoint_path=job.resume_from_checkpoint_path,
        kind=job.kind,
        source_job_id=job.source_job_id,
        export_quant=job.export_quant,
        modelfile=job.modelfile,
        evaluation=job.evaluation,
        exports=[
            ExportSummary(job_id=e.id, quant=e.export_quant, status=e.status)
            for e in await _exports_of(db, job.id)
        ] if job.kind == "train" else [],
    )


async def _exports_of(db: AsyncSession, job_id: uuid.UUID) -> list[Job]:
    result = await db.execute(
        select(Job).where(Job.source_job_id == job_id).order_by(Job.created_at)
    )
    return list(result.scalars().all())


# ── Export ────────────────────────────────────────────────────────────────────


def _require_exportable(job: Job) -> None:
    if job.kind != "train" or job.status != "completed" or not job.model_s3_path:
        raise HTTPException(
            status_code=409, detail="Only a completed training job's model can be exported"
        )


@router.get("/{job_id}/export-options", response_model=ExportOptionsResponse)
async def export_options(
    job_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Each GGUF format with its file size, the memory needed to run it, and
    what exporting to it costs."""
    job = await _get_job_or_404(db, job_id, user.id, is_admin=settings.is_admin(user.email))
    _require_exportable(job)
    cfg = yaml.safe_load(job.config_yaml) or {}
    params_b = parse_param_billions(str(cfg.get("base_model") or ""))
    estimate = cost_service.export_estimate(cfg)
    cost = Decimal("0") if settings.local_mode else estimate["estimated_cost"]
    return ExportOptionsResponse(
        options=[
            ExportOption(
                quant=quant,
                file_gb=round(gguf_size_gb(params_b, quant), 2),
                memory_gb=round(inference_memory_gb(params_b, quant), 2),
                estimated_cost=cost,
                estimated_minutes=estimate["estimated_minutes"],
            )
            for quant in GGUF_QUANTS
        ],
        recommended=recommend_quant(params_b, None),
    )


@router.post("/{job_id}/export", response_model=JobSubmitResponse, status_code=201)
async def start_export(
    job_id: uuid.UUID,
    req: ExportRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Queue an export of a finished training job's model to GGUF for Ollama.

    A job like any other — queued, billed and settled — that runs in the
    training image with JOB_KIND=export.
    """
    source = await _get_job_or_404(db, job_id, user.id, is_admin=settings.is_admin(user.email))
    _require_exportable(source)
    quant = req.quant.upper()
    if quant not in GGUF_QUANTS:
        raise HTTPException(
            status_code=422, detail=f"Unknown format {req.quant!r}; choose one of {', '.join(GGUF_QUANTS)}"
        )
    for existing in await _exports_of(db, source.id):
        if existing.export_quant == quant and existing.status not in ("failed", "cancelled"):
            raise HTTPException(
                status_code=409,
                detail=f"This model is already exported to {quant} — see job {existing.id}",
            )

    if not settings.local_mode:
        await _check_rate_limits(db, user.id)

    cfg = yaml.safe_load(source.config_yaml) or {}
    estimate = cost_service.export_estimate(cfg)
    estimated_cost_cents = int(float(estimate["estimated_cost"]) * 100)
    held_cents = math.ceil(estimated_cost_cents * float(BUFFER_FACTOR))
    new_id = uuid.uuid4()

    if settings.billing_enabled:
        result = await db.execute(select(User).where(User.id == user.id).with_for_update())
        db_user = result.scalar_one()
        if db_user.credit_balance_cents < held_cents:
            raise HTTPException(
                status_code=402,
                detail=(
                    f"Insufficient credits. Export estimated cost is ${estimate['estimated_cost']:.2f} — "
                    f"we hold ${held_cents / 100:.2f} (50% buffer). "
                    f"You have ${db_user.credit_balance_cents / 100:.2f}. Visit /credits to top up."
                ),
            )

    try:
        config_s3_path = s3_service.upload_bytes(
            source.config_yaml.encode(), f"configs/{new_id}/config.yaml", "application/x-yaml"
        )
    except (ClientError, BotoCoreError) as exc:
        raise HTTPException(status_code=503, detail="Storage service unavailable") from exc

    hf_token_secret_arn = None
    if req.hf_token and req.hf_token.strip():
        try:
            hf_token_secret_arn = secrets_service.store_hf_token(str(new_id), req.hf_token.strip())
        except (ClientError, BotoCoreError) as exc:
            raise HTTPException(status_code=503, detail="Failed to store credentials") from exc

    job = Job(
        id=new_id,
        session_id=user.id,
        status="pending",
        kind="export",
        source_job_id=source.id,
        export_quant=quant,
        config_yaml=source.config_yaml,
        config_name=source.config_name,
        config_s3_path=config_s3_path,
        model_s3_path=None,
        instance_type=estimate["instance_type"],
        disk_gb=estimate["disk_gb"],
        estimated_cost=Decimal(str(held_cents)) / 100,
        estimated_duration_minutes=estimate["estimated_minutes"],
        hf_token_secret_arn=hf_token_secret_arn,
    )
    db.add(job)
    await db.flush()
    if settings.billing_enabled:
        db_user.credit_balance_cents -= held_cents
        db.add(CreditTransaction(
            user_id=user.id,
            amount_cents=-held_cents,
            description=f"Reserved ${estimate['estimated_cost']:.2f} est. + 50% buffer — export {quant}",
            job_id=new_id,
        ))
    await db.commit()
    await db.refresh(job)

    return JobSubmitResponse(
        job_id=job.id,
        status=job.status,
        estimated_cost=job.estimated_cost,
        estimated_duration_minutes=job.estimated_duration_minutes,
        instance_type=job.instance_type,
        created_at=job.created_at,
        message=f"Export to {quant} queued.",
    )


@router.get("/{job_id}/logs")
async def stream_logs(
    job_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_job_or_404(db, job_id, user.id, is_admin=settings.is_admin(user.email))

    async def event_stream():
        from ..database import async_session_maker

        last_id = 0
        async with async_session_maker() as stream_db:
            while True:
                result = await stream_db.execute(
                    select(Log)
                    .where(Log.job_id == job_id, Log.id > last_id)
                    .order_by(Log.id)
                    .limit(100)
                )
                new_logs = result.scalars().all()

                for log in new_logs:
                    payload = json.dumps({"line": log.line, "ts": log.timestamp.isoformat() + 'Z'})
                    yield f"data: {payload}\n\n"
                    last_id = log.id

                status_result = await stream_db.execute(
                    select(Job.status).where(Job.id == job_id)
                )
                status = status_result.scalar_one()
                if status in ("completed", "failed", "cancelled") and not new_logs:
                    yield f"data: {json.dumps({'event': 'done', 'status': status})}\n\n"
                    break

                await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/{job_id}/metrics", response_model=list[MetricResponse])
async def get_metrics(
    job_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_job_or_404(db, job_id, user.id, is_admin=settings.is_admin(user.email))
    result = await db.execute(
        select(Metric).where(Metric.job_id == job_id).order_by(Metric.step)
    )
    return result.scalars().all()


@router.delete("/{job_id}", status_code=204)
async def cancel_job(
    job_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job = await _get_job_or_404(db, job_id, user.id, is_admin=settings.is_admin(user.email))
    if job.status not in ("pending", "provisioning", "running", "checkpointing"):
        raise HTTPException(status_code=409, detail=f"Cannot cancel a job in '{job.status}' state")

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    job.status = "cancelled"
    job.cancelled_at = now
    job.completed_at = now

    # What the job actually consumed, recorded whether or not credits were held,
    # so the job row can be reconciled against its ledger entries.
    if job.started_at is not None:
        elapsed_hours = (now - job.started_at).total_seconds() / 3600
        try:
            rate = get_rate(job.instance_type or "")
        except KeyError:
            rate = 0.0
        actual_cost_cents = int(elapsed_hours * rate * float(cost_service.MARKUP) * 100)
    else:
        actual_cost_cents = 0  # never provisioned, so nothing was spent
    job.actual_cost = Decimal(actual_cost_cents) / 100

    # Settle reservation: refund unused portion of what was reserved at submission
    estimated_cost_cents = int(float(job.estimated_cost or 0) * 100)
    db_user = None
    if estimated_cost_cents > 0:
        refund_cents = estimated_cost_cents - actual_cost_cents
        result = await db.execute(select(User).where(User.id == user.id).with_for_update())
        db_user = result.scalar_one()
        if refund_cents > 0:
            db_user.credit_balance_cents += refund_cents
            db.add(CreditTransaction(
                user_id=user.id,
                amount_cents=refund_cents,
                description="Refund — job cancelled",
                job_id=job.id,
            ))
        elif refund_cents < 0:
            overage = abs(refund_cents)
            db_user.credit_balance_cents -= overage
            db.add(CreditTransaction(
                user_id=user.id,
                amount_cents=-overage,
                description="Overage — job cancelled",
                job_id=job.id,
            ))

    await db.commit()

    # If the balance went negative, immediately charge the saved card for the difference
    if db_user is not None and db_user.credit_balance_cents < 0:
        overage_cents = abs(db_user.credit_balance_cents)
        if db_user.stripe_customer_id and db_user.stripe_payment_method_id:
            try:
                charge_overage(
                    db_user.stripe_customer_id,
                    db_user.stripe_payment_method_id,
                    overage_cents,
                    str(job.id),
                )
                db_user.credit_balance_cents = 0
                db.add(CreditTransaction(
                    user_id=user.id,
                    amount_cents=overage_cents,
                    description="Overage charge — balance recovered",
                    job_id=job.id,
                ))
                await db.commit()
            except Exception:
                logger.exception(
                    "Overage charge failed for user %s job %s — balance remains negative",
                    user.id, job.id,
                )


@router.post("/{job_id}/resume", response_model=JobSubmitResponse, status_code=201)
async def resume_job(
    job_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    source = await _get_job_or_404(db, job_id, user.id, is_admin=settings.is_admin(user.email))
    if source.status != "failed":
        raise HTTPException(status_code=409, detail="Can only resume a failed job")
    if not source.last_checkpoint_s3_path:
        raise HTTPException(status_code=409, detail="No checkpoint available to resume from")

    await _check_rate_limits(db, user.id)

    estimate = cost_service.parse_and_estimate(source.config_yaml, source.dataset_size_mb or 0)

    estimated_cost_cents = int(float(estimate["estimated_cost"]) * 100)
    held_cents = math.ceil(estimated_cost_cents * float(BUFFER_FACTOR))
    held_cost = Decimal(str(held_cents)) / 100

    new_job_id = uuid.uuid4()

    if settings.billing_enabled:
        result = await db.execute(select(User).where(User.id == user.id).with_for_update())
        db_user = result.scalar_one()
        if db_user.credit_balance_cents < held_cents:
            raise HTTPException(
                status_code=402,
                detail=(
                    f"Insufficient credits. Resume estimated cost is ${estimate['estimated_cost']:.2f} — "
                    f"we hold ${held_cents / 100:.2f} (50% buffer). "
                    f"You have ${db_user.credit_balance_cents / 100:.2f}. "
                    "Visit /credits to top up."
                ),
            )
        db_user.credit_balance_cents -= held_cents
        db.add(CreditTransaction(
            user_id=user.id,
            amount_cents=-held_cents,
            description=f"Reserved ${estimate['estimated_cost']:.2f} est. + 50% buffer — resume",
            job_id=new_job_id,
        ))

    config_key = f"configs/{new_job_id}/config.yaml"
    try:
        config_s3_path = s3_service.upload_bytes(
            source.config_yaml.encode(), config_key, "application/x-yaml"
        )
    except (ClientError, BotoCoreError) as exc:
        raise HTTPException(status_code=503, detail="Storage service unavailable") from exc

    job = Job(
        id=new_job_id,
        session_id=user.id,
        status="pending",
        config_yaml=source.config_yaml,
        config_name=source.config_name,
        config_s3_path=config_s3_path,
        dataset_s3_path=source.dataset_s3_path,
        dataset_size_mb=source.dataset_size_mb,
        instance_type=estimate["instance_type"],
        estimated_cost=held_cost,
        estimated_duration_minutes=estimate["estimated_duration_minutes"],
        hf_token_secret_arn=None,
        resume_from_checkpoint_path=source.last_checkpoint_s3_path,
        queue_position=0,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    return JobSubmitResponse(
        job_id=job.id,
        status=job.status,
        estimated_cost=job.estimated_cost,
        estimated_duration_minutes=job.estimated_duration_minutes,
        instance_type=job.instance_type,
        created_at=job.created_at,
        message="Resume job queued at front of queue",
    )


@router.get("/{job_id}/download", response_model=DownloadResponse)
async def get_download_url(
    job_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job = await _get_job_or_404(db, job_id, user.id, is_admin=settings.is_admin(user.email))
    if job.status != "completed":
        raise HTTPException(status_code=409, detail="Model is only available for completed jobs")
    if not job.model_s3_path:
        raise HTTPException(status_code=404, detail="Model file not found")
    key = s3_service.s3_key_from_path(job.model_s3_path)
    return DownloadResponse(url=s3_service.generate_presigned_url(key))
