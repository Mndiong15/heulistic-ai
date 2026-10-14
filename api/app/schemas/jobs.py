import uuid
from datetime import datetime
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, PlainSerializer

# Naive UTC datetimes serialized with explicit Z so JS always parses them as UTC.
UTCDatetime = Annotated[datetime, PlainSerializer(lambda dt: dt.isoformat() + 'Z', return_type=str)]


class JobCosts(BaseModel):
    estimated: Decimal | None
    actual: Decimal | None


class JobSubmitResponse(BaseModel):
    job_id: uuid.UUID
    status: str
    estimated_cost: Decimal | None
    estimated_duration_minutes: int | None
    instance_type: str | None
    created_at: UTCDatetime
    message: str


class JobListItem(BaseModel):
    job_id: uuid.UUID
    status: str
    kind: str = "train"
    config_name: str | None
    instance_type: str | None
    created_at: UTCDatetime
    started_at: UTCDatetime | None
    completed_at: UTCDatetime | None
    costs: JobCosts
    queue_position: int | None = None


class JobDetail(JobListItem):
    config_yaml: str
    dataset_size_mb: int | None
    error_message: str | None
    estimated_duration_minutes: int | None
    hourly_rate: Decimal | None = None
    provisioned_at: UTCDatetime | None = None
    checkpointed_at: UTCDatetime | None = None
    last_checkpoint_step: int | None = None
    last_checkpoint_s3_path: str | None = None
    resume_from_checkpoint_path: str | None = None
    source_job_id: uuid.UUID | None = None
    export_quant: str | None = None
    modelfile: str | None = None
    evaluation: dict | None = None
    exports: list["ExportSummary"] = []


class ExportSummary(BaseModel):
    job_id: uuid.UUID
    quant: str | None
    status: str


class ExportRequest(BaseModel):
    quant: str = "Q4_K_M"
    hf_token: str | None = None


class ExportOption(BaseModel):
    quant: str
    file_gb: float
    memory_gb: float
    estimated_cost: Decimal
    estimated_minutes: int


class ExportOptionsResponse(BaseModel):
    options: list[ExportOption]
    recommended: str


class EvaluationRequest(BaseModel):
    report: dict


class MetricResponse(BaseModel):
    step: int
    epoch: float | None
    train_loss: float | None
    learning_rate: float | None
    gpu_utilization: float | None
    gpu_memory_used_mb: int | None
    tokens_per_second: float | None
    timestamp: UTCDatetime

    model_config = {"from_attributes": True}


class DownloadResponse(BaseModel):
    url: str
    expires_in_seconds: int = 3600


# ── /api/estimate ─────────────────────────────────────────────────────────────

class EstimateRequest(BaseModel):
    config_yaml: str
    dataset_size_mb: int = 0
    instance_type: str | None = None


class EstimateResponse(BaseModel):
    estimated_cost: Decimal
    estimated_duration_minutes: int
    instance_type: str
    hourly_rate: Decimal
    dataset_known: bool = True


# ── Internal: training instance → API ────────────────────────────────────────

class HeartbeatRequest(BaseModel):
    instance_id: str


class MetricBatch(BaseModel):
    step: int
    epoch: float | None = None
    train_loss: float | None = None
    learning_rate: float | None = None
    gpu_utilization: float | None = None
    gpu_memory_used_mb: int | None = None
    tokens_per_second: float | None = None


class MetricsRequest(BaseModel):
    metrics: list[MetricBatch]


class LogsBatch(BaseModel):
    lines: list[str]


class CompleteRequest(BaseModel):
    model_s3_path: str | None = None
    actual_cost: float | None = None
    error_message: str | None = None
    modelfile: str | None = None


class CheckpointRequest(BaseModel):
    checkpoint_s3_path: str
    step: int


class ValidateRequest(BaseModel):
    config_yaml: str
    # When given, the config is also checked against that instance's VRAM.
    instance_type: str | None = None


class DiagnosticResponse(BaseModel):
    level: str
    code: str
    message: str
    field: str | None = None
    line: int | None = None


class ValidateResponse(BaseModel):
    ok: bool
    counts: dict[str, int]
    diagnostics: list[DiagnosticResponse]


JobDetail.model_rebuild()
