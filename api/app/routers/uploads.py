"""Direct-to-S3 dataset uploads.

The browser asks for a multipart upload, PUTs each part to its presigned URL,
then completes it here and submits the job with the resulting key. The file
never passes through the API, so its size is bounded by the training
instance's disk rather than the API server's memory.

Cloud only: in local mode the presigned URLs would name MinIO's in-container
hostname, which the browser cannot reach, so uploads stream through the API.
"""

import asyncio
import math
import uuid

from fastapi import APIRouter, Depends, HTTPException, Response

from ..config import settings
from ..dependencies import get_current_user
from ..models.db import User
from ..schemas.uploads import (
    AbortUploadRequest,
    CompleteUploadRequest,
    CompleteUploadResponse,
    CreateUploadRequest,
    CreateUploadResponse,
)
from ..services import s3 as s3_service

router = APIRouter(prefix="/uploads", tags=["uploads"])

_MB = 1024 * 1024
# 64 MB parts: large enough that a 50 GB file needs ~800 of them (S3 allows
# 10,000), small enough that a failed part is cheap to retry.
_MIN_PART_SIZE = 64 * _MB
_MAX_PARTS = 10_000
# Long enough for a big file on a slow link. URLs signed with the instance
# role's temporary credentials can lapse sooner when those rotate.
_PART_URL_TTL_SECONDS = 6 * 3600


def _part_size(size_bytes: int) -> int:
    needed = math.ceil(size_bytes / _MAX_PARTS)
    return max(_MIN_PART_SIZE, math.ceil(needed / _MB) * _MB)


def require_own_dataset_key(key: str, user_id: str) -> None:
    """403 unless `key` lies under this user's dataset prefix."""
    prefix = s3_service.user_dataset_prefix(user_id)
    if not key.startswith(prefix) or ".." in key.split("/"):
        raise HTTPException(status_code=403, detail="That upload belongs to another account")


def _cloud_only() -> None:
    if settings.local_mode:
        raise HTTPException(
            status_code=404,
            detail="Direct uploads are not available in local mode — attach the dataset to the job instead",
        )


@router.post("", response_model=CreateUploadResponse, status_code=201)
async def create_upload(req: CreateUploadRequest, user: User = Depends(get_current_user)):
    _cloud_only()
    if req.size_bytes > settings.max_dataset_size_mb * _MB:
        raise HTTPException(
            status_code=413,
            detail=f"Dataset exceeds the {settings.max_dataset_size_mb // 1024} GB limit",
        )

    key = f"{s3_service.user_dataset_prefix(user.id)}{uuid.uuid4().hex}/data.jsonl"
    part_size = _part_size(req.size_bytes)
    upload_id = await asyncio.to_thread(s3_service.create_multipart_upload, key)
    part_urls = [
        s3_service.presign_upload_part(key, upload_id, n, _PART_URL_TTL_SECONDS)
        for n in range(1, math.ceil(req.size_bytes / part_size) + 1)
    ]
    return CreateUploadResponse(
        key=key, upload_id=upload_id, part_size=part_size, part_urls=part_urls
    )


@router.post("/complete", response_model=CompleteUploadResponse)
async def complete_upload(req: CompleteUploadRequest, user: User = Depends(get_current_user)):
    _cloud_only()
    require_own_dataset_key(req.key, user.id)
    parts = sorted((p.model_dump() for p in req.parts), key=lambda p: p["part_number"])
    await asyncio.to_thread(s3_service.complete_multipart_upload, req.key, req.upload_id, parts)
    return CompleteUploadResponse(key=req.key)


@router.post("/abort", status_code=204)
async def abort_upload(req: AbortUploadRequest, user: User = Depends(get_current_user)):
    _cloud_only()
    require_own_dataset_key(req.key, user.id)
    await asyncio.to_thread(s3_service.abort_multipart_upload, req.key, req.upload_id)
    return Response(status_code=204)
