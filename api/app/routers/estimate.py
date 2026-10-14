import yaml

import httpx
from fastapi import APIRouter, HTTPException

from ..schemas.jobs import EstimateRequest, EstimateResponse
from ..services import cost as cost_service
from ..services.instances import valid_instance_types

router = APIRouter(tags=["estimate"])

_HF_SIZE_URL = "https://datasets-server.huggingface.co/size?dataset={}"


def _is_hf_path(path: str) -> bool:
    """True if path looks like a HuggingFace Hub dataset (not S3/HTTP/local)."""
    if not path:
        return False
    return not path.startswith(("s3://", "http://", "https://", "./", "/", "~"))


async def _hf_train_rows(dataset_path: str) -> int | None:
    """Return training row count for a HuggingFace dataset, or None on any failure."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(_HF_SIZE_URL.format(dataset_path))
        if resp.status_code != 200:
            return None
        data = resp.json()
        train_splits = [
            s for s in data.get("size", {}).get("splits", [])
            if s.get("split") == "train"
        ]
        if train_splits:
            return sum(s.get("num_rows", 0) for s in train_splits)
        # Fall back to total dataset rows if no train split is listed
        total = data.get("size", {}).get("dataset", {}).get("num_rows")
        return int(total) if total else None
    except Exception:
        return None


@router.post("/estimate", response_model=EstimateResponse)
async def estimate(req: EstimateRequest):
    valid = valid_instance_types()
    if req.instance_type and req.instance_type not in valid:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid instance_type '{req.instance_type}'. "
                   f"Valid options: {sorted(valid)}",
        )

    try:
        parsed = yaml.safe_load(req.config_yaml)
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid YAML: {exc}")
    if not isinstance(parsed, dict) or "base_model" not in parsed:
        raise HTTPException(status_code=422, detail="Config must include 'base_model'")

    # When no file was uploaded (dataset_size_mb=0) and config references HF Hub datasets,
    # fetch exact training row counts so the estimate uses real sample counts.
    num_train_samples: int | None = None
    dataset_known = req.dataset_size_mb > 0  # known when a file was uploaded
    if req.dataset_size_mb == 0:
        datasets = parsed.get("datasets") or []
        for ds in datasets:
            if isinstance(ds, dict):
                path = ds.get("path", "")
                if _is_hf_path(path):
                    rows = await _hf_train_rows(path)
                    if rows:
                        num_train_samples = (num_train_samples or 0) + rows
                        dataset_known = True

    return EstimateResponse(
        **cost_service.parse_and_estimate(
            req.config_yaml,
            req.dataset_size_mb,
            req.instance_type,
            num_train_samples=num_train_samples,
        ),
        dataset_known=dataset_known,
    )
