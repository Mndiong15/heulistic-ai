from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..services.s3 import _s3
from ..config import settings

router = APIRouter(tags=["local"])


@router.get("/local-download/{path:path}")
async def local_download(path: str):
    """Serve files from MinIO — replaces presigned S3 URLs in local mode."""
    try:
        obj = _s3().get_object(Bucket=settings.s3_bucket, Key=path)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc
    filename = path.split("/")[-1]
    return StreamingResponse(
        obj["Body"],
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
