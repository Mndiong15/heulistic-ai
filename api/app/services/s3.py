import boto3
from botocore.config import Config

from ..config import settings

_client = None


def _s3():
    global _client
    if _client is None:
        kwargs: dict = {
            "region_name": settings.aws_region,
            "config": Config(signature_version="s3v4"),
        }
        if settings.s3_endpoint_url:
            kwargs["endpoint_url"] = settings.s3_endpoint_url
            kwargs["aws_access_key_id"] = settings.minio_access_key
            kwargs["aws_secret_access_key"] = settings.minio_secret_key
        _client = boto3.client("s3", **kwargs)
    return _client


def upload_bytes(data: bytes, key: str, content_type: str = "application/octet-stream") -> str:
    _s3().put_object(
        Bucket=settings.s3_bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
    )
    return f"s3://{settings.s3_bucket}/{key}"


def upload_fileobj(file_obj, key: str) -> str:
    _s3().upload_fileobj(file_obj, settings.s3_bucket, key)
    return f"s3://{settings.s3_bucket}/{key}"


def generate_presigned_url(key: str, expires_in: int = 3600) -> str:
    if settings.local_mode:
        base = settings.api_public_url.rstrip("/")
        return f"{base}/local-download/{key}"
    return _s3().generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3_bucket, "Key": key},
        ExpiresIn=expires_in,
    )


def s3_key_from_path(s3_path: str) -> str:
    """Extract the key from an s3://bucket/key path."""
    return s3_path.split("/", 3)[-1]


# ── Direct uploads ────────────────────────────────────────────────────────────
# The browser sends dataset parts straight to S3; the API only signs the parts
# and reads back the first megabyte to validate it.


def user_dataset_prefix(user_id: str) -> str:
    return f"temp-datasets/{user_id}/"


def s3_path_for(key: str) -> str:
    return f"s3://{settings.s3_bucket}/{key}"


def create_multipart_upload(key: str) -> str:
    resp = _s3().create_multipart_upload(
        Bucket=settings.s3_bucket, Key=key, ContentType="application/x-ndjson"
    )
    return resp["UploadId"]


def presign_upload_part(key: str, upload_id: str, part_number: int, expires_in: int) -> str:
    return _s3().generate_presigned_url(
        "upload_part",
        Params={
            "Bucket": settings.s3_bucket,
            "Key": key,
            "UploadId": upload_id,
            "PartNumber": part_number,
        },
        ExpiresIn=expires_in,
    )


def complete_multipart_upload(key: str, upload_id: str, parts: list[dict]) -> None:
    _s3().complete_multipart_upload(
        Bucket=settings.s3_bucket,
        Key=key,
        UploadId=upload_id,
        MultipartUpload={
            "Parts": [{"PartNumber": p["part_number"], "ETag": p["etag"]} for p in parts]
        },
    )


def abort_multipart_upload(key: str, upload_id: str) -> None:
    _s3().abort_multipart_upload(Bucket=settings.s3_bucket, Key=key, UploadId=upload_id)


def object_size(key: str) -> int | None:
    """Size of an object in bytes, or None if there is no such object."""
    from botocore.exceptions import ClientError

    try:
        return _s3().head_object(Bucket=settings.s3_bucket, Key=key)["ContentLength"]
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def read_head(key: str, nbytes: int) -> bytes:
    """The first `nbytes` of an object, without fetching the rest."""
    resp = _s3().get_object(Bucket=settings.s3_bucket, Key=key, Range=f"bytes=0-{nbytes - 1}")
    return resp["Body"].read()
