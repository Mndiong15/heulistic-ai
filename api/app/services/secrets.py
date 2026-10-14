import boto3
from botocore.exceptions import BotoCoreError, ClientError

from ..config import settings

_client = None


def _sm():
    global _client
    if _client is None:
        _client = boto3.client("secretsmanager", region_name=settings.aws_region)
    return _client


def store_hf_token(job_id: str, token: str) -> str:
    """Write HF token to Secrets Manager. Returns the secret ARN."""
    if settings.local_mode:
        return f"arn:aws:secretsmanager:local:000000000000:secret:ftp/jobs/{job_id}/hf_token"
    response = _sm().create_secret(
        Name=f"ftp/jobs/{job_id}/hf_token",
        SecretString=token,
        Description=f"HuggingFace token for fine-tuning job {job_id}",
    )
    return response["ARN"]


def delete_secret(arn: str) -> None:
    """Permanently delete a secret. Best-effort — never raises."""
    if settings.local_mode:
        return
    try:
        _sm().delete_secret(SecretId=arn, ForceDeleteWithoutRecovery=True)
    except (ClientError, BotoCoreError):
        pass
