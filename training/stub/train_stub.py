"""
Minimal training stub for local smoke testing.
Mimics the real training container's API contract without Axolotl or a GPU.
"""
import os
import time
import requests

JOB_ID = os.environ["JOB_ID"]
API_URL = os.environ["API_URL"].rstrip("/")
INTERNAL_TOKEN = os.environ.get("INTERNAL_TOKEN", "")

session = requests.Session()
session.headers.update({"Content-Type": "application/json", "X-Internal-Token": INTERNAL_TOKEN})

def api(method, path, **kwargs):
    url = f"{API_URL}/internal/jobs/{JOB_ID}/{path}"
    try:
        r = session.request(method, url, timeout=10, **kwargs)
        r.raise_for_status()
    except Exception as e:
        print(f"[stub] API call failed: {e}")

print(f"[stub] Starting stub training for job {JOB_ID}")
api("POST", "heartbeat", json={"instance_id": JOB_ID})

# Emit fake metrics over ~15 seconds
for step in range(1, 6):
    time.sleep(3)
    api("POST", "logs", json={"lines": [f"[stub] Step {step}/5 — loss: {1.0 - step * 0.15:.3f}"]})
    api("POST", "metrics", json={"metrics": [{
        "step": step,
        "epoch": round(step / 5, 2),
        "train_loss": round(1.0 - step * 0.15, 4),
        "learning_rate": 2e-4,
        "gpu_utilization": 0.0,
        "gpu_memory_used_mb": 0,
        "tokens_per_second": 0.0,
    }]})


def upload_placeholder_model() -> str:
    """Upload a tiny tarball where a real model would go.

    Without this the job completes with a `model_s3_path` pointing at nothing,
    and `heulistic download` correctly 404s — leaving the download half of the
    flow impossible to exercise locally.
    """
    import io
    import tarfile

    import boto3

    bucket = os.environ.get("S3_BUCKET", "heulistic-local")
    key = f"models/{JOB_ID}/model.tar.gz"

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        note = b"This is a stub artifact, not a trained model.\n"
        info = tarfile.TarInfo("model/README.txt")
        info.size = len(note)
        tar.addfile(info, io.BytesIO(note))
    body = buf.getvalue()

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT_URL", "http://minio:9000"),
        aws_access_key_id=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )
    s3.put_object(Bucket=bucket, Key=key, Body=body)
    return f"s3://{bucket}/{key}"


try:
    model_path = upload_placeholder_model()
    api("POST", "logs", json={"lines": [
        "[stub] Training complete (stub — placeholder artifact, not a real model)"
    ]})
except Exception as exc:  # the job should still settle if MinIO is unreachable
    print(f"[stub] Could not upload placeholder model: {exc}")
    model_path = f"s3://{os.environ.get('S3_BUCKET', 'heulistic-local')}/models/{JOB_ID}/model.tar.gz"
    api("POST", "logs", json={"lines": [f"[stub] Model upload failed: {exc}"]})

api("POST", "complete", json={"model_s3_path": model_path})
print("[stub] Done")
