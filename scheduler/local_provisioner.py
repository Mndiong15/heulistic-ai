"""
Provisions training jobs as local Docker containers instead of EC2 instances.

Mirrors the interface of provisioner.py:
  provision(job) -> str   # returns job_id used as instance_id
  terminate(id)  -> None  # stops container found via heulistic.job_id label
"""

import hashlib
import logging
import os
import secrets

import docker
import docker.errors
import docker.types

from dispatch import CapacityUnavailable

logger = logging.getLogger(__name__)

_LABEL_KEY = "heulistic.job_id"
# Comma-separated GPU indices a container was pinned to, so the next job can
# see which cards are taken.
_GPU_LABEL_KEY = "heulistic.gpus"


class NoFreeGpus(CapacityUnavailable):
    pass


def _gpu_request(client: docker.DockerClient) -> tuple[list, str | None]:
    """The device request for a new job, and the GPU label to put on it.

    With GPUS_PER_JOB unset or 0 the job gets every GPU — the fastest choice
    for one job at a time. Otherwise it is pinned to that many cards that no
    running job holds.
    """
    per_job = int(os.environ.get("GPUS_PER_JOB") or 0)
    if per_job <= 0:
        return [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])], None

    total = int(os.environ.get("LOCAL_GPU_COUNT") or 0)
    taken: set[str] = set()
    for container in client.containers.list(filters={"label": _LABEL_KEY}):
        held = container.labels.get(_GPU_LABEL_KEY) or ""
        taken.update(g for g in held.split(",") if g)
    free = [str(i) for i in range(total) if str(i) not in taken]
    if len(free) < per_job:
        raise NoFreeGpus(f"{len(free)} of {total} GPUs free, {per_job} needed")

    chosen = free[:per_job]
    request = docker.types.DeviceRequest(device_ids=chosen, capabilities=[["gpu"]])
    return [request], ",".join(chosen)


def _client() -> docker.DockerClient:
    socket = os.environ.get("DOCKER_SOCKET", "unix://var/run/docker.sock")
    return docker.DockerClient(base_url=socket)


def provision(job: dict) -> tuple[str, str]:
    """
    Run the training image as a sibling Docker container.
    Returns (job_id, token_hash) — job_id is used as the instance_id to avoid
    a chicken-and-egg problem (container ID is unknown before docker run completes).
    """
    job_id = str(job["id"])
    token = secrets.token_hex(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    image = os.environ["LOCAL_TRAINING_IMAGE"]
    network = os.environ.get("DOCKER_NETWORK", "heulistic_default")
    cache_volume = os.environ.get("LOCAL_CACHE_VOLUME", "heulistic-cache")
    use_gpu = os.environ.get("LOCAL_GPU", "true").lower() == "true"

    env = {
        "JOB_ID": job_id,
        "API_URL": os.environ.get("API_INTERNAL_URL", "http://api:8000"),
        "INTERNAL_TOKEN": token,
        "S3_BUCKET": os.environ.get("S3_BUCKET", "heulistic-local"),
        "S3_ENDPOINT_URL": os.environ.get("S3_ENDPOINT_URL", "http://minio:9000"),
        "MINIO_ACCESS_KEY": os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        "MINIO_SECRET_KEY": os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        "CONFIG_S3_PATH": job.get("config_s3_path") or "",
        "DATASET_S3_PATH": job.get("dataset_s3_path") or "",
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "RESUME_FROM_CHECKPOINT_PATH": job.get("resume_from_checkpoint_path") or "",
        "INSTANCE_ID": job_id,
        "INSTANCE_TYPE": "local-gpu",
        "AWS_REGION": os.environ.get("AWS_REGION", "us-east-1"),
        "JOB_KIND": job.get("kind") or "train",
        "SOURCE_MODEL_S3_PATH": job.get("source_model_s3_path") or "",
        "EXPORT_QUANT": job.get("export_quant") or "",
    }

    client = _client()
    labels = {_LABEL_KEY: job_id}
    device_requests: list = []
    # An export merges and quantizes on the CPU; holding GPUs would only
    # keep training jobs waiting.
    if use_gpu and job.get("kind") != "export":
        device_requests, gpu_label = _gpu_request(client)
        if gpu_label:
            labels[_GPU_LABEL_KEY] = gpu_label
    container = client.containers.run(
        image=image,
        detach=True,
        remove=True,
        network=network,
        volumes={cache_volume: {"bind": "/efs", "mode": "rw"}},
        labels=labels,
        device_requests=device_requests,
        environment=env,
    )

    logger.info("Job %s: started local container %s", job_id, container.short_id)
    return job_id, token_hash


def terminate(instance_id: str) -> None:
    """Stop the local container for the given job_id (stored as instance_id)."""
    client = _client()
    containers = client.containers.list(filters={"label": f"{_LABEL_KEY}={instance_id}"})
    if not containers:
        logger.info("No running container found for job %s — already stopped", instance_id)
        return
    for container in containers:
        try:
            container.stop(timeout=10)
            logger.info("Stopped local container %s for job %s", container.short_id, instance_id)
        except docker.errors.APIError as exc:
            logger.warning("Could not stop container %s: %s", container.short_id, exc)
