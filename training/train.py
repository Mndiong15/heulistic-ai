"""
Training container entrypoint.

Env vars injected by provisioner user-data script:
  JOB_ID, API_URL, S3_BUCKET, CONFIG_S3_PATH, DATASET_S3_PATH,
  HF_TOKEN_SECRET_ARN, RESUME_FROM_CHECKPOINT_PATH,
  INSTANCE_TYPE, AWS_REGION, INSTANCE_ID, INTERNAL_TOKEN
"""

import json
import logging
import os
import subprocess
import sys
import tarfile
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

import boto3
import requests
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("trainer")

# ── Config from environment ───────────────────────────────────────────────────

JOB_ID = os.environ["JOB_ID"]
API_URL = os.environ["API_URL"].rstrip("/")
S3_BUCKET = os.environ["S3_BUCKET"]
CONFIG_S3_PATH = os.environ["CONFIG_S3_PATH"]
DATASET_S3_PATH = os.environ.get("DATASET_S3_PATH", "")
HF_TOKEN_SECRET_ARN = os.environ.get("HF_TOKEN_SECRET_ARN", "")
RESUME_FROM_CHECKPOINT_PATH = os.environ.get("RESUME_FROM_CHECKPOINT_PATH", "")
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "g5.xlarge")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
INSTANCE_ID = os.environ.get("INSTANCE_ID", "local")
INTERNAL_TOKEN = os.environ.get("INTERNAL_TOKEN", "")
# "train", or "export" to turn a finished job's model into a GGUF for Ollama.
JOB_KIND = os.environ.get("JOB_KIND", "train")
SOURCE_MODEL_S3_PATH = os.environ.get("SOURCE_MODEL_S3_PATH", "")
EXPORT_QUANT = os.environ.get("EXPORT_QUANT", "Q4_K_M")

_INSTANCE_RATES = {
    # g4dn — T4 16 GB
    "g4dn.xlarge":   0.526,
    "g4dn.2xlarge":  0.752,
    "g4dn.4xlarge":  1.204,
    "g4dn.8xlarge":  2.264,
    "g4dn.16xlarge": 4.528,
    "g4dn.12xlarge": 3.912,
    # g5 — A10G 22 GB
    "g5.xlarge":     1.006,
    "g5.2xlarge":    1.212,
    "g5.4xlarge":    1.624,
    "g5.8xlarge":    2.248,
    "g5.16xlarge":   4.096,
    "g5.12xlarge":   5.672,
    # g6 — L4 22 GB
    "g6.xlarge":     0.805,
    "g6.2xlarge":    0.978,
    "g6.4xlarge":    1.323,
    "g6.8xlarge":    2.013,
    "g6.16xlarge":   3.394,
    "g6.12xlarge":   4.602,
    # g6e — L40S 44 GB
    "g6e.xlarge":    1.861,
    "g6e.2xlarge":   2.373,
    "g6e.4xlarge":   3.397,
    "g6e.8xlarge":   5.445,
    "g6e.16xlarge":  9.541,
    "g6e.12xlarge":  13.556,
}

EFS_MODELS = Path("/efs/models")
WORK_DIR = Path("/tmp/training")

# Axolotl ships its DeepSpeed configs inside the base image, not beside train.py.
AXOLOTL_DEEPSPEED_DIR = Path(
    os.environ.get("AXOLOTL_DEEPSPEED_DIR", "/workspace/axolotl/deepspeed_configs")
)
# zero3.json rather than zero3_bf16.json: the bf16 variant hard-enables bf16,
# which T4s (the 4-GPU g4dn boxes) do not support. zero3.json leaves precision
# on "auto", so it follows whatever the Axolotl config asks for.
_AUTO_SHARD_CONFIG = "zero3.json"

_s3_kwargs: dict = {"region_name": AWS_REGION}
if _s3_endpoint := os.environ.get("S3_ENDPOINT_URL"):
    # Local mode — MinIO. Prefer explicit creds (e.g. scoped STS) over admin fallback.
    _s3_kwargs["endpoint_url"] = _s3_endpoint
    _s3_kwargs["aws_access_key_id"] = (
        os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
    )
    _s3_kwargs["aws_secret_access_key"] = (
        os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("MINIO_SECRET_KEY", "minioadmin")
    )
    if _session_token := os.environ.get("AWS_SESSION_TOKEN"):
        _s3_kwargs["aws_session_token"] = _session_token
elif os.environ.get("AWS_ACCESS_KEY_ID"):
    # Cloud mode with explicit STS credentials (TRAINING_ROLE_ARN set on scheduler).
    # boto3 would pick up the instance profile automatically when these aren't set,
    # so we only override when the scheduler has explicitly passed scoped creds.
    _s3_kwargs["aws_access_key_id"] = os.environ["AWS_ACCESS_KEY_ID"]
    _s3_kwargs["aws_secret_access_key"] = os.environ["AWS_SECRET_ACCESS_KEY"]
    if _session_token := os.environ.get("AWS_SESSION_TOKEN"):
        _s3_kwargs["aws_session_token"] = _session_token
s3 = boto3.client("s3", **_s3_kwargs)

# ── API helpers ───────────────────────────────────────────────────────────────

_SESSION = requests.Session()
_SESSION.headers.update({"Content-Type": "application/json", "X-Internal-Token": INTERNAL_TOKEN})


def _api(method: str, path: str, **kwargs) -> Optional[requests.Response]:
    url = f"{API_URL}/internal/jobs/{JOB_ID}/{path}"
    try:
        resp = _SESSION.request(method, url, timeout=10, **kwargs)
        resp.raise_for_status()
        return resp
    except Exception as exc:
        logger.warning("API call %s %s failed: %s", method, path, exc)
        return None


def heartbeat_loop(stop_event: threading.Event) -> None:
    while not stop_event.wait(60):
        try:
            resp = _SESSION.request(
                "POST",
                f"{API_URL}/internal/jobs/{JOB_ID}/heartbeat",
                json={"instance_id": INSTANCE_ID},
                timeout=10,
            )
            if resp.status_code == 409:
                logger.warning("Job %s is terminal — shutting down", JOB_ID)
                os._exit(1)
        except Exception as exc:
            logger.warning("Heartbeat failed: %s", exc)


def post_logs(lines: list[str]) -> None:
    if lines:
        _api("POST", "logs", json={"lines": lines})


def post_metrics(metrics: list[dict]) -> None:
    if metrics:
        _api("POST", "metrics", json={"metrics": metrics})


def record_checkpoint(checkpoint_s3_path: str, step: int) -> None:
    _api("PATCH", "checkpoint", json={"checkpoint_s3_path": checkpoint_s3_path, "step": step})


def complete_job(
    model_s3_path: str = "",
    error_message: str = "",
    actual_cost: float = 0.0,
    modelfile: str = "",
) -> None:
    payload: dict = {}
    if error_message:
        payload["error_message"] = error_message
    else:
        payload["model_s3_path"] = model_s3_path
        if actual_cost:
            payload["actual_cost"] = actual_cost
        if modelfile:
            payload["modelfile"] = modelfile
    _api("POST", "complete", json=payload)


# ── HuggingFace token ─────────────────────────────────────────────────────────

def fetch_and_apply_hf_token() -> None:
    """Fetch HF token from Secrets Manager and set env vars for the HF hub."""
    if token := os.environ.get("HF_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        logger.info("HF token loaded from environment")
        return
    if not HF_TOKEN_SECRET_ARN:
        return
    try:
        sm = boto3.client("secretsmanager", region_name=AWS_REGION)
        resp = sm.get_secret_value(SecretId=HF_TOKEN_SECRET_ARN)
        token = resp["SecretString"]
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        os.environ["HF_TOKEN"] = token
        logger.info("HF token loaded from Secrets Manager")
    except Exception as exc:
        logger.warning("Could not fetch HF token: %s", exc)


# ── S3 / URL helpers ──────────────────────────────────────────────────────────

def _s3_key(s3_path: str) -> tuple[str, str]:
    if s3_path.startswith("s3://"):
        parts = s3_path[5:].split("/", 1)
        return parts[0], parts[1]
    return S3_BUCKET, s3_path


def download_from_s3(s3_path: str, local_path: Path) -> None:
    bucket, key = _s3_key(s3_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading s3://%s/%s → %s", bucket, key, local_path)
    s3.download_file(bucket, key, str(local_path))


def download_from_url(url: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s", url, local_path)
    subprocess.run(
        ["wget", "-q", "-O", str(local_path), url],
        timeout=3600,
        check=True,
    )


def ensure_dataset(dataset_path: str) -> Optional[Path]:
    """Download the job's dataset to the instance's own disk.

    Not to EFS. Datasets used to be copied there under a content hash, but the
    file had to be downloaded in full to compute that hash, so the "cache"
    never saved a download — and EFS billed every copy, forever. The instance
    disk is already paid for by the job and is deleted with it.
    """
    if not dataset_path:
        return None

    dest = WORK_DIR / "dataset" / "data.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dataset_path.startswith("http://") or dataset_path.startswith("https://"):
        download_from_url(dataset_path, dest)
    else:
        download_from_s3(dataset_path, dest)
    return dest


# ── Checkpoint management ─────────────────────────────────────────────────────

_MAX_CHECKPOINTS = 3


def upload_checkpoint(local_checkpoint_dir: Path, step: int) -> str:
    """Tar and upload a checkpoint to S3, prune old ones. Returns s3:// path."""
    tarball = WORK_DIR / f"checkpoint-step{step}.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(local_checkpoint_dir, arcname=f"checkpoint-step{step}")

    key = f"checkpoints/{JOB_ID}/step-{step}/checkpoint.tar.gz"
    s3.upload_file(str(tarball), S3_BUCKET, key)
    tarball.unlink(missing_ok=True)

    path = f"s3://{S3_BUCKET}/{key}"
    logger.info("Checkpoint uploaded: %s", path)
    _prune_old_checkpoints(step)
    return path


def _prune_old_checkpoints(current_step: int) -> None:
    prefix = f"checkpoints/{JOB_ID}/"
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix, Delimiter="/")
    step_dirs = [
        cp["Prefix"]
        for cp in resp.get("CommonPrefixes", [])
        if cp["Prefix"] != f"{prefix}step-{current_step}/"
    ]

    def _step_num(p: str) -> int:
        try:
            return int(p.rstrip("/").rsplit("-", 1)[-1])
        except ValueError:
            return 0

    step_dirs.sort(key=_step_num, reverse=True)
    to_delete = step_dirs[_MAX_CHECKPOINTS - 1:]

    for prefix_to_delete in to_delete:
        keys_resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix_to_delete)
        keys = [obj["Key"] for obj in keys_resp.get("Contents", [])]
        if keys:
            s3.delete_objects(
                Bucket=S3_BUCKET,
                Delete={"Objects": [{"Key": k} for k in keys]},
            )
            logger.info("Pruned old checkpoint: %s", prefix_to_delete)


def download_checkpoint_for_resume(checkpoint_s3_path: str) -> Path:
    """Download and untar a checkpoint from S3. Returns the local checkpoint dir."""
    bucket, key = _s3_key(checkpoint_s3_path)
    tarball = WORK_DIR / "resume_checkpoint.tar.gz"
    local_dir = WORK_DIR / "resume_checkpoint"

    logger.info("Downloading resume checkpoint from %s", checkpoint_s3_path)
    s3.download_file(bucket, key, str(tarball))

    local_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(local_dir)
    tarball.unlink(missing_ok=True)

    subdirs = list(local_dir.iterdir())
    if len(subdirs) == 1 and subdirs[0].is_dir():
        return subdirs[0]
    return local_dir


# ── Model upload ──────────────────────────────────────────────────────────────

def upload_model(output_dir: Path) -> str:
    """Tar and upload final model to models/ prefix. Returns s3:// path."""
    tarball = WORK_DIR / "model.tar.gz"
    logger.info("Archiving model from %s → %s", output_dir, tarball)
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(output_dir, arcname="model")

    key = f"models/{JOB_ID}/model.tar.gz"
    logger.info("Uploading model to s3://%s/%s", S3_BUCKET, key)
    s3.upload_file(str(tarball), S3_BUCKET, key)
    tarball.unlink(missing_ok=True)
    return f"s3://{S3_BUCKET}/{key}"


# ── Axolotl config patching ───────────────────────────────────────────────────

def _resolve_deepspeed_path(value: str) -> str:
    """Point a relative DeepSpeed path at Axolotl's shipped configs.

    Axolotl's own examples write `deepspeed: deepspeed_configs/zero2.json`,
    relative to a checkout of the Axolotl repo. Here the working directory is
    /app, so that path did not resolve and the run died at launch. Anything
    absolute, or that resolves as given, is left alone.
    """
    path = Path(value)
    if path.is_absolute() or path.exists():
        return value
    shipped = AXOLOTL_DEEPSPEED_DIR / path.name
    return str(shipped) if shipped.exists() else value


def _gpu_inventory() -> tuple[int, float]:
    """(GPU count, VRAM per GPU in GB). Without NVML, (1, 0.0).

    Sized by the *smallest* card: under data parallelism every rank holds its
    own share, so a mixed rig is only as roomy as its tightest GPU.
    """
    if not _nvml_ok:
        return 1, 0.0
    try:
        count = pynvml.nvmlDeviceGetCount()
        totals = [
            pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).total
            for i in range(count)
        ]
        return count, round(min(totals) / 1024**3, 2)
    except Exception:
        return 1, 0.0


def plan_gpus(cfg: dict, gpu_count: int, per_gpu_vram_gb: float):
    """Decide how to spread training over the GPUs; shard when one is too small.

    Uses heulistic-axolotl's planner — the same one the API used to choose this
    instance and the validator used to check the config, so what was promised
    is what runs. Mutates `cfg` to add a ZeRO-3 config when sharding is needed
    and the user configured none. Returns the plan, or None on a single GPU or
    when VRAM is unknown.
    """
    if gpu_count < 2 or per_gpu_vram_gb <= 0:
        return None
    from heulistic_axolotl import Hardware, plan_parallelism

    plan = plan_parallelism(
        cfg, Hardware(total_vram_gb=per_gpu_vram_gb * gpu_count, gpu_count=gpu_count)
    )
    if plan.auto_sharded:
        cfg["deepspeed"] = str(AXOLOTL_DEEPSPEED_DIR / _AUTO_SHARD_CONFIG)
    return plan


def _describe_plan(plan, gpu_count: int, per_gpu_vram_gb: float) -> str:
    if not plan.fits:
        return (
            f"[trainer] Warning: estimated ~{plan.per_gpu_gb:.0f} GB per GPU exceeds "
            f"the {per_gpu_vram_gb:g} GB available — this may run out of memory "
            f"({plan.reason})"
        )
    if plan.auto_sharded:
        return (
            f"[trainer] Too large for one {per_gpu_vram_gb:g} GB GPU — sharding with "
            f"DeepSpeed ZeRO-3 across {gpu_count} GPUs (~{plan.per_gpu_gb:.0f} GB each)"
        )
    if plan.strategy == "ddp":
        return (
            f"[trainer] DDP across {gpu_count} GPUs, a full model copy on each "
            f"(~{plan.per_gpu_gb:.0f} of {per_gpu_vram_gb:g} GB per GPU)"
        )
    return f"[trainer] Using your {plan.strategy} config across {gpu_count} GPUs"


def apply_gpu_plan(config_path: Path, gpu_count: int, per_gpu_vram_gb: float):
    """Plan against the patched config on disk, rewriting it if sharding is added."""
    cfg = yaml.safe_load(config_path.read_text())
    plan = plan_gpus(cfg, gpu_count, per_gpu_vram_gb)
    if plan is not None and plan.auto_sharded:
        config_path.write_text(yaml.dump(cfg))
    return plan


def _tokens_per_log(cfg: dict, gpu_count: int) -> int:
    """Tokens processed between two consecutive loss lines in the log.

    One loss line covers `logging_steps` optimiser steps, each of which sees
    sequence_len x micro_batch x grad_accum tokens on every GPU. Approximate —
    padding counts as tokens when sample packing is off — but replaces a fixed
    512 that ignored all of these.
    """
    return (
        int(cfg.get("sequence_len") or 512)
        * int(cfg.get("micro_batch_size") or 1)
        * int(cfg.get("gradient_accumulation_steps") or 1)
        * int(cfg.get("logging_steps") or 1)
        * max(1, gpu_count)
    )


# ── Held-out split for the before/after evaluation ───────────────────────────

_MIN_ROWS_TO_HOLD_OUT = 100
_MAX_HELD_OUT = 50


def plan_eval_split(cfg: dict, rows: Optional[int]) -> Optional[int]:
    """How many rows to hold out to compare the fine-tuned model with the base.

    5% of the dataset, at most 50 rows, and nothing below 100 rows — a small
    dataset needs every example for training. The user's own choice always
    wins: any val_set_size, including 0 to switch this off, or test_datasets.
    The rows are held out through Axolotl's own seeded split, so evaluation
    later formats them exactly as training did.
    """
    if "val_set_size" in cfg or cfg.get("test_datasets") or cfg.get("rl"):
        return None
    if rows is None or rows < _MIN_ROWS_TO_HOLD_OUT:
        return None
    return min(_MAX_HELD_OUT, rows // 20)


def count_rows(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


def hub_rows(dataset_path: str) -> Optional[int]:
    """Training rows of a Hugging Face Hub dataset, or None if unknown."""
    try:
        resp = requests.get(
            "https://datasets-server.huggingface.co/size",
            params={"dataset": dataset_path},
            timeout=5,
        )
        if resp.status_code != 200:
            return None
        splits = resp.json().get("size", {}).get("splits", [])
        rows = sum(s.get("num_rows", 0) for s in splits if s.get("split") == "train")
        return rows or None
    except Exception:
        return None


def dataset_rows(cfg: dict, dataset_local: Optional[Path]) -> Optional[int]:
    """Rows in the job's training data, when it can be found out cheaply."""
    if dataset_local:
        return count_rows(dataset_local)
    datasets = cfg.get("datasets") or []
    if len(datasets) != 1 or not isinstance(datasets[0], dict):
        return None
    path = str(datasets[0].get("path") or "")
    if not path or path.startswith(("/", ".", "s3://", "http://", "https://")):
        return None
    return hub_rows(path)


def patch_config(
    config_path: Path,
    dataset_local: Optional[Path],
    output_dir: Path,
    resume_checkpoint_dir: Optional[Path],
    *,
    holdout: Optional[int] = None,
) -> Path:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if dataset_local and "datasets" in cfg:
        for ds in cfg["datasets"]:
            ds["path"] = str(dataset_local)
            ds.pop("ds_type", None)
    elif dataset_local:
        cfg["datasets"] = [{"path": str(dataset_local), "type": "alpaca"}]

    cfg["output_dir"] = str(output_dir)

    if isinstance(cfg.get("deepspeed"), str):
        cfg["deepspeed"] = _resolve_deepspeed_path(cfg["deepspeed"])

    if resume_checkpoint_dir:
        cfg["resume_from_checkpoint"] = str(resume_checkpoint_dir)

    if holdout:
        cfg["val_set_size"] = holdout
        # Too few rows to pack into full sequences; Axolotl refuses to start
        # when it is asked to.
        cfg["eval_sample_packing"] = False

    if "save_steps" not in cfg:
        cfg["save_steps"] = 100
    if "save_total_limit" not in cfg:
        cfg["save_total_limit"] = 3  # matches S3 checkpoint retention (_MAX_CHECKPOINTS = 3)

    cfg.pop("wandb_project", None)
    cfg.pop("wandb_entity", None)
    cfg.pop("wandb_watch", None)
    cfg["use_wandb"] = False
    cfg.pop("mlflow_experiment_name", None)

    _use_model_cache()

    patched = config_path.parent / "config_patched.yaml"
    with open(patched, "w") as f:
        yaml.dump(cfg, f)
    return patched


# ── GPU stats ─────────────────────────────────────────────────────────────────

try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_ok = True
except Exception:
    _nvml_ok = False


def _get_gpu_count() -> int:
    if not _nvml_ok:
        return 1
    try:
        return pynvml.nvmlDeviceGetCount()
    except Exception:
        return 1


def _gpu_stats() -> tuple[float, float]:
    """Return (avg_utilization_pct, total_memory_used_mb) across all GPUs."""
    if not _nvml_ok:
        return 0.0, 0.0
    try:
        count = pynvml.nvmlDeviceGetCount()
        total_util, total_mem = 0.0, 0.0
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            total_util += pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            total_mem += pynvml.nvmlDeviceGetMemoryInfo(handle).used
        return total_util / count, total_mem / (1024 * 1024)
    except Exception:
        return 0.0, 0.0


# ── Axolotl output parsing ────────────────────────────────────────────────────

def _parse_axolotl_step_line(line: str) -> Optional[dict]:
    import json
    import re

    if "'loss'" in line or '"loss"' in line:
        try:
            cleaned = line.strip().replace("'", '"')
            m = re.search(r"\{[^}]+\}", cleaned)
            if m:
                data = json.loads(m.group())
                if "loss" in data:
                    return data
        except Exception:
            pass
    return None


def _detect_checkpoint_save(line: str) -> Optional[int]:
    import re
    m = re.search(r"[Ss]aving.*checkpoint.*checkpoint-(\d+)", line)
    if m:
        return int(m.group(1))
    return None


# ── Streaming subprocess output ──────────────────────────────────────────────

def _upload_checkpoint_async(ckpt_local: Path, ckpt_step: int) -> None:
    """
    Wait for the checkpoint to be fully written (trainer_state.json is the last
    file HuggingFace Trainer writes), then tar and upload to S3.

    Runs in a background thread so it doesn't block the stdout reading loop —
    blocking would fill the axolotl pipe and deadlock training.
    """
    deadline = time.time() + 600  # give up after 10 min
    while time.time() < deadline:
        if (ckpt_local / "trainer_state.json").exists():
            break
        time.sleep(3)
    else:
        logger.warning("Checkpoint step %d: trainer_state.json never appeared, skipping upload", ckpt_step)
        return
    try:
        ckpt_s3_path = upload_checkpoint(ckpt_local, ckpt_step)
        record_checkpoint(ckpt_s3_path, ckpt_step)
        post_logs([f"[trainer] Checkpoint step {ckpt_step} saved to S3"])
    except Exception as exc:
        logger.warning("Checkpoint upload failed at step %d: %s", ckpt_step, exc)


def stream_and_collect(
    proc: subprocess.Popen,
    log_buffer: list,
    metrics_buffer: list,
    output_dir: Path,
    start_time: float,
    checkpoint_threads: list,
    tokens_per_log: int = 512,
) -> None:
    step = 0
    last_log_flush = time.time()
    last_metric_flush_step = 0
    METRIC_FLUSH_EVERY = 20
    LOG_BATCH = 50
    LOG_FLUSH_INTERVAL = 5.0
    pending_logs: list[str] = []

    for raw_line in proc.stdout:  # type: ignore[union-attr]
        line = raw_line.rstrip("\n")
        pending_logs.append(line)
        logger.info("[axolotl] %s", line)

        parsed = _parse_axolotl_step_line(line)
        if parsed:
            step += 1
            gpu_util, gpu_mem = _gpu_stats()
            elapsed = time.time() - start_time
            tps = (step * tokens_per_log) / elapsed if elapsed > 0 else 0.0

            metrics_buffer.append({
                "step": step,
                "epoch": parsed.get("epoch"),
                "train_loss": parsed.get("loss"),
                "learning_rate": parsed.get("learning_rate"),
                "gpu_utilization": gpu_util,
                "gpu_memory_used_mb": int(gpu_mem),
                "tokens_per_second": round(tps, 2),
            })

            if step - last_metric_flush_step >= METRIC_FLUSH_EVERY:
                post_metrics(list(metrics_buffer))
                metrics_buffer.clear()
                last_metric_flush_step = step

        # Detect checkpoint saves and upload asynchronously.
        # HuggingFace logs "Saving model checkpoint to ..." before writing files,
        # so we spin up a thread that waits for trainer_state.json (written last)
        # before tarballing — avoiding a corrupt upload from a race condition.
        ckpt_step = _detect_checkpoint_save(line)
        if ckpt_step is not None:
            ckpt_local = output_dir / f"checkpoint-{ckpt_step}"
            t = threading.Thread(
                target=_upload_checkpoint_async,
                args=(ckpt_local, ckpt_step),
                daemon=True,
            )
            t.start()
            checkpoint_threads.append(t)

        now = time.time()
        if len(pending_logs) >= LOG_BATCH or (now - last_log_flush) >= LOG_FLUSH_INTERVAL:
            post_logs(pending_logs)
            log_buffer.extend(pending_logs)
            pending_logs.clear()
            last_log_flush = now

    if pending_logs:
        post_logs(pending_logs)
        log_buffer.extend(pending_logs)
    if metrics_buffer:
        post_metrics(list(metrics_buffer))
        metrics_buffer.clear()


# ── Error summariser ─────────────────────────────────────────────────────────

def _extract_axolotl_error(log_buffer: list[str]) -> str:
    """
    Scan captured Axolotl output for the last traceback block and return it.
    Falls back to the last 40 lines if no traceback is found.
    """
    tb_start = -1
    for i, line in enumerate(log_buffer):
        if "Traceback (most recent call last)" in line:
            tb_start = i

    if tb_start != -1:
        return "\n".join(log_buffer[tb_start:tb_start + 60])

    return "\n".join(log_buffer[-40:])


def _friendly_error(tb: str) -> str:
    """Return a human-readable error summary, falling back to the raw traceback tail."""
    tb_lower = tb.lower()
    if "gatedrepoerror" in tb_lower or "access to model" in tb_lower:
        return (
            "Model access denied — the model is gated. "
            "Request access on huggingface.co and ensure your HF token has read permissions."
        )
    if "repositorynotfounderror" in tb_lower or "model not found" in tb_lower:
        return (
            "Model not found on HuggingFace Hub — check that the base_model ID is correct "
            "and that any private model is accessible with your HF token."
        )
    if ("401" in tb and "huggingface" in tb_lower) or "invalid user token" in tb_lower:
        return (
            "HuggingFace authentication failed — the HF token appears to be invalid or expired. "
            "Generate a new token at huggingface.co/settings/tokens."
        )
    if "cuda out of memory" in tb_lower or "out of memory" in tb_lower:
        return (
            "GPU out of memory (OOM). Try: reducing micro_batch_size, enabling "
            "gradient_checkpointing: true, using adapter: qlora with load_in_4bit: true, "
            "or choosing a larger instance type."
        )
    if "nosuchkey" in tb_lower or ("s3" in tb_lower and "clienterror" in tb_lower):
        return (
            "Dataset file not found in S3 — the uploaded dataset may have expired or the path is wrong. "
            "Re-submit the job and re-upload the dataset."
        )
    if "config name is missing" in tb_lower or "please pick one among the available configs" in tb_lower:
        import re
        m = re.search(r"\[([^\]]+)\]", tb)
        available = f" Available configs: [{m.group(1)}]." if m else ""
        return (
            "Dataset configuration name required — this HuggingFace dataset has multiple subsets "
            "and needs a 'name' field to select one."
            + available
            + " Add 'name: <config>' under the dataset entry in your config YAML "
            "(e.g. name: en)."
        )
    return tb[-2000:]


# ── Running post-training tools ───────────────────────────────────────────────

class _Finished:
    def __init__(self, returncode: int):
        self.returncode = returncode


def run_streaming(cmd: list[str], check: bool = False, timeout: Optional[float] = None) -> _Finished:
    """Run a command, sending its output to the job log as it goes.

    A drop-in for subprocess.run in export and evaluation, which can take a
    long time; without this the user would watch a silent log until the end.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    timer = threading.Timer(timeout, proc.kill) if timeout else None
    if timer:
        timer.start()
    batch: list[str] = []
    last_post = time.time()
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            logger.info(line)
            batch.append(line)
            if len(batch) >= 50 or time.time() - last_post > 5:
                post_logs(batch)
                batch, last_post = [], time.time()
        proc.wait()
    finally:
        if timer:
            timer.cancel()
        post_logs(batch)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return _Finished(proc.returncode)


def _use_model_cache() -> None:
    os.environ["TRANSFORMERS_CACHE"] = str(EFS_MODELS)
    os.environ["HF_HOME"] = str(EFS_MODELS)
    # Hub datasets would otherwise land under HF_HOME, on EFS, and stay there.
    # Fetching them again costs nothing; storing them on EFS does.
    os.environ["HF_DATASETS_CACHE"] = str(WORK_DIR / "hf-datasets")


# ── Before/after evaluation ───────────────────────────────────────────────────

_EVAL_SCRIPT = Path(__file__).with_name("model_eval.py")
_EVAL_TIMEOUT_SECONDS = 3600


def should_evaluate(cfg: dict, plan) -> tuple[bool, str]:
    """Whether to compare the fine-tuned model with the base, and if not, why."""
    if not (cfg.get("val_set_size") or cfg.get("test_datasets")):
        return False, "no rows were held out (the dataset is small, or val_set_size is 0)"
    if plan is not None and getattr(plan, "strategy", "") in ("zero3", "fsdp", "deepspeed"):
        return False, "the model was split across GPUs to train and does not fit one GPU to evaluate"
    return True, ""


def evaluate_after_training(config_path: Path, output_dir: Path, run=run_streaming) -> None:
    """Score the base and fine-tuned models on the held-out rows and report it.

    Never raises: the model is already trained, and a failed comparison must
    not turn a successful job into a failed one.
    """
    report_path = WORK_DIR / "evaluation.json"
    post_logs(["[trainer] Comparing the fine-tuned model with the base model…"])
    try:
        result = run(
            [sys.executable, str(_EVAL_SCRIPT), "--config", str(config_path),
             "--output-dir", str(output_dir), "--report", str(report_path)],
            check=False, timeout=_EVAL_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            raise RuntimeError(f"evaluation exited with code {result.returncode}")
        report = json.loads(report_path.read_text())
    except Exception as exc:
        logger.warning("Evaluation failed: %s", exc)
        post_logs([f"[trainer] Evaluation skipped: {exc}"])
        return
    _api("POST", "evaluation", json={"report": report})
    post_logs(["[trainer] Evaluation complete"])


# ── Main ──────────────────────────────────────────────────────────────────────

def run_training() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    EFS_MODELS.mkdir(parents=True, exist_ok=True)

    fetch_and_apply_hf_token()

    stop_heartbeat = threading.Event()
    heartbeat_thread = threading.Thread(target=heartbeat_loop, args=(stop_heartbeat,), daemon=True)
    heartbeat_thread.start()
    _api("POST", "heartbeat", json={"instance_id": INSTANCE_ID})

    output_dir = WORK_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_buffer: list[str] = []

    try:
        config_local = WORK_DIR / "config.yaml"
        download_from_s3(CONFIG_S3_PATH, config_local)
        post_logs([f"[trainer] Config downloaded from {CONFIG_S3_PATH}"])

        dataset_local = ensure_dataset(DATASET_S3_PATH)
        if dataset_local:
            post_logs([f"[trainer] Dataset ready at {dataset_local}"])
        else:
            post_logs(["[trainer] No external dataset; using config-specified path"])

        resume_checkpoint_dir: Optional[Path] = None
        if RESUME_FROM_CHECKPOINT_PATH:
            resume_checkpoint_dir = download_checkpoint_for_resume(RESUME_FROM_CHECKPOINT_PATH)
            post_logs([f"[trainer] Resume checkpoint ready at {resume_checkpoint_dir}"])

        raw_cfg = yaml.safe_load(config_local.read_text()) or {}
        rows = dataset_rows(raw_cfg, dataset_local)
        holdout = plan_eval_split(raw_cfg, rows)
        if holdout:
            post_logs([
                f"[trainer] Holding out {holdout} of {rows:,} examples to compare the "
                "fine-tuned model with the base model afterwards (set val_set_size: 0 to skip)"
            ])

        patched_config = patch_config(
            config_local, dataset_local, output_dir, resume_checkpoint_dir, holdout=holdout
        )

        num_gpus, per_gpu_vram_gb = _gpu_inventory()
        plan = apply_gpu_plan(patched_config, num_gpus, per_gpu_vram_gb)
        if plan is not None:
            post_logs([_describe_plan(plan, num_gpus, per_gpu_vram_gb)])
        tokens_per_log = _tokens_per_log(yaml.safe_load(patched_config.read_text()), num_gpus)
        post_logs(["[trainer] Config patched, starting Axolotl training"])
        if num_gpus > 1:
            cmd = [
                "accelerate", "launch",
                f"--num_processes={num_gpus}",
                "-m", "axolotl.cli.train",
                str(patched_config),
            ]
            post_logs([f"[trainer] Multi-GPU run: {num_gpus} GPUs via accelerate launch"])
        else:
            cmd = ["python", "-m", "axolotl.cli.train", str(patched_config)]
        logger.info("Running: %s", " ".join(cmd))

        start_time = time.time()
        metrics_buffer: list[dict] = []
        checkpoint_threads: list[threading.Thread] = []

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        stream_and_collect(
            proc, log_buffer, metrics_buffer, output_dir, start_time, checkpoint_threads,
            tokens_per_log=tokens_per_log,
        )
        proc.wait()
        elapsed = time.time() - start_time

        # Wait for any in-flight checkpoint uploads before proceeding
        for t in checkpoint_threads:
            t.join(timeout=600)

        if proc.returncode != 0:
            raise RuntimeError(f"Axolotl exited with code {proc.returncode}")

        post_logs([f"[trainer] Training complete in {elapsed:.0f}s"])

        evaluate, reason = should_evaluate(yaml.safe_load(patched_config.read_text()), plan)
        if evaluate:
            evaluate_after_training(patched_config, output_dir)
        else:
            post_logs([f"[trainer] No before/after comparison: {reason}"])
        elapsed = time.time() - start_time  # the comparison ran on the same paid instance

        model_s3_path = upload_model(output_dir)
        post_logs([f"[trainer] Model uploaded to {model_s3_path}"])

        hours = elapsed / 3600
        rate = _INSTANCE_RATES.get(INSTANCE_TYPE, 1.006)
        actual_cost = round(hours * rate, 4)

        stop_heartbeat.set()
        complete_job(model_s3_path=model_s3_path, actual_cost=actual_cost)
        logger.info("Job %s completed successfully", JOB_ID)

    except Exception:
        tb = traceback.format_exc()
        logger.error("Training failed:\n%s", tb)
        post_logs([f"[trainer] FATAL: {tb}"])
        stop_heartbeat.set()
        # Prefer the Axolotl output as the error source — it contains the root
        # cause. The wrapper traceback only has "Axolotl exited with code N".
        error_source = _extract_axolotl_error(log_buffer) if log_buffer else tb
        complete_job(error_message=_friendly_error(error_source))
        sys.exit(1)


def run_export_job() -> None:
    """Turn a finished job's model into a quantized GGUF plus Ollama Modelfile."""
    import export as exporter

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    EFS_MODELS.mkdir(parents=True, exist_ok=True)
    fetch_and_apply_hf_token()
    _use_model_cache()

    stop_heartbeat = threading.Event()
    threading.Thread(target=heartbeat_loop, args=(stop_heartbeat,), daemon=True).start()
    _api("POST", "heartbeat", json={"instance_id": INSTANCE_ID})

    try:
        start_time = time.time()
        config_local = WORK_DIR / "config.yaml"
        download_from_s3(CONFIG_S3_PATH, config_local)
        cfg = yaml.safe_load(config_local.read_text()) or {}

        tarball = WORK_DIR / "source-model.tar.gz"
        download_from_s3(SOURCE_MODEL_S3_PATH, tarball)
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(WORK_DIR, filter="data")
        tarball.unlink()
        model_dir = WORK_DIR / "model"
        post_logs([f"[export] Model ready; merging and converting to GGUF ({EXPORT_QUANT})"])

        gguf = exporter.run_export(cfg, model_dir, WORK_DIR, EXPORT_QUANT, run=run_streaming)
        name = exporter.gguf_name(str(cfg.get("base_model") or ""), JOB_ID, EXPORT_QUANT)
        key = f"models/{JOB_ID}/{name}"
        size_gb = gguf.stat().st_size / 1e9
        post_logs([f"[export] Uploading {name} ({size_gb:.1f} GB)"])
        s3.upload_file(str(gguf), S3_BUCKET, key)

        hours = (time.time() - start_time) / 3600
        stop_heartbeat.set()
        complete_job(
            model_s3_path=f"s3://{S3_BUCKET}/{key}",
            actual_cost=round(hours * _INSTANCE_RATES.get(INSTANCE_TYPE, 1.006), 4),
            modelfile=exporter.ollama_modelfile(cfg, name),
        )
        logger.info("Export %s completed", JOB_ID)
    except Exception:
        tb = traceback.format_exc()
        logger.error("Export failed:\n%s", tb)
        post_logs([f"[export] FATAL: {tb}"])
        stop_heartbeat.set()
        complete_job(error_message=_friendly_error(tb))
        sys.exit(1)


def main() -> None:
    if JOB_KIND == "export":
        run_export_job()
    else:
        run_training()


if __name__ == "__main__":
    main()
