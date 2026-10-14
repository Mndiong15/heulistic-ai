import os
import threading
import time
from pathlib import Path

import pytest
from unittest.mock import MagicMock

import train
from train import (
    _detect_checkpoint_save,
    _extract_axolotl_error,
    _friendly_error,
    _parse_axolotl_step_line,
    _s3_key,
    _upload_checkpoint_async,
    heartbeat_loop,
    patch_config,
    stream_and_collect,
)


# ── _extract_axolotl_error ────────────────────────────────────────────────────

class TestExtractAxolotlError:
    def test_returns_last_traceback_block(self):
        logs = [
            "[INFO] Starting training",
            "Traceback (most recent call last):",
            '  File "train.py", line 10, in main',
            "ValueError: something went wrong",
        ]
        result = _extract_axolotl_error(logs)
        assert "Traceback (most recent call last):" in result
        assert "ValueError: something went wrong" in result

    def test_returns_last_of_multiple_tracebacks(self):
        logs = [
            "Traceback (most recent call last):",
            "  File x, line 1",
            "OldError: first failure",
            "[INFO] Retrying...",
            "Traceback (most recent call last):",
            "  File y, line 2",
            "ValueError: actual failure",
        ]
        result = _extract_axolotl_error(logs)
        assert "ValueError: actual failure" in result
        assert "OldError: first failure" not in result

    def test_falls_back_to_last_40_lines_when_no_traceback(self):
        logs = [f"log line {i}" for i in range(60)]
        result = _extract_axolotl_error(logs)
        assert "log line 59" in result
        assert "log line 0" not in result

    def test_empty_buffer_returns_empty_string(self):
        result = _extract_axolotl_error([])
        assert result == ""

    def test_includes_up_to_60_lines_from_traceback(self):
        logs = ["Traceback (most recent call last):"] + [f"  frame {i}" for i in range(100)]
        result = _extract_axolotl_error(logs)
        lines = result.splitlines()
        assert len(lines) <= 60


# ── _friendly_error ───────────────────────────────────────────────────────────

class TestFriendlyError:
    def test_gated_model(self):
        tb = "GatedRepoError: Access to model meta-llama/Llama-2-7b-hf is restricted"
        result = _friendly_error(tb)
        assert "gated" in result.lower()
        assert "hf token" in result.lower()

    def test_model_not_found(self):
        tb = "RepositoryNotFoundError: model not found: foo/bar"
        result = _friendly_error(tb)
        assert "not found" in result.lower()
        assert "base_model" in result.lower()

    def test_hf_auth_failure(self):
        tb = "401 unauthorized — invalid user token"
        result = _friendly_error(tb)
        assert "authentication" in result.lower() or "token" in result.lower()

    def test_cuda_oom(self):
        tb = "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
        result = _friendly_error(tb)
        assert "out of memory" in result.lower() or "oom" in result.lower()
        assert "micro_batch_size" in result

    def test_s3_missing_dataset(self):
        tb = "botocore.exceptions.ClientError: NoSuchKey s3://bucket/key"
        result = _friendly_error(tb)
        assert "s3" in result.lower() or "dataset" in result.lower()

    def test_config_name_missing_with_available_configs(self):
        tb = (
            "ValueError: Config name is missing.\n"
            "Please pick one among the available configs: ['ar', 'en', 'fr', 'zh']\n"
            "Example of usage:\n"
            "    `load_dataset('Qwen/PolyMath', 'ar')`"
        )
        result = _friendly_error(tb)
        assert "name" in result.lower()
        assert "subset" in result.lower() or "config" in result.lower()
        assert "ar" in result or "en" in result

    def test_config_name_missing_surfaces_add_name_instruction(self):
        tb = "ValueError: Config name is missing.\nPlease pick one among the available configs: ['en']"
        result = _friendly_error(tb)
        assert "name:" in result or "'name'" in result

    def test_unknown_error_falls_back_to_tail(self):
        long_tb = "some preamble\n" + "x" * 3000
        result = _friendly_error(long_tb)
        assert len(result) <= 2000

    def test_unknown_error_includes_end_of_traceback(self):
        tb = "irrelevant stuff\n" * 10 + "ActualError: the real message"
        result = _friendly_error(tb)
        assert "ActualError: the real message" in result


# ── _parse_axolotl_step_line ──────────────────────────────────────────────────

class TestParseAxolotlStepLine:
    def test_parses_loss_from_standard_format(self):
        line = "{'loss': 1.2345, 'grad_norm': 0.5, 'learning_rate': 2e-05, 'epoch': 0.5}"
        result = _parse_axolotl_step_line(line)
        assert result is not None
        assert abs(result["loss"] - 1.2345) < 1e-6

    def test_returns_none_for_non_metric_line(self):
        assert _parse_axolotl_step_line("Loading model from checkpoint") is None
        assert _parse_axolotl_step_line("") is None
        assert _parse_axolotl_step_line("Training complete") is None

    def test_parses_epoch(self):
        line = "{'loss': 0.9, 'learning_rate': 1e-04, 'epoch': 1.5}"
        result = _parse_axolotl_step_line(line)
        assert result is not None
        assert result["epoch"] == 1.5

    def test_parses_learning_rate(self):
        line = "{'loss': 0.7, 'learning_rate': 0.0002, 'epoch': 2.0}"
        result = _parse_axolotl_step_line(line)
        assert result is not None
        assert "learning_rate" in result


# ── _detect_checkpoint_save ───────────────────────────────────────────────────

class TestDetectCheckpointSave:
    def test_detects_save_line(self):
        line = "Saving model checkpoint to /tmp/training/output/checkpoint-100"
        step = _detect_checkpoint_save(line)
        assert step == 100

    def test_detects_lowercase_saving(self):
        line = "saving checkpoint to checkpoint-250"
        step = _detect_checkpoint_save(line)
        assert step == 250

    def test_no_match_on_regular_line(self):
        assert _detect_checkpoint_save("Training step 50 complete") is None
        assert _detect_checkpoint_save("Loading checkpoint for resume") is None
        assert _detect_checkpoint_save("") is None


# ── _s3_key ───────────────────────────────────────────────────────────────────

class TestS3Key:
    def test_parses_standard_s3_path(self):
        bucket, key = _s3_key("s3://my-bucket/path/to/model.tar.gz")
        assert bucket == "my-bucket"
        assert key == "path/to/model.tar.gz"

    def test_single_segment_key(self):
        bucket, key = _s3_key("s3://my-bucket/file.yaml")
        assert bucket == "my-bucket"
        assert key == "file.yaml"

    def test_non_s3_path_uses_default_bucket(self):
        # bare key (no s3:// prefix) falls back to module-level S3_BUCKET
        bucket, key = _s3_key("configs/job123/config.yaml")
        assert key == "configs/job123/config.yaml"


# ── patch_config ──────────────────────────────────────────────────────────────

class TestPatchConfig:
    def _write_config(self, tmp_path, content: str) -> Path:
        p = tmp_path / "config.yaml"
        p.write_text(content)
        return p

    def test_sets_output_dir(self, tmp_path):
        cfg_path = self._write_config(tmp_path, "base_model: llama\nnum_epochs: 1\n")
        output_dir = tmp_path / "output"
        patched = patch_config(cfg_path, None, output_dir, None)

        import yaml
        with open(patched) as f:
            cfg = yaml.safe_load(f)
        assert cfg["output_dir"] == str(output_dir)

    def test_injects_dataset_path_into_existing_datasets(self, tmp_path):
        content = "base_model: llama\ndatasets:\n  - path: /old/path\n    type: alpaca\n"
        cfg_path = self._write_config(tmp_path, content)
        dataset_local = tmp_path / "data.jsonl"
        dataset_local.touch()

        patched = patch_config(cfg_path, dataset_local, tmp_path / "out", None)

        import yaml
        with open(patched) as f:
            cfg = yaml.safe_load(f)
        assert cfg["datasets"][0]["path"] == str(dataset_local)
        assert "ds_type" not in cfg["datasets"][0]

    def test_disables_wandb(self, tmp_path):
        content = "base_model: llama\nwandb_project: my-project\nwandb_entity: org\n"
        cfg_path = self._write_config(tmp_path, content)

        patched = patch_config(cfg_path, None, tmp_path / "out", None)

        import yaml
        with open(patched) as f:
            cfg = yaml.safe_load(f)
        assert cfg.get("use_wandb") is False
        assert "wandb_project" not in cfg
        assert "wandb_entity" not in cfg

    def test_sets_resume_checkpoint(self, tmp_path):
        cfg_path = self._write_config(tmp_path, "base_model: llama\n")
        resume_dir = tmp_path / "resume_ckpt"
        resume_dir.mkdir()

        patched = patch_config(cfg_path, None, tmp_path / "out", resume_dir)

        import yaml
        with open(patched) as f:
            cfg = yaml.safe_load(f)
        assert cfg["resume_from_checkpoint"] == str(resume_dir)

    def test_adds_save_steps_defaults(self, tmp_path):
        cfg_path = self._write_config(tmp_path, "base_model: llama\n")

        patched = patch_config(cfg_path, None, tmp_path / "out", None)

        import yaml
        with open(patched) as f:
            cfg = yaml.safe_load(f)
        assert cfg["save_steps"] == 100
        assert cfg["save_total_limit"] == 3

    def test_does_not_override_existing_save_steps(self, tmp_path):
        cfg_path = self._write_config(tmp_path, "base_model: llama\nsave_steps: 50\n")

        patched = patch_config(cfg_path, None, tmp_path / "out", None)

        import yaml
        with open(patched) as f:
            cfg = yaml.safe_load(f)
        assert cfg["save_steps"] == 50


# ── heartbeat_loop ────────────────────────────────────────────────────────────

class TestHeartbeatLoop:
    """
    heartbeat_loop runs stop_event.wait(60) at the top of each iteration.
    We patch wait() to return False (run body) on the first call and True
    (stop loop) on subsequent calls, so tests exercise exactly one iteration
    without sleeping.
    """

    def _response(self, status_code: int) -> MagicMock:
        r = MagicMock()
        r.status_code = status_code
        return r

    def _one_shot_stop(self, stop: threading.Event) -> None:
        """Patch stop.wait so the loop body runs once then exits."""
        call_count = 0

        def fake_wait(timeout):
            nonlocal call_count
            call_count += 1
            return call_count > 1

        stop.wait = fake_wait

    def test_409_calls_os_exit(self, monkeypatch):
        stop = threading.Event()
        self._one_shot_stop(stop)
        monkeypatch.setattr(train._SESSION, "request", lambda *a, **kw: self._response(409))

        exit_codes = []
        monkeypatch.setattr(os, "_exit", lambda code: exit_codes.append(code))

        heartbeat_loop(stop)

        assert exit_codes == [1]

    def test_200_does_not_call_os_exit(self, monkeypatch):
        stop = threading.Event()
        self._one_shot_stop(stop)
        monkeypatch.setattr(train._SESSION, "request", lambda *a, **kw: self._response(200))

        exit_codes = []
        monkeypatch.setattr(os, "_exit", lambda code: exit_codes.append(code))

        heartbeat_loop(stop)

        assert exit_codes == []

    def test_other_4xx_does_not_call_os_exit(self, monkeypatch):
        stop = threading.Event()
        self._one_shot_stop(stop)
        monkeypatch.setattr(train._SESSION, "request", lambda *a, **kw: self._response(404))

        exit_codes = []
        monkeypatch.setattr(os, "_exit", lambda code: exit_codes.append(code))

        heartbeat_loop(stop)

        assert exit_codes == []

    def test_network_error_does_not_call_os_exit(self, monkeypatch):
        stop = threading.Event()
        self._one_shot_stop(stop)
        monkeypatch.setattr(
            train._SESSION, "request",
            lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("timeout")),
        )

        exit_codes = []
        monkeypatch.setattr(os, "_exit", lambda code: exit_codes.append(code))

        heartbeat_loop(stop)

        assert exit_codes == []

    def test_posts_to_heartbeat_endpoint(self, monkeypatch):
        stop = threading.Event()
        self._one_shot_stop(stop)

        calls = []
        def fake_request(method, url, **kwargs):
            calls.append((method, url))
            return self._response(200)

        monkeypatch.setattr(train._SESSION, "request", fake_request)
        monkeypatch.setattr(os, "_exit", lambda code: None)

        heartbeat_loop(stop)

        assert len(calls) == 1
        method, url = calls[0]
        assert method == "POST"
        assert "heartbeat" in url
        assert train.JOB_ID in url

    def test_sends_instance_id_in_payload(self, monkeypatch):
        stop = threading.Event()
        self._one_shot_stop(stop)

        payloads = []
        def fake_request(method, url, **kwargs):
            payloads.append(kwargs.get("json", {}))
            return self._response(200)

        monkeypatch.setattr(train._SESSION, "request", fake_request)

        heartbeat_loop(stop)

        assert payloads[0]["instance_id"] == train.INSTANCE_ID


# ── gpu_memory_used_mb must be int ────────────────────────────────────────────
#
# Regression: _gpu_stats() returns a float (bytes / 1024² = e.g. 3921.333).
# Pydantic v2 in lax mode rejects non-integer floats for int | None fields,
# so every metrics POST returned 422 and metrics never appeared in the UI.
# Fix: cast to int() before building the metrics dict.

class TestGpuMemoryIsInt:
    class _FakeProc:
        def __init__(self, lines: list[str]):
            self.stdout = iter(line + "\n" for line in lines)

    def test_gpu_memory_used_mb_is_int_for_fractional_raw_value(self, monkeypatch, tmp_path):
        monkeypatch.setattr(train, "_gpu_stats", lambda: (75.0, 3921.333))

        posted: list[dict] = []
        monkeypatch.setattr(train, "post_metrics", lambda m: posted.extend(m))
        monkeypatch.setattr(train, "post_logs", lambda lines: None)

        proc = self._FakeProc(["{'loss': 1.2, 'learning_rate': 2e-05, 'epoch': 0.1}"])
        stream_and_collect(proc, [], [], tmp_path, time.time(), [])

        assert posted, "expected at least one metric to be posted"
        assert isinstance(posted[0]["gpu_memory_used_mb"], int)
        assert posted[0]["gpu_memory_used_mb"] == 3921

    def test_gpu_memory_truncates_not_rounds(self, monkeypatch, tmp_path):
        # int(3921.999) == 3921, not 3922 — verify truncation behaviour
        monkeypatch.setattr(train, "_gpu_stats", lambda: (0.0, 3921.999))

        posted: list[dict] = []
        monkeypatch.setattr(train, "post_metrics", lambda m: posted.extend(m))
        monkeypatch.setattr(train, "post_logs", lambda lines: None)

        proc = self._FakeProc(["{'loss': 0.5, 'learning_rate': 1e-05, 'epoch': 0.2}"])
        stream_and_collect(proc, [], [], tmp_path, time.time(), [])

        assert isinstance(posted[0]["gpu_memory_used_mb"], int)
        assert posted[0]["gpu_memory_used_mb"] == 3921


# ── _upload_checkpoint_async ──────────────────────────────────────────────────
#
# Regression: HuggingFace Trainer logs "Saving model checkpoint to …" *before*
# writing any files. Uploading immediately produced a corrupt or empty tarball.
# Fix: spin-wait for trainer_state.json (the last file written) before uploading.

class TestUploadCheckpointAsync:
    def test_calls_upload_when_state_file_exists(self, monkeypatch, tmp_path):
        (tmp_path / "trainer_state.json").write_text("{}")

        uploaded: list[tuple] = []

        def fake_upload(path, step):
            uploaded.append((path, step))
            return "s3://bucket/key"

        monkeypatch.setattr(train, "upload_checkpoint", fake_upload)
        monkeypatch.setattr(train, "record_checkpoint", lambda path, step: None)
        monkeypatch.setattr(train, "post_logs", lambda lines: None)

        _upload_checkpoint_async(tmp_path, 100)

        assert uploaded == [(tmp_path, 100)]

    def test_skips_upload_when_state_file_never_appears(self, monkeypatch, tmp_path):
        # Fast-forward time past the 10-minute deadline without touching the filesystem.
        call_count = [0]

        def fake_time():
            call_count[0] += 1
            # First call sets deadline = 0 + 600 = 600; subsequent calls return 601
            return 0 if call_count[0] == 1 else 601

        monkeypatch.setattr(train.time, "time", fake_time)
        monkeypatch.setattr(train.time, "sleep", lambda s: None)

        uploaded: list = []
        monkeypatch.setattr(train, "upload_checkpoint", lambda path, step: uploaded.append((path, step)))

        _upload_checkpoint_async(tmp_path, 200)

        assert uploaded == []


# ── Multi-GPU: DeepSpeed paths, sharding, throughput ─────────────────────────

import train as _train  # noqa: E402


@pytest.fixture
def shipped_configs(tmp_path, monkeypatch):
    """Stand in for /workspace/axolotl/deepspeed_configs in the real image."""
    d = tmp_path / "axolotl_ds"
    d.mkdir()
    for name in ("zero2.json", "zero3.json", "zero3_bf16.json"):
        (d / name).write_text("{}")
    monkeypatch.setattr(_train, "AXOLOTL_DEEPSPEED_DIR", d)
    return d


class TestDeepspeedPath:
    """Axolotl's examples write `deepspeed: deepspeed_configs/zero2.json`. In
    the image those files live under /workspace/axolotl while the working
    directory is /app, so the path did not resolve and the run died at launch."""

    def test_relative_path_is_pointed_at_the_shipped_config(self, shipped_configs, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert _train._resolve_deepspeed_path("deepspeed_configs/zero2.json") == str(
            shipped_configs / "zero2.json"
        )

    def test_absolute_path_is_left_alone(self, shipped_configs):
        assert _train._resolve_deepspeed_path("/custom/zero2.json") == "/custom/zero2.json"

    def test_a_path_that_resolves_locally_is_left_alone(self, shipped_configs, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "mine.json").write_text("{}")
        assert _train._resolve_deepspeed_path("mine.json") == "mine.json"

    def test_an_unknown_config_is_left_for_axolotl_to_report(self, shipped_configs, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert _train._resolve_deepspeed_path("nope/zero9.json") == "nope/zero9.json"

    def test_patch_config_applies_it(self, shipped_configs, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("base_model: llama\ndeepspeed: deepspeed_configs/zero2.json\n")
        patched = patch_config(cfg_path, None, tmp_path / "out", None)
        import yaml
        assert yaml.safe_load(patched.read_text())["deepspeed"] == str(shipped_configs / "zero2.json")


class TestPlanGpus:
    def test_one_gpu_needs_no_plan(self, shipped_configs):
        cfg = {"base_model": "org/m-7b"}
        assert _train.plan_gpus(cfg, gpu_count=1, per_gpu_vram_gb=24) is None
        assert "deepspeed" not in cfg

    def test_a_model_that_fits_one_card_runs_ddp(self, shipped_configs):
        cfg = {"base_model": "org/m-7b", "adapter": "lora"}  # 16.8 GB
        plan = _train.plan_gpus(cfg, gpu_count=4, per_gpu_vram_gb=22)
        assert plan.strategy == "ddp"
        assert "deepspeed" not in cfg

    def test_a_model_too_big_for_one_card_is_sharded(self, shipped_configs):
        cfg = {"base_model": "org/m-20b", "adapter": "lora"}  # 48 GB on 4x16
        plan = _train.plan_gpus(cfg, gpu_count=4, per_gpu_vram_gb=16)
        assert plan.strategy == "zero3" and plan.auto_sharded
        assert cfg["deepspeed"] == str(shipped_configs / "zero3.json")

    def test_auto_sharding_never_forces_bf16(self, shipped_configs):
        """zero3_bf16.json hard-enables bf16, which T4s (g4dn) do not support —
        and the 4xT4 g4dn.12xlarge is exactly where sharded jobs get routed."""
        cfg = {"base_model": "org/m-20b", "adapter": "lora"}
        _train.plan_gpus(cfg, gpu_count=4, per_gpu_vram_gb=16)
        assert "bf16" not in Path(cfg["deepspeed"]).name

    def test_the_users_own_deepspeed_config_is_kept(self, shipped_configs):
        cfg = {"base_model": "org/m-20b", "adapter": "lora", "deepspeed": "/mine/zero3.json"}
        _train.plan_gpus(cfg, gpu_count=4, per_gpu_vram_gb=16)
        assert cfg["deepspeed"] == "/mine/zero3.json"

    def test_a_quantized_model_too_big_for_one_card_is_not_sharded(self, shipped_configs):
        cfg = {"base_model": "org/m-70b", "adapter": "qlora", "load_in_4bit": True}  # 42 GB
        plan = _train.plan_gpus(cfg, gpu_count=4, per_gpu_vram_gb=22)
        assert plan.auto_sharded is False and plan.fits is False
        assert "deepspeed" not in cfg

    def test_unknown_vram_needs_no_plan(self, shipped_configs):
        # No NVML in the container: nothing to plan against.
        assert _train.plan_gpus({"base_model": "org/m-7b"}, gpu_count=4, per_gpu_vram_gb=0) is None


class TestGpuInventory:
    def test_without_nvml(self, monkeypatch):
        monkeypatch.setattr(_train, "_nvml_ok", False)
        assert _train._gpu_inventory() == (1, 0.0)

    def test_reports_count_and_the_smallest_card(self, monkeypatch):
        """A mixed rig is sized by its smallest card: every rank must fit."""
        gib = 1024 ** 3
        sizes = [24 * gib, 16 * gib, 24 * gib]

        class FakeNvml:
            def nvmlDeviceGetCount(self): return len(sizes)
            def nvmlDeviceGetHandleByIndex(self, i): return i
            def nvmlDeviceGetMemoryInfo(self, h):
                return type("M", (), {"total": sizes[h]})()

        monkeypatch.setattr(_train, "_nvml_ok", True)
        monkeypatch.setattr(_train, "pynvml", FakeNvml(), raising=False)
        assert _train._gpu_inventory() == (3, 16.0)


class TestTokensPerLog:
    """Throughput was `step * 512 / elapsed` — a fixed 512 tokens per logged
    step regardless of sequence length, batch size, or GPU count."""

    def test_counts_every_dimension(self):
        cfg = {"sequence_len": 2048, "micro_batch_size": 2, "gradient_accumulation_steps": 4}
        assert _train._tokens_per_log(cfg, gpu_count=4) == 2048 * 2 * 4 * 4

    def test_defaults(self):
        # Axolotl's own defaults: 512 tokens, batch 1, no accumulation.
        assert _train._tokens_per_log({}, gpu_count=1) == 512

    def test_logging_interval_scales_it(self):
        # One loss line covers `logging_steps` optimiser steps.
        cfg = {"sequence_len": 1024, "logging_steps": 10}
        assert _train._tokens_per_log(cfg, gpu_count=1) == 1024 * 10


# ── Dataset storage ───────────────────────────────────────────────────────────

class TestDatasetsStayOnInstanceDisk:
    """Datasets used to be copied to EFS under a content hash and never
    deleted. The "cache" could not save anything — the file had to be
    downloaded in full to compute the hash it was looked up by — while EFS
    billed every copy at ~$0.30/GB-month. The instance's own disk is already
    paid for by the job and disappears with it."""

    def test_uploaded_dataset_is_downloaded_to_the_work_dir(self, tmp_path, monkeypatch):
        efs = tmp_path / "efs-datasets"
        monkeypatch.setattr(train, "WORK_DIR", tmp_path / "work")
        monkeypatch.setattr(train, "EFS_DATASETS", efs, raising=False)

        def fake_download(s3_path, local_path):
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text('{"text": "hi"}\n')

        monkeypatch.setattr(train, "download_from_s3", fake_download)

        path = train.ensure_dataset("s3://bucket/temp-datasets/u/1/data.jsonl")

        assert path.read_text() == '{"text": "hi"}\n'
        assert (tmp_path / "work") in path.parents
        assert not efs.exists() or not any(efs.iterdir())

    def test_no_dataset_means_no_path(self):
        assert train.ensure_dataset("") is None

    def test_hub_datasets_are_cached_on_the_work_dir_not_efs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(train, "WORK_DIR", tmp_path / "work")
        monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("base_model: llama\n")

        patch_config(cfg_path, None, tmp_path / "output", None)

        cache = Path(os.environ["HF_DATASETS_CACHE"])
        assert (tmp_path / "work") in cache.parents
        # Model weights keep the shared EFS cache: re-downloading them costs
        # the job minutes of GPU time.
        assert os.environ["HF_HOME"] == str(train.EFS_MODELS)
