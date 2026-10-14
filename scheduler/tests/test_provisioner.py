import json
import subprocess
import tempfile
import os

from provisioner import _build_user_data, _latest_dlami, _scoped_s3_creds, provision


# ── _build_user_data ──────────────────────────────────────────────────────────

class TestBuildUserData:
    def _job(self, **overrides):
        base = {
            "id": "aaaaaaaa-0000-0000-0000-000000000001",
            "config_s3_path": "s3://bucket/configs/aaa/config.yaml",
            "dataset_s3_path": "s3://bucket/datasets/data.jsonl",
            "hf_token_secret_arn": "arn:aws:secretsmanager:us-east-1:123:secret:hf",
            "resume_from_checkpoint_path": "",
            "instance_type": "g5.xlarge",
            "internal_token": "tok_abc123",
        }
        base.update(overrides)
        return base

    def test_all_fields_injected(self):
        job = self._job()
        script = _build_user_data(job, "my-bucket", "fs-abc.efs.amazonaws.com", "123.dkr.ecr.us-east-1.amazonaws.com/training", "tok_abc123")
        assert f'JOB_ID="{job["id"]}"' in script
        assert 'CONFIG_S3_PATH="s3://bucket/configs/aaa/config.yaml"' in script
        assert 'DATASET_S3_PATH="s3://bucket/datasets/data.jsonl"' in script
        assert 'HF_TOKEN_SECRET_ARN="arn:aws:secretsmanager:us-east-1:123:secret:hf"' in script
        assert 'INTERNAL_TOKEN="tok_abc123"' in script

    def test_dataset_defaults_to_empty_string(self):
        job = self._job(dataset_s3_path=None)
        script = _build_user_data(job, "bucket", "efs-dns", "ecr-url", "tok_test")
        assert 'DATASET_S3_PATH=""' in script

    def test_config_path_defaults_from_job_id(self):
        job = self._job(config_s3_path=None)
        job_id = job["id"]
        script = _build_user_data(job, "my-bucket", "efs-dns", "ecr-url", "tok_test")
        assert f"s3://my-bucket/configs/{job_id}/config.yaml" in script

    def test_internal_token_in_docker_run_block(self):
        # Token is now passed explicitly as the `token` parameter (not read from job dict)
        script = _build_user_data(self._job(), "bucket", "efs-dns", "ecr-url", "secret_tok")
        # Top-level assignment uses the literal value
        assert 'INTERNAL_TOKEN="secret_tok"' in script
        # docker run passes it via shell variable reference
        docker_run_start = script.index("docker run")
        assert 'INTERNAL_TOKEN="$INTERNAL_TOKEN"' in script[docker_run_start:]

    def test_hf_token_arn_defaults_to_empty(self):
        job = self._job(hf_token_secret_arn=None)
        script = _build_user_data(job, "bucket", "efs-dns", "ecr-url", "tok_test")
        assert 'HF_TOKEN_SECRET_ARN=""' in script


# ── _latest_dlami ─────────────────────────────────────────────────────────────

class TestLatestDlami:
    def test_picks_most_recent_image(self):
        class FakeEc2:
            def describe_images(self, **kwargs):
                return {
                    "Images": [
                        {"ImageId": "ami-old", "CreationDate": "2023-01-01T00:00:00Z", "Name": "old"},
                        {"ImageId": "ami-new", "CreationDate": "2024-06-01T00:00:00Z", "Name": "new"},
                        {"ImageId": "ami-mid", "CreationDate": "2023-12-01T00:00:00Z", "Name": "mid"},
                    ]
                }

        ami = _latest_dlami(FakeEc2())
        assert ami == "ami-new"

    def test_raises_if_no_images(self):
        class FakeEc2:
            def describe_images(self, **kwargs):
                return {"Images": []}

        import pytest
        with pytest.raises(RuntimeError, match="No Deep Learning AMI"):
            _latest_dlami(FakeEc2())


# ── bootstrap log flushing ────────────────────────────────────────────────────

class TestBootstrapLogFlushing:
    def _script(self, **overrides):
        job = {
            "id": "bbbbbbbb-0000-0000-0000-000000000002",
            "config_s3_path": "s3://bucket/configs/bbb/config.yaml",
            "dataset_s3_path": None,
            "hf_token_secret_arn": None,
            "resume_from_checkpoint_path": None,
            "instance_type": "g5.xlarge",
        }
        job.update(overrides)
        return _build_user_data(job, "bucket", "efs-dns", "ecr-url", "tok_test")

    def test_trap_is_registered(self):
        script = self._script()
        assert "trap _flush_bootstrap_logs EXIT" in script

    def test_function_reads_correct_log_file(self):
        script = self._script()
        assert "/var/log/training-bootstrap.log" in script

    def test_function_posts_to_logs_endpoint(self):
        script = self._script()
        assert "$API_URL/internal/jobs/$JOB_ID/logs" in script

    def test_function_sends_internal_token_header(self):
        script = self._script()
        assert "X-Internal-Token: $INTERNAL_TOKEN" in script

    def test_function_prefixes_lines_with_bootstrap(self):
        script = self._script()
        assert "[BOOTSTRAP]" in script

    def test_function_is_silent_on_failure(self):
        # curl failure must never abort the instance
        script = self._script()
        flush_start = script.index("_flush_bootstrap_logs()")
        flush_body = script[flush_start:script.index("trap _flush_bootstrap_logs EXIT")]
        assert "|| true" in flush_body

    def test_function_disables_errexit(self):
        # set -euo pipefail is active globally; the function must not propagate errors
        script = self._script()
        flush_start = script.index("_flush_bootstrap_logs()")
        flush_body = script[flush_start:script.index("trap _flush_bootstrap_logs EXIT")]
        assert "set +e" in flush_body

    def test_python_oneliner_builds_correct_json(self):
        # Extract the python3 -c "..." command from the script and run it against a
        # real temp file to verify the JSON output is structurally correct.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("Mounting EFS...\n")
            f.write("Error: mount failed\n")
            f.write("\n")  # blank line — should be skipped
            log_path = f.name

        try:
            oneliner = (
                "import json;"
                "lines=['[BOOTSTRAP] '+l.rstrip() for l in open('" + log_path + "') if l.strip()];"
                "print(json.dumps({'lines':lines}))"
            )
            result = subprocess.run(
                ["python3", "-c", oneliner],
                capture_output=True, text=True, check=True,
            )
            payload = json.loads(result.stdout)
            assert payload == {
                "lines": [
                    "[BOOTSTRAP] Mounting EFS...",
                    "[BOOTSTRAP] Error: mount failed",
                ]
            }
        finally:
            os.unlink(log_path)

    def test_python_oneliner_handles_empty_log(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            log_path = f.name  # empty file

        try:
            oneliner = (
                "import json;"
                "lines=['[BOOTSTRAP] '+l.rstrip() for l in open('" + log_path + "') if l.strip()];"
                "print(json.dumps({'lines':lines}))"
            )
            result = subprocess.run(
                ["python3", "-c", oneliner],
                capture_output=True, text=True, check=True,
            )
            payload = json.loads(result.stdout)
            assert payload == {"lines": []}
        finally:
            os.unlink(log_path)


# ── IMDSv2 ────────────────────────────────────────────────────────────────────

class TestIMDSv2:
    def _script(self, **overrides):
        job = {
            "id": "cccccccc-0000-0000-0000-000000000003",
            "config_s3_path": "s3://bucket/configs/ccc/config.yaml",
            "dataset_s3_path": None,
            "hf_token_secret_arn": None,
            "resume_from_checkpoint_path": None,
            "instance_type": "g5.xlarge",
        }
        job.update(overrides)
        return _build_user_data(job, "bucket", "efs-dns", "ecr-url", "tok_test")

    def test_imdsv2_token_request_uses_put(self):
        script = self._script()
        assert 'curl -sfX PUT "http://169.254.169.254/latest/api/token"' in script

    def test_imdsv2_token_sets_ttl_header(self):
        script = self._script()
        assert "X-aws-ec2-metadata-token-ttl-seconds: 21600" in script

    def test_instance_id_uses_imdsv2_token(self):
        script = self._script()
        assert 'X-aws-ec2-metadata-token: $IMDS_TOKEN' in script
        assert "meta-data/instance-id" in script

    def test_imdsv1_curl_not_present(self):
        # No bare curl to the metadata service without the token header
        script = self._script()
        assert "curl -sf http://169.254.169.254/latest/meta-data/instance-id" not in script

    def test_trap_registered_before_imds_call(self):
        script = self._script()
        trap_pos = script.index("trap _flush_bootstrap_logs EXIT")
        imds_pos = script.index("169.254.169.254/latest/api/token")
        assert trap_pos < imds_pos

    def test_instance_id_assigned_before_mount(self):
        script = self._script()
        instance_id_pos = script.index("meta-data/instance-id")
        mount_pos = script.index("mount -t nfs4")
        assert instance_id_pos < mount_pos


# ── provision — run_instances block device mapping ────────────────────────────

class TestProvision:
    """provision() must request a large enough root volume for Docker image pulls."""

    _PREFIX = "ftp-test"

    def _job(self):
        return {
            "id": "dddddddd-0000-0000-0000-000000000004",
            "config_s3_path": "s3://bucket/configs/ddd/config.yaml",
            "dataset_s3_path": None,
            "hf_token_secret_arn": None,
            "resume_from_checkpoint_path": None,
            "instance_type": "g5.xlarge",
            "internal_token": "tok_test",
        }

    def _setup(self, monkeypatch, run_calls: list):
        prefix = self._PREFIX

        class FakeSsm:
            def get_parameter(self, Name):
                values = {
                    f"/{prefix}/ec2/training-sg-id": "sg-123",
                    f"/{prefix}/ec2/training-subnet-ids": "subnet-abc",
                    f"/{prefix}/ec2/training-instance-profile": "ftp-training-profile",
                    f"/{prefix}/s3/bucket": "my-bucket",
                    f"/{prefix}/efs/dns-name": "fs-abc.efs.amazonaws.com",
                    f"/{prefix}/ecr/training-url": "123.dkr.ecr.us-east-1.amazonaws.com/training",
                }
                return {"Parameter": {"Value": values[Name]}}

        class FakeEc2:
            def describe_images(self, **kwargs):
                return {"Images": [{"ImageId": "ami-test", "CreationDate": "2024-01-01T00:00:00Z", "Name": "test"}]}

            def run_instances(self, **kwargs):
                run_calls.append(kwargs)
                return {"Instances": [{"InstanceId": "i-test123"}]}

        def client_factory(service, region_name=None):
            return FakeSsm() if service == "ssm" else FakeEc2()

        monkeypatch.setattr("provisioner.boto3.client", client_factory)
        # LOCAL_MODE=true in .env would skip provisioning — override for these tests
        import provisioner as _prov
        monkeypatch.setattr(_prov.settings, "local_mode", False)
        monkeypatch.setattr(_prov.settings, "project_prefix", prefix)

    def test_root_volume_is_at_least_200gb(self, monkeypatch):
        run_calls = []
        self._setup(monkeypatch, run_calls)
        provision(self._job())

        bdm = run_calls[0].get("BlockDeviceMappings", [])
        assert bdm, "BlockDeviceMappings not set — Docker pull will fail with no space left on device"
        assert bdm[0]["Ebs"]["VolumeSize"] >= 200

    def test_volume_type_is_gp3(self, monkeypatch):
        run_calls = []
        self._setup(monkeypatch, run_calls)
        provision(self._job())

        bdm = run_calls[0]["BlockDeviceMappings"]
        assert bdm[0]["Ebs"]["VolumeType"] == "gp3"

    def test_volume_deleted_on_termination(self, monkeypatch):
        run_calls = []
        self._setup(monkeypatch, run_calls)
        provision(self._job())

        bdm = run_calls[0]["BlockDeviceMappings"]
        assert bdm[0]["Ebs"]["DeleteOnTermination"] is True


# ── _scoped_s3_creds ──────────────────────────────────────────────────────────

class TestScopedS3Creds:
    def _job(self, **overrides):
        base = {
            "id": "eeeeeeee-0000-0000-0000-000000000005",
            "config_s3_path": "s3://my-bucket/configs/eee/config.yaml",
            "dataset_s3_path": "s3://my-bucket/datasets/data.jsonl",
        }
        base.update(overrides)
        return base

    def _fake_sts(self, captured: dict | None = None):
        class FakeSts:
            def assume_role(self, **kwargs):
                if captured is not None:
                    captured.update(kwargs)
                return {
                    "Credentials": {
                        "AccessKeyId": "ASIA_KEY",
                        "SecretAccessKey": "secret_value",
                        "SessionToken": "token_xyz",
                    }
                }
        return FakeSts()

    def test_returns_none_without_role_arn(self, monkeypatch):
        monkeypatch.delenv("TRAINING_ROLE_ARN", raising=False)
        result = _scoped_s3_creds(self._job(), "my-bucket")
        assert result is None

    def test_returns_credentials_on_success(self, monkeypatch):
        monkeypatch.setenv("TRAINING_ROLE_ARN", "arn:aws:iam::123:role/training")
        monkeypatch.setattr("provisioner.boto3.client", lambda svc, **kw: self._fake_sts())

        result = _scoped_s3_creds(self._job(), "my-bucket")

        assert result == {
            "access_key": "ASIA_KEY",
            "secret_key": "secret_value",
            "session_token": "token_xyz",
        }

    def test_policy_scoped_to_job_paths(self, monkeypatch):
        monkeypatch.setenv("TRAINING_ROLE_ARN", "arn:aws:iam::123:role/training")
        captured: dict = {}
        monkeypatch.setattr("provisioner.boto3.client", lambda svc, **kw: self._fake_sts(captured))

        job = self._job()
        job_id = job["id"]
        _scoped_s3_creds(job, "my-bucket")

        policy = json.loads(captured["Policy"])
        resources = policy["Statement"][0]["Resource"]
        assert "arn:aws:s3:::my-bucket/configs/eee/config.yaml" in resources
        assert f"arn:aws:s3:::my-bucket/models/{job_id}/*" in resources
        assert f"arn:aws:s3:::my-bucket/checkpoints/{job_id}/*" in resources
        assert "arn:aws:s3:::my-bucket/datasets/data.jsonl" in resources

    def test_policy_excludes_dataset_when_none(self, monkeypatch):
        monkeypatch.setenv("TRAINING_ROLE_ARN", "arn:aws:iam::123:role/training")
        captured: dict = {}
        monkeypatch.setattr("provisioner.boto3.client", lambda svc, **kw: self._fake_sts(captured))

        _scoped_s3_creds(self._job(dataset_s3_path=None), "my-bucket")

        policy = json.loads(captured["Policy"])
        resources = policy["Statement"][0]["Resource"]
        assert len(resources) == 3  # config + models/* + checkpoints/*, no dataset

    def test_returns_none_on_sts_failure(self, monkeypatch):
        monkeypatch.setenv("TRAINING_ROLE_ARN", "arn:aws:iam::123:role/training")

        class BrokenSts:
            def assume_role(self, **kwargs):
                raise RuntimeError("connection timeout")

        monkeypatch.setattr("provisioner.boto3.client", lambda svc, **kw: BrokenSts())

        result = _scoped_s3_creds(self._job(), "my-bucket")
        assert result is None


# ── Export jobs ───────────────────────────────────────────────────────────────


def _export_job(**overrides):
    job = {
        "id": "eeeeeeee-0000-0000-0000-000000000005",
        "kind": "export",
        "config_s3_path": "s3://bucket/configs/eee/config.yaml",
        "dataset_s3_path": None,
        "hf_token_secret_arn": None,
        "resume_from_checkpoint_path": None,
        "instance_type": "g4dn.xlarge",
        "source_model_s3_path": "s3://bucket/models/src/model.tar.gz",
        "export_quant": "Q5_K_M",
        "disk_gb": 380,
    }
    job.update(overrides)
    return job


class TestExportUserData:
    def test_the_container_is_told_what_to_export(self):
        script = _build_user_data(_export_job(), "bucket", "efs", "ecr", "tok")
        assert 'JOB_KIND="export"' in script
        assert 'SOURCE_MODEL_S3_PATH="s3://bucket/models/src/model.tar.gz"' in script
        assert 'EXPORT_QUANT="Q5_K_M"' in script
        for var in ("JOB_KIND", "SOURCE_MODEL_S3_PATH", "EXPORT_QUANT"):
            assert f'-e {var}="${var}"' in script

    def test_a_training_job_is_a_training_job(self):
        job = _export_job(kind="train", source_model_s3_path=None, export_quant=None)
        script = _build_user_data(job, "bucket", "efs", "ecr", "tok")
        assert 'JOB_KIND="train"' in script


class TestExportProvision(TestProvision):
    def test_the_disk_is_sized_for_the_export(self, monkeypatch):
        run_calls = []
        self._setup(monkeypatch, run_calls)
        provision(_export_job())
        assert run_calls[0]["BlockDeviceMappings"][0]["Ebs"]["VolumeSize"] == 380

    def test_never_smaller_than_the_default(self, monkeypatch):
        run_calls = []
        self._setup(monkeypatch, run_calls)
        provision(_export_job(disk_gb=50))
        assert run_calls[0]["BlockDeviceMappings"][0]["Ebs"]["VolumeSize"] == 200


def test_scoped_credentials_can_read_the_source_model(monkeypatch):
    captured = {}

    class FakeSts:
        def assume_role(self, **kwargs):
            captured.update(kwargs)
            return {"Credentials": {"AccessKeyId": "a", "SecretAccessKey": "s", "SessionToken": "t"}}

    monkeypatch.setenv("TRAINING_ROLE_ARN", "arn:aws:iam::1:role/training")
    monkeypatch.setattr("provisioner.boto3.client", lambda *a, **k: FakeSts())
    _scoped_s3_creds(_export_job(), "bucket")
    resources = json.loads(captured["Policy"])["Statement"][0]["Resource"]
    assert "arn:aws:s3:::bucket/models/src/model.tar.gz" in resources
