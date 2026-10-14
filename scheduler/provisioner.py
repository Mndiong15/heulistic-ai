"""
Provisions an EC2 GPU instance for a training job.

The user data script:
  1. Mounts EFS at /efs
  2. Logs into ECR and pulls the training image
  3. Runs the container with job-specific env vars
  4. Self-terminates the instance on container exit (success or failure)
"""

import hashlib
import json
import logging
import os
import secrets
import textwrap

import boto3
from botocore.config import Config

from config import settings

logger = logging.getLogger(__name__)


# Root volume: the DLAMI default (~50 GB) is too small to pull the training
# image. Exports ask for more — sized by the API to the model being exported.
_DEFAULT_DISK_GB = 200


def _ssm(name: str) -> str:
    client = boto3.client("ssm", region_name=settings.aws_region)
    return client.get_parameter(Name=name)["Parameter"]["Value"]


def _latest_dlami(ec2) -> str:
    """Return the latest Deep Learning OSS AMI (Amazon Linux 2023, GPU, PyTorch)."""
    resp = ec2.describe_images(
        Owners=["amazon"],
        Filters=[
            {
                "Name": "name",
                "Values": ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Amazon Linux 2023) *"],
            },
            {"Name": "state", "Values": ["available"]},
        ],
    )
    images = sorted(resp["Images"], key=lambda x: x["CreationDate"], reverse=True)
    if not images:
        raise RuntimeError("No Deep Learning AMI found in this region")
    ami = images[0]["ImageId"]
    logger.info("Using DLAMI %s (%s)", ami, images[0]["Name"])
    return ami


def _scoped_s3_creds(job: dict, s3_bucket: str) -> dict | None:
    """Return temporary S3 credentials scoped to this job's paths.

    Only active when TRAINING_ROLE_ARN is set. Requires the scheduler's IAM role to have
    sts:AssumeRole permission on that role. The role itself can have broad S3 access — the
    session policy applied here narrows it to just this job's objects.
    """
    role_arn = os.environ.get("TRAINING_ROLE_ARN")
    if not role_arn:
        return None

    job_id = str(job["id"])
    config_key = (job.get("config_s3_path") or "").replace(f"s3://{s3_bucket}/", "")
    dataset_key = (job.get("dataset_s3_path") or "").replace(f"s3://{s3_bucket}/", "")

    resources = [
        f"arn:aws:s3:::{s3_bucket}/{config_key}",
        f"arn:aws:s3:::{s3_bucket}/models/{job_id}/*",
        f"arn:aws:s3:::{s3_bucket}/checkpoints/{job_id}/*",
    ]
    if dataset_key:
        resources.append(f"arn:aws:s3:::{s3_bucket}/{dataset_key}")
    # An export reads the model of the job it exports.
    source_key = (job.get("source_model_s3_path") or "").replace(f"s3://{s3_bucket}/", "")
    if source_key:
        resources.append(f"arn:aws:s3:::{s3_bucket}/{source_key}")

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:HeadObject"],
                "Resource": resources,
            }
        ],
    }

    try:
        sts = boto3.client("sts", region_name=settings.aws_region)
        resp = sts.assume_role(
            RoleArn=role_arn,
            RoleSessionName=f"heulistic-job-{job_id[:16]}",
            Policy=json.dumps(policy),
            DurationSeconds=43200,  # 12h
        )
        c = resp["Credentials"]
        logger.info("Job %s: issued scoped S3 credentials via STS", job_id)
        return {
            "access_key": c["AccessKeyId"],
            "secret_key": c["SecretAccessKey"],
            "session_token": c["SessionToken"],
        }
    except Exception:
        logger.warning("Job %s: STS AssumeRole failed — falling back to instance profile", job_id, exc_info=True)
        return None


def _build_user_data(job: dict, s3_bucket: str, efs_dns: str, ecr_url: str, token: str) -> str:
    job_id = str(job["id"])
    config_path = job.get("config_s3_path") or f"s3://{s3_bucket}/configs/{job_id}/config.yaml"
    dataset_path = job.get("dataset_s3_path") or ""
    hf_token_arn = job.get("hf_token_secret_arn") or ""
    resume_path = job.get("resume_from_checkpoint_path") or ""
    instance_type = job.get("instance_type") or "g5.xlarge"
    job_kind = job.get("kind") or "train"
    source_model = job.get("source_model_s3_path") or ""
    export_quant = job.get("export_quant") or ""

    s3_creds = _scoped_s3_creds(job, s3_bucket)
    scoped_creds_block = ""
    if s3_creds:
        scoped_creds_block = textwrap.dedent(f"""\
        AWS_ACCESS_KEY_ID="{s3_creds['access_key']}"
        AWS_SECRET_ACCESS_KEY="{s3_creds['secret_key']}"
        AWS_SESSION_TOKEN="{s3_creds['session_token']}"
        """)

    script = textwrap.dedent(f"""\
        #!/bin/bash
        set -euo pipefail
        exec > >(tee /var/log/training-bootstrap.log /dev/console) 2>&1

        JOB_ID="{job_id}"
        API_URL="{settings.api_internal_url}"
        REGION="{settings.aws_region}"
        S3_BUCKET="{s3_bucket}"
        EFS_DNS="{efs_dns}"
        ECR_URL="{ecr_url}"
        CONFIG_S3_PATH="{config_path}"
        DATASET_S3_PATH="{dataset_path}"
        HF_TOKEN_SECRET_ARN="{hf_token_arn}"
        RESUME_FROM_CHECKPOINT_PATH="{resume_path}"
        INSTANCE_TYPE="{instance_type}"
        INTERNAL_TOKEN="{token}"
        JOB_KIND="{job_kind}"
        SOURCE_MODEL_S3_PATH="{source_model}"
        EXPORT_QUANT="{export_quant}"
        {scoped_creds_block}
        # On any exit, flush bootstrap log to the job's log stream so it's visible in the UI
        _flush_bootstrap_logs() {{
          set +e
          [ -f /var/log/training-bootstrap.log ] || return 0
          python3 -c "import json;lines=['[BOOTSTRAP] '+l.rstrip() for l in open('/var/log/training-bootstrap.log') if l.strip()];print(json.dumps({{'lines':lines}}))" \
            | curl -sf -X POST "$API_URL/internal/jobs/$JOB_ID/logs" \
                -H "Content-Type: application/json" \
                -H "X-Internal-Token: $INTERNAL_TOKEN" \
                -d @- 2>/dev/null || true
        }}
        trap _flush_bootstrap_logs EXIT

        # IMDSv2 — account enforces HttpTokens=required so a token is mandatory
        IMDS_TOKEN=$(curl -sfX PUT "http://169.254.169.254/latest/api/token" \
          -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
        INSTANCE_ID=$(curl -sf -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" \
          http://169.254.169.254/latest/meta-data/instance-id)

        # Mount EFS (NFS4)
        mkdir -p /efs
        mount -t nfs4 \\
          -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 \\
          "${{EFS_DNS}}":/ /efs

        # ECR login + pull
        aws ecr get-login-password --region "$REGION" | \\
          docker login --username AWS --password-stdin "$ECR_URL"
        docker pull "${{ECR_URL}}:latest"

        # Run training — container reports logs/metrics/completion to the API itself
        docker run --rm --gpus all \\
          -e JOB_ID="$JOB_ID" \\
          -e API_URL="$API_URL" \\
          -e S3_BUCKET="$S3_BUCKET" \\
          -e CONFIG_S3_PATH="$CONFIG_S3_PATH" \\
          -e DATASET_S3_PATH="$DATASET_S3_PATH" \\
          -e HF_TOKEN_SECRET_ARN="$HF_TOKEN_SECRET_ARN" \\
          -e RESUME_FROM_CHECKPOINT_PATH="$RESUME_FROM_CHECKPOINT_PATH" \\
          -e INSTANCE_TYPE="$INSTANCE_TYPE" \\
          -e AWS_REGION="$REGION" \\
          -e INSTANCE_ID="$INSTANCE_ID" \\
          -e INTERNAL_TOKEN="$INTERNAL_TOKEN" \\
          -e JOB_KIND="$JOB_KIND" \\
          -e SOURCE_MODEL_S3_PATH="$SOURCE_MODEL_S3_PATH" \\
          -e EXPORT_QUANT="$EXPORT_QUANT" \\
          $( [ -n "$AWS_ACCESS_KEY_ID" ] && echo '-e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" -e AWS_SESSION_TOKEN="$AWS_SESSION_TOKEN"' ) \\
          -v /efs:/efs \\
          "${{ECR_URL}}:latest"

        # Trigger instance self-termination
        shutdown -h now
    """)
    return script


def provision(job: dict) -> tuple[str, str]:
    """
    Launch a GPU instance for the given job.
    In local mode, runs a Docker container; otherwise launches EC2.
    Returns (instance_id, token_hash) — token_hash is stored in the DB so the
    API can validate internal requests from the training container.
    """
    if settings.local_mode:
        from local_provisioner import provision as local_provision
        return local_provision(job)

    token = secrets.token_hex(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    prefix = settings.project_prefix

    # Read infra config from SSM (set by Terraform)
    sg_id = _ssm(f"/{prefix}/ec2/training-sg-id")
    subnet_ids = _ssm(f"/{prefix}/ec2/training-subnet-ids").split(",")
    instance_profile = _ssm(f"/{prefix}/ec2/training-instance-profile")
    s3_bucket = _ssm(f"/{prefix}/s3/bucket")
    efs_dns = _ssm(f"/{prefix}/efs/dns-name")
    ecr_url = _ssm(f"/{prefix}/ecr/training-url")

    ec2 = boto3.client("ec2", region_name=settings.aws_region)
    ami_id = _latest_dlami(ec2)
    user_data = _build_user_data(job, s3_bucket, efs_dns, ecr_url, token)
    instance_type = job.get("instance_type") or "g5.xlarge"
    job_id = str(job["id"])

    resp = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=instance_type,
        MinCount=1,
        MaxCount=1,
        SecurityGroupIds=[sg_id],
        SubnetId=subnet_ids[0],
        IamInstanceProfile={"Name": instance_profile},
        UserData=user_data,
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/xvda",
                "Ebs": {
                    "VolumeSize": max(_DEFAULT_DISK_GB, job.get("disk_gb") or 0),
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            }
        ],
        # Instance shuts itself down via `shutdown -h now` after training
        InstanceInitiatedShutdownBehavior="terminate",
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"{prefix}-training-{job_id[:8]}"},
                    {"Key": "JobId", "Value": job_id},
                    {"Key": "ManagedBy", "Value": "ftp-scheduler"},
                ],
            }
        ],
    )
    instance_id = resp["Instances"][0]["InstanceId"]
    logger.info("Job %s: launched %s (%s)", job_id, instance_id, instance_type)
    return instance_id, token_hash


def terminate(instance_id: str) -> None:
    if settings.local_mode:
        from local_provisioner import terminate as local_terminate
        local_terminate(instance_id)
        return
    ec2 = boto3.client("ec2", region_name=settings.aws_region)
    ec2.terminate_instances(InstanceIds=[instance_id])
    logger.info("Terminated instance %s", instance_id)
