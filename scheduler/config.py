from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    local_mode: bool = False

    @property
    def db_url(self) -> str:
        """psycopg2-compatible URL — strips +asyncpg driver prefix if present."""
        return self.database_url.replace("postgresql+asyncpg://", "postgresql://")

    aws_region: str = "us-east-1"
    project_prefix: str = "ftp-prod"
    api_internal_url: str = "http://localhost:8000"

    poll_interval_seconds: int = 10
    watchdog_interval_seconds: int = 60
    heartbeat_timeout_minutes: int = 10
    # Cloud capacity: the EC2 G/VT on-demand quota, in vCPUs. Jobs are started
    # while their instances fit in it — raise it after AWS raises the quota.
    vcpu_quota: int = 64
    # While other users are waiting, one user may hold at most this fraction
    # of the quota. With nobody else waiting they may use all of it.
    user_vcpu_share: float = 0.5
    # Local capacity, in jobs. 0 means "decide from local_mode" — see below.
    # Not used in the cloud, where vcpu_quota is the limit.
    max_concurrent_jobs: int = 0
    # Local GPU sharing, set by `heulistic start --gpus-per-job`. 0 gives each
    # job every GPU, so jobs run one at a time.
    gpus_per_job: int = 0
    local_gpu_count: int = 0

    @model_validator(mode="after")
    def _resolve_concurrency(self) -> "Settings":
        """Local capacity, in jobs. (The cloud is bounded by vcpu_quota.)

        By default every local job is handed all of the machine's GPUs, so two
        at once would fight over the same cards: one job at a time. With
        `gpus_per_job` set, each job is pinned to that many cards and as many
        run as the machine has room for.

        A real MAX_CONCURRENT_JOBS always wins. `model_fields_set` cannot be
        used to detect that: pydantic-settings marks every field as set once a
        .env file is read.
        """
        if self.max_concurrent_jobs == 0:
            if self.local_mode and 0 < self.gpus_per_job <= self.local_gpu_count:
                self.max_concurrent_jobs = self.local_gpu_count // self.gpus_per_job
            else:
                self.max_concurrent_jobs = 1
        return self

    # Unknown keys are ignored so a .env written for an older release — e.g.
    # one still setting MAX_JOBS_PER_SESSION — does not stop the scheduler.
    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
