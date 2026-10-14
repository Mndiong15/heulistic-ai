from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    s3_bucket: str
    aws_region: str = "us-east-1"

    # How many jobs *run* at once is the scheduler's decision, bounded by the
    # EC2 GPU quota. Submit only guards against abuse: credits are reserved per
    # job, so a long queue cannot spend more than the balance holds.
    max_unfinished_jobs_per_user: int = 50
    max_submissions_per_hour: int = 100
    # Datasets reach S3 directly from the browser, so the cap is about the
    # training instance's disk (200 GB, shared with the model and checkpoints),
    # not the API. Uploads that do pass through the API are spooled to its disk
    # first — 30 GB on the cloud t3.small — so they get a lower cap there.
    # 0 means: the full dataset cap locally, 5 GB in the cloud.
    max_dataset_size_mb: int = 51200
    max_proxied_upload_mb: int = 0

    # Set LOCAL_MODE=true in .env.local to bypass AWS calls during local dev
    local_mode: bool = False

    # MinIO / local S3 — only used when local_mode=True
    s3_endpoint_url: str = ""
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"

    # Billing — auto-disabled in local mode; set BILLING_ENABLED=false for no-billing cloud deploys
    billing_enabled: bool = True

    # Cognito — required in production; ignored when local_mode=True
    cognito_user_pool_id: str = ""
    cognito_client_id: str = ""
    cognito_region: str = "us-east-1"

    # Stripe
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_meter_event_name: str = "finetuning_compute"
    stripe_metered_price_id: str = ""  # set via env in prod; empty = metering disabled
    frontend_url: str = "http://localhost:3000"
    # Where this API is reachable from a browser. Only used in local mode,
    # to build download links — so it must track --api-port.
    api_public_url: str = "http://localhost:8000"
    admin_emails: str = ""
    sentry_dsn: str = ""

    @model_validator(mode="after")
    def _no_local_mode_in_production(self) -> "Settings":
        if self.local_mode and self.stripe_secret_key.startswith("sk_live"):
            raise ValueError(
                "local_mode cannot be enabled with a live Stripe key — "
                "set LOCAL_MODE=false in your production environment"
            )
        if self.local_mode:
            self.billing_enabled = False
        if self.max_proxied_upload_mb == 0:
            self.max_proxied_upload_mb = (
                self.max_dataset_size_mb if self.local_mode
                else min(5120, self.max_dataset_size_mb)
            )
        return self

    @property
    def admin_email_set(self) -> set[str]:
        return {e.strip().lower() for e in self.admin_emails.split(",") if e.strip()}

    def is_admin(self, email: str) -> bool:
        return email.lower() in self.admin_email_set

    @property
    def cors_origins(self) -> list[str]:
        url = self.frontend_url.rstrip("/")
        origins = [url]
        if url.startswith("https://www."):
            origins.append("https://" + url[len("https://www."):])
        elif url.startswith("https://") and not url.startswith("https://www."):
            origins.append("https://www." + url[len("https://"):])
        return origins

    # Unknown keys are ignored so that a .env written for an older release —
    # e.g. one still setting MAX_JOBS_PER_SESSION — does not stop the API.
    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
