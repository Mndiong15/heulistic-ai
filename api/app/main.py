import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .config import settings
from .extensions import load_cloud_extensions
from .routers import admin, credits, estimate, instances, internal, jobs, uploads, webhooks
from .routers import validate as validate_router
from .routers import local_download as local_download_router

if settings.sentry_dsn:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            integrations=[FastApiIntegration(), SqlalchemyIntegration()],
            traces_sample_rate=0.1,
        )
    except ImportError:
        pass

# Room for the largest dataset that may pass through the API, plus the config
# and form fields alongside it. Larger datasets go straight to S3.
_MAX_BODY_BYTES = (settings.max_proxied_upload_mb + 64) * 1024 * 1024


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > _MAX_BODY_BYTES:
            return JSONResponse(status_code=413, content={"detail": "Request body too large"})
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.local_mode and settings.stripe_secret_key.startswith("sk_live_"):
        raise RuntimeError("LOCAL_MODE must not be enabled with a live Stripe key")
    alembic = Path(sys.executable).parent / "alembic"
    result = subprocess.run(
        [str(alembic), "upgrade", "head"],
        capture_output=True,
        text=True,
    )
    sys.stdout.write(result.stdout)
    if result.returncode != 0:
        if "already exists" in result.stderr:
            # Tables were created outside Alembic (e.g. Base.metadata.create_all in tests).
            # Stamp to the current head so subsequent upgrades are no-ops.
            subprocess.run([str(alembic), "stamp", "head"], check=True)
            subprocess.run([str(alembic), "upgrade", "head"], check=True)
        else:
            sys.stderr.write(result.stderr)
            raise RuntimeError("alembic upgrade head failed")
    yield


app = FastAPI(title="Fine-Tuning Platform API", lifespan=lifespan)

app.add_middleware(BodySizeLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(admin.router, prefix="/api")
app.include_router(jobs.router, prefix="/api")
app.include_router(estimate.router, prefix="/api")
app.include_router(validate_router.router, prefix="/api")
app.include_router(instances.router, prefix="/api")
app.include_router(credits.router, prefix="/api")
app.include_router(uploads.router, prefix="/api")
app.include_router(webhooks.router, prefix="/api")
app.include_router(internal.router, prefix="/internal")

if settings.local_mode:
    app.include_router(local_download_router.router)

# Heulistic Cloud's own features, from app/cloud_only — absent in the
# open-source build, where this registers nothing.
cloud_features = load_cloud_extensions(app)


@app.get("/health")
async def health():
    return {"status": "ok"}
