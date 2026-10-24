"""The one place cloud-only features join the shared API.

Heulistic Cloud and the open-source product run this same API. A feature only
the managed platform offers goes in `app/cloud_only/`, which
scripts/export_public.py deletes from the open-source repository. Nothing else
may import from it — the export refuses if anything does — so the public API
simply runs without those features.
"""

import importlib

from fastapi import FastAPI

_PACKAGE = "app.cloud_only"


def load_cloud_extensions(app: FastAPI) -> list[str]:
    """Let app/cloud_only register its routes; return the names of what it added.

    Absent (the open-source build): nothing happens. Present but broken: the
    error propagates, rather than the cloud silently losing a paid feature.
    """
    try:
        module = importlib.import_module(_PACKAGE)
    except ModuleNotFoundError as exc:
        if exc.name == _PACKAGE:
            return []
        raise
    return list(module.register(app) or [])
