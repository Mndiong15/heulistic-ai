"""Cloud-only features plug into the shared API through one hook.

Heulistic Cloud and the open-source product share this API. A feature only the
managed platform offers lives in `app/cloud_only/`, which the open-source
export deletes; `load_cloud_extensions` registers it when it is there and does
nothing when it is not.
"""

import types

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from app import extensions


def _module_with(register):
    module = types.ModuleType("app.cloud_only")
    module.register = register
    return module


def test_without_cloud_features_nothing_is_registered(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(f"No module named {name!r}", name="app.cloud_only")

    monkeypatch.setattr(extensions.importlib, "import_module", missing)
    app = FastAPI()
    before = len(app.routes)
    assert extensions.load_cloud_extensions(app) == []
    assert len(app.routes) == before


def test_cloud_features_register_their_routes(monkeypatch):
    def register(app):
        router = APIRouter()

        @router.get("/api/cloud-feature")
        def feature():
            return {"cloud": True}

        app.include_router(router)
        return ["cloud-feature"]

    monkeypatch.setattr(extensions.importlib, "import_module", lambda name: _module_with(register))
    app = FastAPI()
    assert extensions.load_cloud_extensions(app) == ["cloud-feature"]
    assert TestClient(app).get("/api/cloud-feature").json() == {"cloud": True}


def test_a_broken_cloud_feature_fails_loudly(monkeypatch):
    """Present but broken must not look like "absent" — the cloud would quietly
    lose a paid feature."""
    def broken(name):
        raise ModuleNotFoundError("No module named 'stripe_v2'", name="stripe_v2")

    monkeypatch.setattr(extensions.importlib, "import_module", broken)
    with pytest.raises(ModuleNotFoundError, match="stripe_v2"):
        extensions.load_cloud_extensions(FastAPI())


def test_the_api_loads_them_at_startup():
    # In this repository app/cloud_only exists and registers nothing yet; the
    # point is that main.py ran the hook, so the first feature only needs
    # adding there.
    from app import main
    assert main.cloud_features == []
