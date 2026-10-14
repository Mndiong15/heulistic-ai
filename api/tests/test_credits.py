import uuid
from datetime import datetime

from sqlalchemy import select

from app.models.db import CreditTransaction, User

from .conftest import SESSION_ID


class FakeStripeSession:
    def __init__(self, user_id, credits, payment_intent_id="pi_test", payment_status="paid"):
        self.payment_status = payment_status
        self.client_reference_id = user_id
        self.metadata = {"credits": str(credits)}
        self.payment_intent = payment_intent_id


# ── GET /api/credits ──────────────────────────────────────────────────────────

class TestGetCredits:
    async def test_returns_balance_and_packages(self, authed_client):
        resp = await authed_client.get("/api/credits")
        assert resp.status_code == 200
        data = resp.json()
        assert data["balance_cents"] == 0
        assert data["balance_display"] == "$0.00"
        assert len(data["packages"]) > 0
        assert "id" in data["packages"][0]

    async def test_payment_past_due_false_by_default(self, authed_client):
        resp = await authed_client.get("/api/credits")
        assert resp.status_code == 200
        assert resp.json()["payment_past_due"] is False

    async def test_payment_past_due_true_when_flagged(self, db_session, anon_client):
        user = User(id=SESSION_ID, email="test@local.dev", payment_past_due=True)
        db_session.add(user)
        await db_session.commit()

        from app.main import app
        from app.dependencies import get_db, get_current_user

        async def _override_db():
            yield db_session

        async def _override_user():
            return user

        app.dependency_overrides[get_db] = _override_db
        app.dependency_overrides[get_current_user] = _override_user
        resp = await anon_client.get("/api/credits")
        app.dependency_overrides.pop(get_current_user, None)

        assert resp.status_code == 200
        assert resp.json()["payment_past_due"] is True

    async def test_requires_auth(self, anon_client):
        resp = await anon_client.get("/api/credits")
        assert resp.status_code == 401


# ── GET /api/credits/transactions ─────────────────────────────────────────────

class TestListTransactions:
    async def test_empty_list(self, authed_client):
        resp = await authed_client.get("/api/credits/transactions")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_returns_transactions_newest_first(self, authed_client, db_session):
        db_session.add(CreditTransaction(
            user_id=SESSION_ID,
            amount_cents=1000,
            description="first",
            created_at=datetime(2024, 1, 1),
        ))
        db_session.add(CreditTransaction(
            user_id=SESSION_ID,
            amount_cents=2000,
            description="second",
            created_at=datetime(2024, 1, 2),
        ))
        await db_session.commit()

        resp = await authed_client.get("/api/credits/transactions")
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 2
        assert items[0]["description"] == "second"
        assert items[1]["description"] == "first"

    async def test_scoped_to_user(self, authed_client, db_session):
        other_id = str(uuid.uuid4())
        db_session.add(User(id=other_id, email="other@local.dev"))
        await db_session.flush()  # ensure User is inserted before the FK-dependent transaction
        db_session.add(CreditTransaction(
            user_id=other_id, amount_cents=500, description="other user"
        ))
        await db_session.commit()

        resp = await authed_client.get("/api/credits/transactions")
        assert resp.status_code == 200
        assert resp.json() == []


# ── POST /api/credits/checkout ────────────────────────────────────────────────

class TestCreateCheckout:
    async def test_valid_package_id(self, authed_client, monkeypatch):
        monkeypatch.setattr(
            "app.routers.credits.create_checkout_session",
            lambda user_id, pkg_id: "https://checkout.stripe.com/test",
        )
        resp = await authed_client.post("/api/credits/checkout", json={"package_id": "standard"})
        assert resp.status_code == 200
        assert resp.json()["checkout_url"] == "https://checkout.stripe.com/test"

    async def test_invalid_package_id(self, authed_client):
        resp = await authed_client.post("/api/credits/checkout", json={"package_id": "nonexistent"})
        assert resp.status_code == 422

    async def test_no_package_no_custom_amount(self, authed_client):
        resp = await authed_client.post("/api/credits/checkout", json={})
        assert resp.status_code == 422

    async def test_stripe_error_returns_503(self, authed_client, monkeypatch):
        def _raise(*_args, **_kwargs):
            raise Exception("Stripe service unavailable")

        monkeypatch.setattr("app.routers.credits.create_checkout_session", _raise)
        resp = await authed_client.post("/api/credits/checkout", json={"package_id": "standard"})
        assert resp.status_code == 503


# ── POST /api/credits/checkout/confirm ───────────────────────────────────────

class TestConfirmCheckout:
    async def test_applies_credits(self, authed_client, db_session, monkeypatch):
        session = FakeStripeSession(SESSION_ID, credits=5500, payment_intent_id="pi_abc")
        monkeypatch.setattr("app.routers.credits.retrieve_session", lambda sid: session)

        resp = await authed_client.post("/api/credits/checkout/confirm?session_id=cs_test")
        assert resp.status_code == 200
        data = resp.json()
        assert data["credits_added"] == 5500
        assert data["balance_cents"] == 5500

    async def test_idempotent(self, authed_client, db_session, monkeypatch):
        session = FakeStripeSession(SESSION_ID, credits=5500, payment_intent_id="pi_dup")
        monkeypatch.setattr("app.routers.credits.retrieve_session", lambda sid: session)

        await authed_client.post("/api/credits/checkout/confirm?session_id=cs_test")
        resp = await authed_client.post("/api/credits/checkout/confirm?session_id=cs_test")

        assert resp.status_code == 200
        assert resp.json()["credits_added"] == 0
        assert resp.json()["balance_cents"] == 5500

    async def test_wrong_user(self, authed_client, monkeypatch):
        session = FakeStripeSession("wrong-user-id", credits=5500, payment_intent_id="pi_xyz")
        monkeypatch.setattr("app.routers.credits.retrieve_session", lambda sid: session)
        resp = await authed_client.post("/api/credits/checkout/confirm?session_id=cs_test")
        assert resp.status_code == 403

    async def test_not_paid(self, authed_client, monkeypatch):
        session = FakeStripeSession(
            SESSION_ID, credits=5500, payment_intent_id="pi_np", payment_status="unpaid"
        )
        monkeypatch.setattr("app.routers.credits.retrieve_session", lambda sid: session)
        resp = await authed_client.post("/api/credits/checkout/confirm?session_id=cs_test")
        assert resp.status_code == 402
