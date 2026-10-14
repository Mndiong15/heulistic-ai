import json

from sqlalchemy import select

from app.models.db import CreditTransaction, User

from .conftest import SESSION_ID, OTHER_SESSION_ID


def _checkout_payload(user_id: str, credits: int, payment_intent_id: str = "pi_test") -> bytes:
    return json.dumps({
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "client_reference_id": user_id,
                "metadata": {"credits": str(credits)},
                "payment_intent": payment_intent_id,
            }
        }
    }).encode()


def _payment_failed_payload(customer_id: str, payment_intent_id: str = "pi_fail", reason: str = "overage") -> bytes:
    return json.dumps({
        "type": "payment_intent.payment_failed",
        "data": {
            "object": {
                "id": payment_intent_id,
                "customer": customer_id,
                "metadata": {"reason": reason},
            }
        }
    }).encode()


_JSON = {"Content-Type": "application/json"}


class TestCheckoutSessionCompleted:
    async def test_applies_credits_on_first_event(self, anon_client, db_session):
        user = User(id=SESSION_ID, email="test@local.dev")
        db_session.add(user)
        await db_session.commit()

        resp = await anon_client.post(
            "/api/stripe/webhook",
            content=_checkout_payload(SESSION_ID, 5500, "pi_first"),
            headers=_JSON,
        )
        assert resp.status_code == 200

        await db_session.refresh(user)
        assert user.credit_balance_cents == 5500

        result = await db_session.execute(
            select(CreditTransaction).where(
                CreditTransaction.stripe_payment_intent_id == "pi_first"
            )
        )
        tx = result.scalar_one()
        assert tx.amount_cents == 5500
        assert tx.user_id == SESSION_ID

    async def test_idempotent_on_retry(self, anon_client, db_session):
        user = User(id=SESSION_ID, email="test@local.dev")
        db_session.add(user)
        await db_session.commit()

        payload = _checkout_payload(SESSION_ID, 5500, "pi_retry")
        await anon_client.post("/api/stripe/webhook", content=payload, headers=_JSON)
        resp = await anon_client.post("/api/stripe/webhook", content=payload, headers=_JSON)

        assert resp.status_code == 200
        await db_session.refresh(user)
        assert user.credit_balance_cents == 5500  # not doubled

        result = await db_session.execute(
            select(CreditTransaction).where(
                CreditTransaction.stripe_payment_intent_id == "pi_retry"
            )
        )
        assert len(result.scalars().all()) == 1

    async def test_unknown_user_silently_ignored(self, anon_client):
        resp = await anon_client.post(
            "/api/stripe/webhook",
            content=_checkout_payload("nonexistent-user", 5500, "pi_unk"),
            headers=_JSON,
        )
        assert resp.status_code == 200
        assert resp.json() == {"received": True}

    async def test_zero_credits_ignored(self, anon_client, db_session):
        user = User(id=SESSION_ID, email="test@local.dev")
        db_session.add(user)
        await db_session.commit()

        payload = _checkout_payload(SESSION_ID, 0, "pi_zero")
        resp = await anon_client.post("/api/stripe/webhook", content=payload, headers=_JSON)

        assert resp.status_code == 200
        await db_session.refresh(user)
        assert user.credit_balance_cents == 0

    async def test_clears_payment_past_due_flag(self, anon_client, db_session):
        user = User(id=SESSION_ID, email="test@local.dev", payment_past_due=True)
        db_session.add(user)
        await db_session.commit()

        await anon_client.post(
            "/api/stripe/webhook",
            content=_checkout_payload(SESSION_ID, 5500, "pi_clear"),
            headers=_JSON,
        )

        await db_session.refresh(user)
        assert user.payment_past_due is False

    async def test_captures_payment_method(self, anon_client, db_session, monkeypatch):
        user = User(id=SESSION_ID, email="test@local.dev")
        db_session.add(user)
        await db_session.commit()

        class _FakePI:
            payment_method = "pm_saved"

        monkeypatch.setattr(
            "app.routers.webhooks.ensure_stripe_customer",
            lambda uid, email: "cus_new",
        )
        monkeypatch.setattr(
            "app.routers.webhooks.stripe.PaymentIntent.retrieve",
            lambda pi_id: _FakePI(),
        )

        await anon_client.post(
            "/api/stripe/webhook",
            content=_checkout_payload(SESSION_ID, 1000, "pi_pm"),
            headers=_JSON,
        )

        await db_session.refresh(user)
        assert user.stripe_payment_method_id == "pm_saved"

    async def test_stripe_failure_does_not_rollback_credits(self, anon_client, db_session, monkeypatch):
        user = User(id=SESSION_ID, email="test@local.dev")
        db_session.add(user)
        await db_session.commit()

        monkeypatch.setattr(
            "app.routers.webhooks.ensure_stripe_customer",
            lambda uid, email: (_ for _ in ()).throw(RuntimeError("Stripe down")),
        )

        resp = await anon_client.post(
            "/api/stripe/webhook",
            content=_checkout_payload(SESSION_ID, 5500, "pi_stripe_fail"),
            headers=_JSON,
        )
        assert resp.status_code == 200

        await db_session.refresh(user)
        assert user.credit_balance_cents == 5500

    async def test_unknown_event_type_ignored(self, anon_client):
        payload = json.dumps({
            "type": "customer.created",
            "data": {"object": {}},
        }).encode()
        resp = await anon_client.post("/api/stripe/webhook", content=payload, headers=_JSON)
        assert resp.status_code == 200
        assert resp.json() == {"received": True}


class TestPaymentIntentFailed:
    async def test_sets_payment_past_due_on_overage_failure(self, anon_client, db_session):
        user = User(
            id=SESSION_ID,
            email="test@local.dev",
            stripe_customer_id="cus_abc",
            credit_balance_cents=-500,
        )
        db_session.add(user)
        await db_session.commit()

        resp = await anon_client.post(
            "/api/stripe/webhook",
            content=_payment_failed_payload("cus_abc"),
            headers=_JSON,
        )
        assert resp.status_code == 200

        await db_session.refresh(user)
        assert user.payment_past_due is True

    async def test_non_overage_failure_ignored(self, anon_client, db_session):
        user = User(
            id=SESSION_ID,
            email="test@local.dev",
            stripe_customer_id="cus_abc",
        )
        db_session.add(user)
        await db_session.commit()

        resp = await anon_client.post(
            "/api/stripe/webhook",
            content=_payment_failed_payload("cus_abc", reason="manual"),
            headers=_JSON,
        )
        assert resp.status_code == 200

        await db_session.refresh(user)
        assert user.payment_past_due is False

    async def test_unknown_customer_silently_ignored(self, anon_client):
        resp = await anon_client.post(
            "/api/stripe/webhook",
            content=_payment_failed_payload("cus_nonexistent"),
            headers=_JSON,
        )
        assert resp.status_code == 200
        assert resp.json() == {"received": True}

    async def test_no_customer_field_silently_ignored(self, anon_client):
        payload = json.dumps({
            "type": "payment_intent.payment_failed",
            "data": {
                "object": {
                    "id": "pi_no_cus",
                    "customer": "",
                    "metadata": {"reason": "overage"},
                }
            }
        }).encode()
        resp = await anon_client.post("/api/stripe/webhook", content=payload, headers=_JSON)
        assert resp.status_code == 200
