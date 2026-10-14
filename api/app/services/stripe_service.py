import logging

import stripe

from ..config import settings

logger = logging.getLogger(__name__)

stripe.api_key = settings.stripe_secret_key

CREDIT_PACKAGES = [
    {"id": "standard", "price_cents": 5000,  "credits": 5500,  "label": "$50",  "bonus_pct": 10},
    {"id": "pro",      "price_cents": 20000, "credits": 24000, "label": "$200", "bonus_pct": 20},
]

PACKAGES_BY_ID = {p["id"]: p for p in CREDIT_PACKAGES}

# Base rate: 100 credits per dollar (1 credit = $0.01)
CREDITS_PER_DOLLAR = 100

# Stripe enforces a $0.50 minimum charge
_STRIPE_MIN_CHARGE_CENTS = 50


def _success_url() -> str:
    # Stripe replaces {CHECKOUT_SESSION_ID} with the real session ID on redirect
    return f"{settings.frontend_url}/credits?success=1&session_id={{CHECKOUT_SESSION_ID}}"


def create_checkout_session(user_id: str, package_id: str) -> str:
    pkg = PACKAGES_BY_ID.get(package_id)
    if not pkg:
        raise ValueError(f"Unknown package: {package_id}")

    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[{
            "price_data": {
                "currency": "usd",
                "unit_amount": pkg["price_cents"],
                "product_data": {
                    "name": f"{pkg['credits']:,} credits",
                },
            },
            "quantity": 1,
        }],
        mode="payment",
        client_reference_id=user_id,
        metadata={"credits": str(pkg["credits"]), "package_id": package_id},
        payment_intent_data={"setup_future_usage": "off_session"},
        success_url=_success_url(),
        cancel_url=f"{settings.frontend_url}/credits",
    )
    return session.url


def create_custom_checkout_session(user_id: str, amount_cents: int) -> str:
    credits = amount_cents  # 100 credits per dollar (1 credit = $0.01)
    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[{
            "price_data": {
                "currency": "usd",
                "unit_amount": amount_cents,
                "product_data": {
                    "name": f"{credits:,} credits",
                },
            },
            "quantity": 1,
        }],
        mode="payment",
        client_reference_id=user_id,
        metadata={"credits": str(credits)},
        payment_intent_data={"setup_future_usage": "off_session"},
        success_url=_success_url(),
        cancel_url=f"{settings.frontend_url}/credits",
    )
    return session.url


def retrieve_session(session_id: str) -> stripe.checkout.Session:
    return stripe.checkout.Session.retrieve(session_id)


def construct_webhook_event(payload: bytes, sig_header: str) -> stripe.Event:
    return stripe.Webhook.construct_event(payload, sig_header, settings.stripe_webhook_secret)


def ensure_stripe_customer(user_id: str, email: str) -> str:
    """Get or create a Stripe Customer for this user. Returns stripe_customer_id."""
    existing = stripe.Customer.search(query=f'metadata["user_id"]:"{user_id}"', limit=1)
    if existing.data:
        return existing.data[0].id
    customer = stripe.Customer.create(
        email=email,
        metadata={"user_id": user_id},
    )
    return customer.id


def charge_overage(
    stripe_customer_id: str,
    payment_method_id: str,
    amount_cents: int,
    job_id: str,
) -> stripe.PaymentIntent | None:
    """Charge a saved card off-session to recover a negative credit balance.

    Returns the PaymentIntent on success, None if the amount is below Stripe's minimum.
    Raises stripe.StripeError on payment failure — caller must handle and leave balance negative.
    """
    if amount_cents < _STRIPE_MIN_CHARGE_CENTS:
        return None
    return stripe.PaymentIntent.create(
        amount=amount_cents,
        currency="usd",
        customer=stripe_customer_id,
        payment_method=payment_method_id,
        confirm=True,
        off_session=True,
        metadata={"job_id": job_id, "reason": "overage"},
    )
