import json
import logging

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..dependencies import get_db
from ..models.db import CreditTransaction, User
from ..services.stripe_service import (
    construct_webhook_event,
    ensure_stripe_customer,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["webhooks"])


@router.post("/stripe/webhook")
async def stripe_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    payload = await request.body()

    if settings.local_mode:
        event = json.loads(payload)
    else:
        sig_header = request.headers.get("stripe-signature", "")
        try:
            event = construct_webhook_event(payload, sig_header)
        except (stripe.error.SignatureVerificationError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Invalid webhook signature") from exc

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_id: str = session.get("client_reference_id", "")
        credits: int = int(session.get("metadata", {}).get("credits", 0))
        payment_intent_id: str = session.get("payment_intent", "")

        if user_id and credits > 0:
            # Idempotency: skip if this payment_intent was already processed
            if payment_intent_id:
                existing = await db.execute(
                    select(CreditTransaction).where(
                        CreditTransaction.stripe_payment_intent_id == payment_intent_id
                    )
                )
                if existing.scalar_one_or_none() is not None:
                    return {"received": True}

            result = await db.execute(
                select(User).where(User.id == user_id).with_for_update()
            )
            user = result.scalar_one_or_none()
            if user:
                user.credit_balance_cents += credits
                user.payment_past_due = False  # successful purchase = card works
                db.add(CreditTransaction(
                    user_id=user_id,
                    amount_cents=credits,
                    description=f"Purchased {credits} credits",
                    stripe_payment_intent_id=payment_intent_id,
                ))

                # Ensure Stripe Customer exists and capture the payment method for future off-session charges
                try:
                    if not user.stripe_customer_id:
                        user.stripe_customer_id = ensure_stripe_customer(user_id, user.email)
                    if payment_intent_id and not user.stripe_payment_method_id:
                        pi = stripe.PaymentIntent.retrieve(payment_intent_id)
                        if pi.payment_method:
                            user.stripe_payment_method_id = pi.payment_method
                except Exception:
                    logger.exception(
                        "Stripe customer/payment method update failed for user %s",
                        user_id,
                    )

                await db.commit()

    elif event["type"] == "payment_intent.payment_failed":
        pi = event["data"]["object"]
        if pi.get("metadata", {}).get("reason") == "overage":
            customer_id: str = pi.get("customer", "")
            if customer_id:
                result = await db.execute(
                    select(User).where(User.stripe_customer_id == customer_id).with_for_update()
                )
                user = result.scalar_one_or_none()
                if user:
                    user.payment_past_due = True
                    await db.commit()
                    logger.warning(
                        "Overage charge failed for user %s (customer %s) — account flagged payment_past_due",
                        user.id, customer_id,
                    )

    return {"received": True}
