import stripe
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..dependencies import get_current_user, get_db
from ..models.db import CreditTransaction, User
from ..schemas.credits import (
    CheckoutRequest,
    CheckoutResponse,
    ConfirmResponse,
    CreditPackage,
    CreditsResponse,
    TransactionResponse,
)
from ..services.stripe_service import (
    CREDIT_PACKAGES,
    PACKAGES_BY_ID,
    create_checkout_session,
    create_custom_checkout_session,
    retrieve_session,
)

router = APIRouter(prefix="/credits", tags=["credits"])


def _make_package(pkg: dict) -> CreditPackage:
    return CreditPackage(
        id=pkg["id"],
        price_cents=pkg["price_cents"],
        credits=pkg["credits"],
        label=pkg["label"],
        bonus_pct=pkg["bonus_pct"],
        price_display=f"${pkg['price_cents'] / 100:.0f}",
        credits_display=f"{pkg['credits']:,} credits",
    )


@router.get("", response_model=CreditsResponse)
async def get_credits(
    user: User = Depends(get_current_user),
):
    if not settings.billing_enabled:
        return CreditsResponse(
            balance_cents=2**31,
            balance_display="Unlimited",
            packages=[],
            payment_past_due=False,
        )
    return CreditsResponse(
        balance_cents=user.credit_balance_cents,
        balance_display=f"${user.credit_balance_cents / 100:.2f}",
        packages=[_make_package(p) for p in CREDIT_PACKAGES],
        payment_past_due=user.payment_past_due,
    )


@router.get("/transactions", response_model=list[TransactionResponse])
async def list_transactions(
    limit: int = Query(50, ge=1, le=100),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(CreditTransaction)
        .where(CreditTransaction.user_id == user.id)
        .order_by(CreditTransaction.created_at.desc())
        .limit(limit)
    )
    return result.scalars().all()


@router.post("/checkout", response_model=CheckoutResponse)
async def create_checkout(
    req: CheckoutRequest,
    user: User = Depends(get_current_user),
):
    if not settings.billing_enabled:
        raise HTTPException(status_code=404, detail="Not found")
    if not req.package_id and not req.custom_amount_cents:
        raise HTTPException(status_code=422, detail="Provide either package_id or custom_amount_cents")

    try:
        if req.custom_amount_cents:
            checkout_url = create_custom_checkout_session(user.id, req.custom_amount_cents)
        else:
            if req.package_id not in PACKAGES_BY_ID:
                raise HTTPException(status_code=422, detail=f"Unknown package '{req.package_id}'")
            checkout_url = create_checkout_session(user.id, req.package_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Payment service unavailable") from exc

    return CheckoutResponse(checkout_url=checkout_url)


@router.post("/checkout/confirm", response_model=ConfirmResponse)
async def confirm_checkout(
    session_id: str = Query(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not settings.billing_enabled:
        raise HTTPException(status_code=404, detail="Not found")
    """
    Called by the frontend on redirect from Stripe checkout.
    Fetches the session, verifies it, and applies credits idempotently.
    Serves as a reliable fallback when webhooks are not delivered (e.g. local dev).
    """
    try:
        session = retrieve_session(session_id)
    except stripe.StripeError as exc:
        raise HTTPException(status_code=400, detail="Invalid session") from exc

    if session.payment_status != "paid":
        raise HTTPException(status_code=402, detail="Payment not completed")

    if session.client_reference_id != user.id:
        raise HTTPException(status_code=403, detail="Session does not belong to this account")

    credits = int(session.metadata.get("credits", 0))
    payment_intent_id: str = session.payment_intent or ""

    if not credits:
        raise HTTPException(status_code=400, detail="Session missing credits metadata")

    # Idempotency — skip if already applied (webhook may have beaten us here)
    if payment_intent_id:
        existing = await db.execute(
            select(CreditTransaction).where(
                CreditTransaction.stripe_payment_intent_id == payment_intent_id
            )
        )
        if existing.scalar_one_or_none() is not None:
            result = await db.execute(select(User).where(User.id == user.id))
            db_user = result.scalar_one()
            return ConfirmResponse(
                balance_cents=db_user.credit_balance_cents,
                balance_display=f"${db_user.credit_balance_cents / 100:.2f}",
                credits_added=0,
            )

    result = await db.execute(select(User).where(User.id == user.id).with_for_update())
    db_user = result.scalar_one()
    db_user.credit_balance_cents += credits
    db.add(CreditTransaction(
        user_id=user.id,
        amount_cents=credits,
        description=f"Purchased {credits:,} credits",
        stripe_payment_intent_id=payment_intent_id,
    ))

    await db.commit()
    await db.refresh(db_user)

    return ConfirmResponse(
        balance_cents=db_user.credit_balance_cents,
        balance_display=f"${db_user.credit_balance_cents / 100:.2f}",
        credits_added=credits,
    )
