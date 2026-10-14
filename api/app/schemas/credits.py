import uuid

from pydantic import BaseModel, ConfigDict, Field

from .jobs import UTCDatetime


class CreditPackage(BaseModel):
    id: str
    price_cents: int
    credits: int
    label: str
    bonus_pct: int
    price_display: str
    credits_display: str


class CreditsResponse(BaseModel):
    balance_cents: int
    balance_display: str
    packages: list[CreditPackage]
    payment_past_due: bool = False


class CheckoutRequest(BaseModel):
    package_id: str | None = None
    custom_amount_cents: int | None = Field(default=None, ge=1000)  # min $10


class CheckoutResponse(BaseModel):
    checkout_url: str


class ConfirmResponse(BaseModel):
    balance_cents: int
    balance_display: str
    credits_added: int


class TransactionResponse(BaseModel):
    id: int
    amount_cents: int
    description: str | None
    job_id: uuid.UUID | None
    created_at: UTCDatetime

    model_config = ConfigDict(from_attributes=True)
