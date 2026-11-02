import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from .jobs import UTCDatetime


class CreateKeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class ApiKeyInfo(BaseModel):
    id: uuid.UUID
    name: str
    prefix: str
    created_at: UTCDatetime
    last_used_at: UTCDatetime | None
    expires_at: UTCDatetime


class CreatedKey(ApiKeyInfo):
    key: str  # the only time it is ever returned


class Me(BaseModel):
    email: str
    credit_balance_cents: int
    billing_enabled: bool
