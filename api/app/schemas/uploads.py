from pydantic import BaseModel, Field


class CreateUploadRequest(BaseModel):
    size_bytes: int = Field(gt=0)


class CreateUploadResponse(BaseModel):
    key: str
    upload_id: str
    part_size: int
    part_urls: list[str]


class UploadedPart(BaseModel):
    part_number: int = Field(ge=1, le=10_000)
    etag: str


class CompleteUploadRequest(BaseModel):
    key: str
    upload_id: str
    parts: list[UploadedPart] = Field(min_length=1)


class CompleteUploadResponse(BaseModel):
    key: str


class AbortUploadRequest(BaseModel):
    key: str
    upload_id: str
