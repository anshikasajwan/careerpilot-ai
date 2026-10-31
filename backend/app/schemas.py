"""Pydantic models that define the shape of API requests and responses.

Pydantic validates data automatically: if a request does not match the
model, FastAPI answers with a clear 422 error and never runs our code.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MIN_TEXT_LENGTH = 20
MAX_TEXT_LENGTH = 20_000  # caps request size so one call cannot hog memory


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    version: str = Field(..., examples=["0.1.0"])


class MatchRequest(BaseModel):
    # Strip surrounding whitespace; reject unexpected extra fields.
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    resume_text: str = Field(
        ..., min_length=MIN_TEXT_LENGTH, max_length=MAX_TEXT_LENGTH
    )
    job_text: str = Field(
        ..., min_length=MIN_TEXT_LENGTH, max_length=MAX_TEXT_LENGTH
    )
    # Only these two values are accepted; anything else gets a 422.
    method: Literal["keyword", "semantic"] = "keyword"


class MatchResponse(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    shared_keywords: list[str]
    method: str = Field(..., examples=["keyword-overlap-baseline"])
