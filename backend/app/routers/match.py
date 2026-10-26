"""HTTP routes for job matching. Contains no matching logic itself."""

from fastapi import APIRouter

from app.schemas import MatchRequest, MatchResponse
from app.services.matching import preview_match

router = APIRouter(prefix="/match", tags=["match"])


@router.post("/preview", response_model=MatchResponse)
def match_preview(payload: MatchRequest) -> MatchResponse:
    """Score how well a resume matches a job description (baseline method)."""
    return preview_match(payload)
