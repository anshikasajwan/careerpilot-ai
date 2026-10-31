"""HTTP routes for job matching. Contains no matching logic itself."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_embedder
from app.schemas import MatchRequest, MatchResponse
from app.services.matching import preview_match
from ml.embeddings import Embedder, EmbeddingUnavailableError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/match", tags=["match"])


# A plain `def` route: FastAPI runs it in a worker thread, so a slow model
# call does not freeze the whole server.
@router.post("/preview", response_model=MatchResponse)
def match_preview(
    payload: MatchRequest, embedder: Embedder = Depends(get_embedder)
) -> MatchResponse:
    """Score how well a resume matches a job description."""
    try:
        return preview_match(payload, embedder)
    except EmbeddingUnavailableError:
        # Log the details for us; tell the client only a generic message.
        logger.exception("Semantic matching is unavailable")
        raise HTTPException(
            status_code=503, detail="Semantic matching is currently unavailable."
        ) from None
