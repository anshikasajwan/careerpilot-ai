"""Service layer: the only place that connects the API to the ML code.

The router talks to this file; this file talks to `ml/`. Neither the router
nor the ML code needs to know about the other.
"""

from app.schemas import MatchRequest, MatchResponse
from ml.baseline import score_match

METHOD_NAME = "keyword-overlap-baseline"


def preview_match(request: MatchRequest) -> MatchResponse:
    result = score_match(request.resume_text, request.job_text)
    return MatchResponse(
        score=result.score,
        shared_keywords=result.shared_keywords,
        method=METHOD_NAME,
    )
