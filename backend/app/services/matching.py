"""Service layer: the only place that connects the API to the ML code.

The router talks to this file; this file talks to `ml/`. Neither the router
nor the ML code needs to know about the other.
"""

from app.schemas import MatchRequest, MatchResponse
from ml.baseline import score_match
from ml.embeddings import MODEL_NAME, Embedder, semantic_score

KEYWORD_METHOD_NAME = "keyword-overlap-baseline"
SEMANTIC_METHOD_NAME = f"semantic-{MODEL_NAME.split('/')[-1]}"


def preview_match(request: MatchRequest, embedder: Embedder) -> MatchResponse:
    if request.method == "semantic":
        score = semantic_score(request.resume_text, request.job_text, embedder)
        # A meaning-based score has no keyword list to show.
        return MatchResponse(
            score=score, shared_keywords=[], method=SEMANTIC_METHOD_NAME
        )

    result = score_match(request.resume_text, request.job_text)
    return MatchResponse(
        score=result.score,
        shared_keywords=result.shared_keywords,
        method=KEYWORD_METHOD_NAME,
    )
