"""Baseline resume-to-job matcher based on keyword overlap.

This module is pure Python: it knows nothing about FastAPI, HTTP or
databases. That separation lets us test and improve the ML logic on its own,
and later swap this baseline for a sentence-transformers model without
touching the API code.

Score = Jaccard similarity = |shared words| / |all distinct words|.
It ranges from 0.0 (nothing in common) to 1.0 (identical word sets).
"""

import re
from dataclasses import dataclass

# Very common words that say nothing about skills. Deliberately small.
STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "have", "in", "is", "it", "of", "on", "or", "our", "the",
        "to", "we", "with", "you", "your",
    }
)

# Keeps tokens like "c++", "c#" and "node.js" intact; drops trailing dots.
_TOKEN_RE = re.compile(r"[a-z0-9+#]+(?:\.[a-z0-9]+)*")

MAX_SHARED_KEYWORDS = 20


@dataclass(frozen=True)
class MatchResult:
    score: float
    shared_keywords: list[str]


def tokenize(text: str) -> set[str]:
    """Lowercase the text and return its distinct non-stopword tokens."""
    tokens = _TOKEN_RE.findall(text.lower())
    return {token for token in tokens if token not in STOPWORDS}


def score_match(resume_text: str, job_text: str) -> MatchResult:
    """Compare a resume and a job description by shared keywords."""
    resume_tokens = tokenize(resume_text)
    job_tokens = tokenize(job_text)

    all_tokens = resume_tokens | job_tokens
    if not all_tokens:
        return MatchResult(score=0.0, shared_keywords=[])

    shared = resume_tokens & job_tokens
    score = len(shared) / len(all_tokens)
    return MatchResult(
        score=round(score, 4),
        shared_keywords=sorted(shared)[:MAX_SHARED_KEYWORDS],
    )
