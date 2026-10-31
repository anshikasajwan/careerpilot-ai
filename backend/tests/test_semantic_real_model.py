"""Opt-in tests that download and run the REAL model (slow, needs internet).

Run them with:  RUN_MODEL_TESTS=1 python -m pytest tests/test_semantic_real_model.py -v
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="Set RUN_MODEL_TESTS=1 to run tests that use the real model.",
)

JOB = "Hiring a backend engineer who knows Python and databases."


def test_related_texts_score_higher_than_unrelated_texts():
    pytest.importorskip("sentence_transformers")
    from ml.embeddings import SentenceTransformerEmbedder, semantic_score

    embedder = SentenceTransformerEmbedder()
    related = semantic_score(
        "Software developer experienced in Python and SQL.", JOB, embedder
    )
    unrelated = semantic_score(
        "Pastry chef experienced in French desserts.", JOB, embedder
    )
    assert related > unrelated
