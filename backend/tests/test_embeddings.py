"""Tests for the semantic matching logic. No real model involved."""

import sys
from unittest.mock import patch

import numpy as np

from ml.embeddings import (
    EmbeddingUnavailableError,
    SentenceTransformerEmbedder,
    cosine_similarity,
    semantic_score,
)


class FixedEmbedder:
    """Returns the two vectors it was given, whatever the text."""

    def __init__(self, first, second):
        self._vectors = np.array([first, second], dtype=float)

    def encode(self, texts):
        return self._vectors


def test_cosine_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_orthogonal_vectors_is_zero():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0


def test_cosine_opposite_vectors_is_minus_one():
    result = cosine_similarity(np.array([1.0, 0.0]), np.array([-1.0, 0.0]))
    assert abs(result + 1.0) < 1e-9


def test_cosine_zero_vector_returns_zero_not_error():
    assert cosine_similarity(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0


def test_semantic_score_clips_negative_to_zero():
    embedder = FixedEmbedder([1.0, 0.0], [-1.0, 0.0])
    assert semantic_score("a", "b", embedder) == 0.0


def test_semantic_score_is_rounded_to_four_places():
    # cosine of [1, 0] and [1, 2] is 1 / sqrt(5) = 0.44721...
    embedder = FixedEmbedder([1.0, 0.0], [1.0, 2.0])
    assert semantic_score("a", "b", embedder) == 0.4472


def test_creating_embedder_does_not_load_the_model():
    assert SentenceTransformerEmbedder()._model is None


def test_load_failure_becomes_embedding_unavailable():
    # Setting a module to None in sys.modules makes importing it fail.
    with patch.dict(sys.modules, {"sentence_transformers": None}):
        try:
            SentenceTransformerEmbedder().encode(["hello"])
        except EmbeddingUnavailableError:
            return
    raise AssertionError("expected EmbeddingUnavailableError")
