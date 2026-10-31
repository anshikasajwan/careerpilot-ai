"""Semantic matching with a small sentence-transformers model.

Pure Python + NumPy: no FastAPI, no HTTP. The heavy `sentence_transformers`
library is imported only when the model is first needed, so importing this
module stays fast and works even if the library is not installed.

How it works:
1. An embedding model turns a text into a list of 384 numbers (a vector).
   Texts with similar meaning get vectors that point in similar directions.
2. Cosine similarity measures the angle between two vectors:
   1.0 = same direction, 0.0 = unrelated, negative = opposite.
"""

import logging
import threading
from functools import lru_cache
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingUnavailableError(RuntimeError):
    """The embedding model could not be loaded or used."""


class Embedder(Protocol):
    """Anything that turns a list of texts into a 2-D array of vectors."""

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Loads the model lazily (on first use) and keeps it in memory."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            # Lock: the web server may handle two first requests at once.
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer

                        self._model = SentenceTransformer(self._model_name)
                    except Exception as exc:
                        raise EmbeddingUnavailableError(
                            f"Could not load model {self._model_name!r}"
                        ) from exc
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        try:
            return np.asarray(model.encode(texts, convert_to_numpy=True))
        except Exception as exc:
            raise EmbeddingUnavailableError("Embedding failed") from exc


@lru_cache(maxsize=1)
def get_default_embedder() -> SentenceTransformerEmbedder:
    """One shared embedder for the whole process (the model loads once)."""
    return SentenceTransformerEmbedder()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of the angle between two vectors; 0.0 if either is all zeros."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def semantic_score(resume_text: str, job_text: str, embedder: Embedder) -> float:
    """Meaning-based similarity between a resume and a job, from 0.0 to 1.0.

    Cosine similarity can be negative; we clip at 0 so the score stays in the
    same 0-1 range as the keyword method. Do not read it as a percentage.
    """
    vectors = embedder.encode([resume_text, job_text])
    if vectors.ndim != 2 or vectors.shape[0] != 2:
        raise EmbeddingUnavailableError("Unexpected embedding shape")
    similarity = cosine_similarity(vectors[0], vectors[1])
    return round(min(max(similarity, 0.0), 1.0), 4)
