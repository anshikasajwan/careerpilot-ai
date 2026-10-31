"""Shared FastAPI dependencies.

A dependency is a function FastAPI calls for us and passes into a route.
Tests can replace it with a fake (`app.dependency_overrides`), so they never
need to download the real model.
"""

from ml.embeddings import Embedder, get_default_embedder


def get_embedder() -> Embedder:
    return get_default_embedder()
