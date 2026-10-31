import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_embedder
from app.main import app
from tests.fakes import FailingEmbedder, FakeEmbedder

client = TestClient(app)

RESUME = "Python developer with SQL and Docker experience."
JOB = "We need a Python engineer who knows SQL databases."
UNRELATED = "Experienced chef specializing in French pastry."


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


def post(payload):
    return client.post("/match/preview", json=payload)


def test_semantic_method_scores_similar_texts_high():
    app.dependency_overrides[get_embedder] = FakeEmbedder
    response = post({"resume_text": RESUME, "job_text": JOB, "method": "semantic"})
    assert response.status_code == 200
    body = response.json()
    # Fake vectors: resume = [python, sql, docker], job = [python, sql]
    # cosine = 2 / sqrt(3 * 2) = 0.8165
    assert body["score"] == 0.8165
    assert body["shared_keywords"] == []
    assert body["method"] == "semantic-all-MiniLM-L6-v2"


def test_semantic_method_scores_unrelated_texts_zero():
    app.dependency_overrides[get_embedder] = FakeEmbedder
    response = post(
        {"resume_text": UNRELATED, "job_text": JOB, "method": "semantic"}
    )
    assert response.status_code == 200
    assert response.json()["score"] == 0.0


def test_semantic_failure_returns_503_without_details():
    app.dependency_overrides[get_embedder] = FailingEmbedder
    response = post({"resume_text": RESUME, "job_text": JOB, "method": "semantic"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Semantic matching is currently unavailable."}


def test_unknown_method_returns_422():
    response = post({"resume_text": RESUME, "job_text": JOB, "method": "magic"})
    assert response.status_code == 422


def test_keyword_method_never_touches_the_embedder():
    app.dependency_overrides[get_embedder] = FailingEmbedder
    response = post({"resume_text": RESUME, "job_text": JOB, "method": "keyword"})
    assert response.status_code == 200
    assert response.json()["method"] == "keyword-overlap-baseline"


def test_method_defaults_to_keyword():
    response = post({"resume_text": RESUME, "job_text": JOB})
    assert response.json()["method"] == "keyword-overlap-baseline"
