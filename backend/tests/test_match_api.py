from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

RESUME = "Python developer with FastAPI, SQL and Docker experience."
JOB = "We need a Python engineer who knows FastAPI and SQL databases."


def post(payload):
    return client.post("/match/preview", json=payload)


def test_valid_request_returns_score_and_keywords():
    response = post({"resume_text": RESUME, "job_text": JOB})
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["score"] <= 1.0
    assert {"python", "fastapi", "sql"} <= set(body["shared_keywords"])
    assert body["method"] == "keyword-overlap-baseline"


def test_identical_texts_score_one():
    response = post({"resume_text": RESUME, "job_text": RESUME})
    assert response.json()["score"] == 1.0


def test_no_overlap_scores_zero():
    response = post(
        {
            "resume_text": "Experienced chef specializing in French pastry.",
            "job_text": "Senior Python engineer building FastAPI services.",
        }
    )
    assert response.status_code == 200
    assert response.json()["score"] == 0.0


def test_missing_field_returns_422():
    assert post({"resume_text": RESUME}).status_code == 422


def test_too_short_text_returns_422():
    assert post({"resume_text": "hi", "job_text": JOB}).status_code == 422


def test_whitespace_only_text_returns_422():
    assert post({"resume_text": " " * 50, "job_text": JOB}).status_code == 422


def test_too_long_text_returns_422():
    too_long = "python " * 4000  # 28,000 characters
    assert post({"resume_text": too_long, "job_text": JOB}).status_code == 422


def test_unknown_field_returns_422():
    payload = {"resume_text": RESUME, "job_text": JOB, "admin": True}
    assert post(payload).status_code == 422


def test_wrong_type_returns_422():
    assert post({"resume_text": 12345, "job_text": JOB}).status_code == 422


def test_get_is_not_allowed():
    assert client.get("/match/preview").status_code == 405
