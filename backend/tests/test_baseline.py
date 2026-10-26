"""Tests for the ML baseline. No FastAPI involved: pure Python only."""

from ml.baseline import MAX_SHARED_KEYWORDS, score_match, tokenize


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Python and SQL") == {"python", "sql"}


def test_tokenize_keeps_special_skill_names():
    assert tokenize("C++ C# Node.js.") == {"c++", "c#", "node.js"}


def test_identical_texts_score_one():
    result = score_match("python sql docker", "python sql docker")
    assert result.score == 1.0


def test_disjoint_texts_score_zero():
    result = score_match("python sql", "cooking baking")
    assert result.score == 0.0
    assert result.shared_keywords == []


def test_partial_overlap_is_jaccard():
    # shared = {python, sql}; all = {python, sql, docker, kubernetes} -> 2/4
    result = score_match("python sql docker", "python sql kubernetes")
    assert result.score == 0.5
    assert result.shared_keywords == ["python", "sql"]


def test_stopword_only_text_scores_zero():
    assert score_match("the and of", "to in a").score == 0.0


def test_score_is_symmetric():
    a, b = "python sql docker", "python aws"
    assert score_match(a, b).score == score_match(b, a).score


def test_shared_keywords_are_capped():
    words = " ".join(f"w{i}" for i in range(MAX_SHARED_KEYWORDS + 10))
    result = score_match(words, words)
    assert len(result.shared_keywords) == MAX_SHARED_KEYWORDS
