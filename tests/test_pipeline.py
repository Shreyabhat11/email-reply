"""
Lightweight tests (plain assert-based, run with `python -m pytest tests/` or
`python tests/test_pipeline.py`). No network/API key required -- exercises
the offline mock backend, which is enough to validate wiring & data shape.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import TfidfRetriever
from src.generator import ReplyGenerator
from src.evaluator import ReplyEvaluator, summarize

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "emails.jsonl")


def test_dataset_shape():
    rows = [json.loads(l) for l in open(DATA_PATH)]
    assert len(rows) >= 100
    required_keys = {"id", "category", "subject", "incoming_email", "sent_reply", "required_elements"}
    for r in rows[:20]:
        assert required_keys.issubset(r.keys())
        assert len(r["required_elements"]) >= 1
        assert len(r["incoming_email"]) > 10
        assert len(r["sent_reply"]) > 10


def test_retriever_returns_neighbors():
    retriever = TfidfRetriever(DATA_PATH)
    results = retriever.retrieve("Where is my order?", "I ordered a lamp and it hasn't shipped.", k=3)
    assert len(results) == 3
    for row, sim in results:
        assert 0.0 <= sim <= 1.0
        assert "sent_reply" in row


def test_generator_produces_text():
    gen = ReplyGenerator(DATA_PATH, k=2)
    out = gen.generate("Refund please", "I'd like a refund for order #123456, it broke.")
    assert isinstance(out["reply"], str) and len(out["reply"]) > 0
    assert len(out["retrieved_examples"]) == 2


def test_evaluator_scores_in_range():
    ev = ReplyEvaluator()
    result = ev.evaluate(
        incoming_email="I'd like a refund for order #123456, it broke.",
        required_elements=["references order number #123456", "confirms refund"],
        generated_reply="Hi, sorry to hear that! I've refunded order #123456, you'll see it in 5 days.",
        reference_reply="Hi, refund issued for #123456.",
    )
    s = result["scores"]
    for k in ("coverage", "relevance", "tone", "completeness"):
        assert 0.0 <= s[k] <= 5.0
    assert 0.0 <= s["overall_score"] <= 100.0


def test_summarize_handles_empty_and_nonempty():
    assert summarize([]) == {"n": 0}
    ev = ReplyEvaluator()
    r = ev.evaluate("test email", ["x"], "test reply")
    out = summarize([r])
    assert out["n"] == 1
    assert "mean_overall_score" in out

def test_evaluator_ignores_legacy_scalar_coverage():
    """
    A legacy scalar 'coverage' field must not be treated as
    element-level coverage.
    """
    from src.evaluator import _normalize_coverage

    required_elements = [
        "mentions the order number",
        "explains the refund process",
    ]

    legacy_response = {
        "coverage": 5
    }

    result = _normalize_coverage(
        required_elements,
        legacy_response.get("element_coverage", {}),
    )

    assert result == {
        "mentions the order number": False,
        "explains the refund process": False,
    }


def test_evaluator_normalizes_element_coverage():
    """
    Coverage must be calculated only from the required elements.
    """
    from src.evaluator import _normalize_coverage

    required_elements = [
        "mentions the order number",
        "explains the refund process",
        "provides a timeframe",
    ]

    judge_response = {
        "element_coverage": {
            "mentions the order number": True,
            "explains the refund process": True,
            "provides a timeframe": False,
            # Extra hallucinated field should be ignored.
            "says thank you": True,
        }
    }

    result = _normalize_coverage(
        required_elements,
        judge_response["element_coverage"],
    )

    assert result == {
        "mentions the order number": True,
        "explains the refund process": True,
        "provides a timeframe": False,
    }


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"OK: {name}")
    print("All tests passed.")
