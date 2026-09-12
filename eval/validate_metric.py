#!/usr/bin/env python3
"""
validate_metric.py
-------------------
Sanity-checks that our automatic `overall_score` tracks real quality rather
than being an arbitrary number.

Method: eval/gold_labels.jsonl contains hand-authored (email, candidate
reply, human 1-5 rating) triples, deliberately including some clearly bad
replies (off-topic, missing required facts, wrong tone) alongside good ones.
We run the same ReplyEvaluator used in the main pipeline on each candidate
and compute Spearman & Pearson correlation between its automatic overall_score
and the human rating.

HONESTY NOTE (read this): n=12 and the "human" ratings are self-authored by
the project author while building the rubric, not an independent human study.
This is a plausibility check ("does the metric at least rank obviously-bad
replies below obviously-good ones, for cases the metric author did not use to
tune the rubric weights"), not a statistically powered validation. See
README "Validating the metric" for how we'd extend this with real judges.

Usage: python eval/validate_metric.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.stats import pearsonr, spearmanr
from src.evaluator import ReplyEvaluator


def main():
    path = os.path.join(os.path.dirname(__file__), "gold_labels.jsonl")
    rows = [json.loads(l) for l in open(path)]
    evaluator = ReplyEvaluator()

    auto_scores, human_scores, detail = [], [], []
    for r in rows:
        ev = evaluator.evaluate(
            incoming_email=r["incoming_email"],
            required_elements=r["required_elements"],
            generated_reply=r["candidate_reply"],
        )
        auto = ev["scores"]["overall_score"]
        auto_scores.append(auto)
        human_scores.append(r["human_rating"])
        detail.append({
            "id": r["id"], "human_rating_1_5": r["human_rating"],
            "auto_overall_score_0_100": auto, "human_note": r.get("human_note", ""),
        })
        print(f"{r['id']:5s} human={r['human_rating']}  auto={auto:5.1f}  ({r['human_note']})")

    pear, p_p = pearsonr(auto_scores, human_scores)
    spear, p_s = spearmanr(auto_scores, human_scores)
    print("\n=== Correlation between automatic overall_score and human rating ===")
    print(f"Pearson r  = {pear:.3f} (p={p_p:.3f})")
    print(f"Spearman rho = {spear:.3f} (p={p_s:.3f})")

    out = {
        "n": len(rows),
        "pearson_r": round(float(pear), 3),
        "spearman_rho": round(float(spear), 3),
        "detail": detail,
    }
    out_path = os.path.join(os.path.dirname(__file__), "..", "reports", "metric_validation.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
