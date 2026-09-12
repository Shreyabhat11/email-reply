"""
pipeline.py
-----------
Wires dataset -> generator -> evaluator -> reports.

Train/test split: the retriever is only allowed to retrieve from `train`
examples; `test` examples are held out and used as the "new incoming emails"
we generate replies for (their sent_reply is used only as (a) a diagnostic
lexical-similarity reference and (b) the source of required_elements -- never
shown to the generator).
"""
import json
import os
import random

from src.generator import ReplyGenerator
from src.evaluator import ReplyEvaluator, summarize


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def split(rows, test_frac=0.2, seed=13):
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    n_test = max(1, int(len(rows) * test_frac))
    return rows[n_test:], rows[:n_test]


def run(data_path, out_dir="reports", k=3, test_frac=0.2, model="gemini-2.5-flash", limit=None):
    os.makedirs(out_dir, exist_ok=True)
    all_rows = load_jsonl(data_path)
    train, test = split(all_rows, test_frac=test_frac)

    train_path = os.path.join(out_dir, "_train_split.jsonl")
    with open(train_path, "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")

    generator = ReplyGenerator(train_path, k=k, model=model)
    evaluator = ReplyEvaluator(model=model)

    if limit:
        test = test[:limit]

    per_response = []
    for row in test:
        gen = generator.generate(row["subject"], row["incoming_email"], exclude_id=row["id"])
        ev = evaluator.evaluate(
            incoming_email=row["incoming_email"],
            required_elements=row["required_elements"],
            generated_reply=gen["reply"],
            reference_reply=row["sent_reply"],
        )
        per_response.append({
            "id": row["id"],
            "category": row["category"],
            "incoming_email": row["incoming_email"],
            "required_elements": row["required_elements"],
            "generated_reply": gen["reply"],
            "retrieved_examples": gen["retrieved_examples"],
            "historical_reply": row["sent_reply"],
            "evaluation": ev,
        })
        print(f"[{row['id']}] {row['category']:20s} overall={ev['scores']['overall_score']:5.1f}")

    overall = summarize([r["evaluation"] for r in per_response])

    with open(os.path.join(out_dir, "per_response_report.json"), "w") as f:
        json.dump(per_response, f, indent=2)
    with open(os.path.join(out_dir, "overall_report.json"), "w") as f:
        json.dump(overall, f, indent=2)

    _write_markdown_report(per_response, overall, os.path.join(out_dir, "eval_report.md"))
    return per_response, overall


def _write_markdown_report(per_response, overall, path):
    lines = ["# Evaluation Report\n"]
    lines.append("## Overall system score\n")
    lines.append("```json")
    lines.append(json.dumps(overall, indent=2))
    lines.append("```\n")
    lines.append("## Per-response scores\n")
    lines.append("| id | category | overall | coverage | relevance | tone | completeness |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in per_response:
        s = r["evaluation"]["scores"]
        lines.append(f"| {r['id']} | {r['category']} | {s['overall_score']} | {s['coverage']} | "
                      f"{s['relevance']} | {s['tone']} | {s['completeness']} |")
    lines.append("\n## Sample detail (first 3)\n")
    for r in per_response[:3]:
        s = r["evaluation"]["scores"]
        lines.append(f"### {r['id']} ({r['category']}) -- overall {s['overall_score']}\n")
        lines.append(f"**Incoming email:**\n> {r['incoming_email']}\n")
        lines.append(f"**Generated reply:**\n> {r['generated_reply']}\n")
        lines.append(f"**Rationale:** {r['evaluation']['rationale']}\n")
    with open(path, "w") as f:
        f.write("\n".join(lines))
