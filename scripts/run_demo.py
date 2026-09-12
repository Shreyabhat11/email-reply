#!/usr/bin/env python3
"""
Run the full pipeline end-to-end: dataset -> RAG generation -> evaluation -> reports.

Usage:
    python scripts/run_demo.py --n 30
    GEMINI_API_KEY=sk-... python scripts/run_demo.py --n 30 --model gemini-2.5-flash

Without GEMINI_API_KEY set, this runs entirely offline using the mock LLM
backend (see src/llm_client.py) -- useful to confirm the plumbing works, but
NOT representative of real generation/evaluation quality. Set the key for
real numbers.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/emails.jsonl")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--k", type=int, default=3, help="number of few-shot retrieved examples")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--model", default="gemini-2.5-flash", help="LLM model to use for generation")
    ap.add_argument("--n", type=int, default=None, help="limit number of test emails (for quick runs)")
    args = ap.parse_args()

    per_response, overall = run(
        data_path=args.data, out_dir=args.out, k=args.k,
        test_frac=args.test_frac, model=args.model, limit=args.n,
    )
    print("\n=== OVERALL SYSTEM REPORT ===")
    print(json.dumps(overall, indent=2))
    print(f"\nFull reports written to: {args.out}/per_response_report.json, "
          f"{args.out}/overall_report.json, {args.out}/eval_report.md")


if __name__ == "__main__":
    main()
