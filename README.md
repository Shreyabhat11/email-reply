# AI Email Suggested-Response System

Given an incoming support email, this system (1) retrieves similar past
email/reply pairs, (2) asks an LLM to draft a suggested reply grounded in
those examples, and (3) scores the suggested reply for quality with an
explainable, multi-dimensional evaluator — reporting both per-response and
overall scores.

```
data/emails.jsonl          220 synthetic (email, sent_reply, required_elements) records
src/retriever.py           TF-IDF retrieval over past emails (the "RAG" part)
src/generator.py           builds a few-shot RAG prompt, calls the LLM
src/llm_client.py          real Anthropic API backend + offline mock fallback
src/evaluator.py           <-- the core: turns "reply quality" into scores
src/pipeline.py            wires it all together, writes reports
eval/gold_labels.jsonl     hand-labeled (email, reply, human rating) set
eval/validate_metric.py    checks the automatic metric against those labels
scripts/run_demo.py        CLI entry point
tests/test_pipeline.py     smoke tests (no API key required)
reports/                   generated output (JSON + Markdown reports)
```

## Quickstart

```bash
pip install -r requirements.txt

# Real run (recommended) — uses the Claude API for both generation and judging
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/run_demo.py --n 30

# Metric sanity-check against hand-labeled examples
python eval/validate_metric.py

# Smoke tests (no API key needed — exercises the offline mock backend)
python tests/test_pipeline.py
```

Without `ANTHROPIC_API_KEY` set, everything still *runs* end-to-end using a
deliberately simple offline mock LLM (`src/llm_client.py`) — useful to prove
the pipeline is wired correctly with zero setup, but **not** representative
of real quality (see "Honest results from this repo" below, where the mock's
own numbers make this point for us).

Outputs land in `reports/`: `per_response_report.json`, `overall_report.json`,
`eval_report.md` (human-readable), and `reports/metric_validation.json`.

---

## 1. The dataset — where it came from and why it's representative

Real support mailboxes are private and full of PII, so there's no way to
"source" a public dataset that's actually usable here without either
violating someone's privacy or using data with unknown license/quality
issues. Instead, `data/generate_dataset.py` **programmatically builds** 220
synthetic (incoming email, sent reply) pairs simulating a mid-size
e-commerce company's support inbox, across 10 recurring intents (order
status, refunds, defective products, address changes, subscription
cancellation, billing disputes, password resets, feature feedback, sales
inquiries, scheduling) — 22 examples each, with randomized names, order
numbers, products, amounts, dates, and some light informality/typos.

**Why this is a fair stand-in for a real inbox, not a toy:**
- Real support inboxes are dominated by a small number of repeating intents
  answered from a consistent "house style" and a fixed set of policies
  (refund windows, discount tiers, escalation paths) — exactly what's
  modeled here. This is the regime where retrieval-augmented generation from
  historical replies is actually a sound approach (see §2).
- Every example carries a `required_elements` checklist — e.g. for a refund
  email: *"references order number", "gives a refund timeframe", "mentions
  return instructions"*. This is authored **as part of building the
  dataset**, and is the backbone of the evaluator (§3): it defines what a
  correct reply must accomplish independent of exact wording, so we're not
  grading against a single "correct" string.

**Honest limitations:** synthetic data is more uniform than a real inbox —
real emails are messier, sometimes ramble across two intents at once, and
"house style" drifts across agents/years. The `required_elements` checklists
are also authored by us, i.e. they encode our judgment of what a good reply
needs, not a company's actual QA rubric. Swapping in a real (properly
licensed/anonymized) support-ticket export would be a drop-in replacement as
long as it's converted to the same JSONL schema with a `required_elements`
field added per example (which would need to be human-annotated).

---

## 2. Generating suggested responses (Gen AI)

**Approach: retrieval-augmented few-shot prompting**, not fine-tuning or a
classifier.

1. `TfidfRetriever` (`src/retriever.py`) retrieves the *k=3* most similar
   historical emails (TF-IDF cosine similarity over subject + body) from the
   training split.
2. `ReplyGenerator` (`src/generator.py`) builds a prompt containing those 3
   (email, reply) pairs as few-shot examples plus the new incoming email, and
   asks Claude to write a new reply in the same style, grounded in the new
   email's specific facts.

**Why RAG + few-shot over the alternatives:**

| Approach | Verdict here | Why |
|---|---|---|
| Fine-tuning an LLM | ❌ not used | 220 examples is too little to fine-tune without overfitting/memorizing; retraining is needed every time a policy changes (e.g. refund window); slow to iterate. Only justified with a much larger, continuously-refreshed dataset. |
| Static few-shot (same fixed examples every time) | ❌ not used | Simpler, but a customer emailing about a billing dispute gets shown examples about scheduling — weak style/content transfer for less common intents. |
| **RAG + few-shot (chosen)** | ✅ | Retrieval keeps examples relevant to *this* email; adding/editing a historical example takes effect on the very next call, no retraining; fully inspectable (you can see exactly which past examples informed a given suggestion via `retrieved_examples` in the report). |

**Why TF-IDF retrieval, not a neural embedding model:** this keeps the
retriever fully offline/dependency-light and works well in a narrow domain
with shared vocabulary ("refund", "order", "subscription"). Its weakness is
paraphrases with no shared words ("my package never came" vs. "where's my
order") — `src/retriever.py` includes an `EmbeddingRetriever` stub showing
exactly where a real embeddings API would slot in as a one-class swap if
that mattered more than offline-runnability.

**Why an LLM at all vs. a classifier:** the task is *generation* (produce
novel, well-formed prose addressing this email's specific facts), which is
what a classifier fundamentally cannot do — a classifier could at best pick
"which of these N template replies to send," which breaks the moment an
email doesn't match a known template.

---

## 3. Measuring accuracy — the core of this project

### What does "accurate" even mean for a suggested reply?

Exact match is nearly meaningless here: two replies can use completely
different words and both be excellent, or share most of the same words and
one still misses the point. So "accurate" is decomposed into the things a
human reviewer would actually check when approving a suggested reply before
it goes out:

1. **Coverage** — does the reply address the *specific facts and required
   actions this email calls for* (order number cited, a refund timeframe
   actually given, etc.)? Checked against the dataset's `required_elements`
   checklist — a per-example, auditable ground truth that's independent of
   wording.
2. **Relevance** — is it actually about what the customer asked, not generic
   boilerplate?
3. **Completeness** — does it leave the customer with a clear resolution or
   next step, or is it a half-answer?
4. **Tone** — does it read as professional and appropriately empathetic?

(2)–(4) are scored 1–5 by an **LLM-as-judge** with a fixed rubric prompt
(`src/evaluator.py::JUDGE_PROMPT`) that also returns a short **rationale**
and, for (1), a true/false judgement per required element (so coverage is
grounded in checking specific facts, not vibes). We use an LLM judge for
these because they're exactly the qualities a human reviewer uses, and no
cheap string-based metric captures "is this actually complete" or "is the
tone right."

**Lexical similarity to the one historical reply is deliberately *not* the
score** — it's computed (TF-IDF cosine) and reported as a *diagnostic* field
only, because rewarding closeness to one specific historical string would
penalize equally-good differently-worded replies and could reward a reply
that reuses the same words while missing the point.

### Aggregate score

```
overall_score (0–100) = 20 * (0.40·coverage + 0.25·relevance + 0.20·completeness + 0.15·tone) / 5
```

Coverage is weighted highest because a polite, on-topic reply that misses
the customer's actual ask (wrong/no order number, no refund timeframe) is
the failure mode that actually generates complaints and escalations; tone is
comparatively the easiest thing to fix in a one-line edit.

### Validating the metric against real quality

Automatic metrics can be self-consistent nonsense, so `eval/gold_labels.jsonl`
holds 12 hand-authored (email, candidate reply, 1–5 human rating) examples —
deliberately including clearly bad candidates (ignores the actual question,
never confirms the action taken, dodges the real ask) next to clearly good
ones for the *same* email, so a working metric should cleanly separate them.
`eval/validate_metric.py` runs the real evaluator on all 12 and reports
Pearson/Spearman correlation against the human ratings.

**Honest result from this repo, as actually run in this environment:** this
sandbox has no network access, so the number below was produced with the
**offline mock judge**, not the real Claude judge:

```
Pearson r  = -0.556 (p=0.060)
Spearman ρ = -0.656 (p=0.021)
```

That's a *negative* correlation — and that's an honest, useful result, not a
bug: the mock judge (`MockBackend._mock_judge` in `src/llm_client.py`) is an
intentionally dumb keyword-overlap heuristic with no real semantic
understanding, and this validation run demonstrates exactly why we don't
ship that as the real metric — a naive lexical-overlap scorer actively
disagrees with human judgment on this task (e.g. it can't tell that "have
you tried charging it overnight?" dodges a refund request, and even rewards
replies that repeat words from the required-elements list without actually
satisfying them). **Anyone running this with `ANTHROPIC_API_KEY` set will
exercise the real LLM-judge path** (`src/evaluator.py` calling
`AnthropicBackend`), which is the one actually designed against the rubric
above; re-run `python eval/validate_metric.py` with the key set to get the
real correlation before trusting scores from a live run. We were not able to
produce that number ourselves inside this sandboxed, network-disabled
environment — this is stated plainly rather than fabricated.

**Further limitations of even the real-judge validation:** n=12 and the
"human" ratings are self-authored by the person who wrote the rubric, not an
independent study — so at best this is a plausibility check ("does the
metric separate obviously-bad from obviously-good replies"), not a
statistically powered validation. A real deployment should replace/augment
this with: (a) a larger gold set labeled by actual support QA staff blind to
the automatic score, (b) inter-rater agreement among 2–3 human raters before
trusting any single human label as ground truth, and (c) periodically
re-checking correlation as the prompt/model changes, since LLM judges drift.

### Reporting

- **Per-response** (`reports/per_response_report.json`, and the table in
  `reports/eval_report.md`): the incoming email, generated reply, retrieved
  few-shot examples used, all four sub-scores, the overall score, the
  per-element true/false coverage breakdown, the judge's written rationale,
  and the diagnostic lexical-similarity number.
- **Overall** (`reports/overall_report.json`): mean and stdev of each
  dimension across the test set, mean diagnostic lexical similarity, and a
  pass-rate (`overall_score ≥ 70`) as one simple deployment-readiness signal.

---

## How AI tools were used

This repository (dataset generator, retriever, generator, evaluator,
pipeline, tests, and this README) was built with Claude as a pair-programmer
in an agentic coding session: Claude wrote the code and prose in this repo
directly, iterating in a real sandbox — the dataset was generated and
inspected, the pipeline was actually executed end-to-end (offline mock mode,
since the sandbox has no network access to call the live API), the test
suite was actually run and passed, and the metric-validation script was
actually executed to produce the correlation numbers quoted above. Where the
sandbox's lack of network access is a real limitation (no live Claude-API
run was possible here), that's stated explicitly above rather than
simulated or invented.

## Known limitations / what I'd do next with more time

- Retrieval is TF-IDF, not semantic embeddings — see `EmbeddingRetriever`
  stub in `src/retriever.py` for the intended upgrade path.
- The gold-label validation set is small and self-authored; needs real
  blind human raters at larger n for a trustworthy correlation number.
- No handling yet for multi-intent emails (an email that's both a complaint
  *and* a scheduling request) — the dataset and required_elements schema
  assume one dominant intent per email.
- The LLM judge is itself an LLM call, so it inherits whatever biases/blind
  spots the judge model has (e.g. possible leniency, position bias if ever
  extended to pairwise comparisons) — worth periodically auditing judge
  outputs against the human gold set as models change.
