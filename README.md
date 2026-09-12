# AI Email Suggested-Response System

Given an incoming support email, this system:

1. Retrieves similar historical email/reply pairs.
2. Uses a GenAI LLM to draft a suggested response grounded in those examples.
3. Evaluates the suggested response using an explainable, multi-dimensional quality metric.
4. Reports both per-response and overall evaluation results.

The project focuses heavily on **how to evaluate whether a generated email reply is actually good**, rather than treating exact string matching as "accuracy."

---

## Project Structure

```text
data/
├── emails.jsonl              # 220 synthetic email/reply pairs
└── generate_dataset.py       # Dataset generation script

src/
├── retriever.py              # TF-IDF retrieval over historical emails
├── generator.py              # Few-shot RAG prompt + response generation
├── llm_client.py             # Gemini API backend + offline mock fallback
├── evaluator.py              # Core multi-dimensional response evaluator
└── pipeline.py               # End-to-end pipeline and report generation

eval/
├── gold_labels.jsonl         # 12 hand-authored evaluation examples
└── validate_metric.py        # Validates automatic metric against gold ratings

scripts/
└── run_demo.py               # CLI entry point

tests/
└── test_pipeline.py          # Smoke/unit tests

reports/
├── per_response_report.json
├── overall_report.json
├── eval_report.md
└── metric_validation.json
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Gemini

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

The project uses **Gemini 2.5 Flash** for both response generation and LLM-based evaluation.

### 3. Generate suggested replies and evaluate them

```bash
python scripts/run_demo.py
```

You can also specify the number of evaluation examples:

```bash
python scripts/run_demo.py --n 30
```

### 4. Validate the evaluation metric

```bash
python eval/validate_metric.py
```

This runs the evaluator against the 12-example gold benchmark and compares the automated score with the benchmark's human-authored ratings.

### 5. Run tests

```bash
python tests/test_pipeline.py
```

The tests can run without an API key because the project includes an offline mock backend.

---

# 1. Dataset — Where It Came From

Real customer-support mailboxes contain private information and PII, making them unsuitable for publishing as part of a coding challenge.

Instead, `data/generate_dataset.py` programmatically creates **220 synthetic email/reply pairs** representing a mid-size e-commerce support inbox.

The dataset covers 10 recurring support intents:

* Order status
* Refund requests
* Defective products
* Address changes
* Subscription cancellation
* Billing disputes
* Password resets
* Feature feedback
* Sales inquiries
* Scheduling

There are 22 examples per category, with variation in:

* Customer names
* Order numbers
* Products
* Dates
* Amounts
* Customer wording
* Informality and minor typos

Each example also contains a `required_elements` checklist describing what a successful response should accomplish.

For example, a refund response may need to:

* Reference the order number
* Confirm that the refund is being processed
* Give a refund timeframe
* Provide return/shipping instructions

This allows the evaluation system to judge **whether the required outcome was achieved**, rather than whether the response matches one particular reference sentence.

### Why this is representative

Support inboxes commonly contain a relatively small number of recurring intents handled according to consistent business policies and communication styles.

That makes retrieval from historical responses a reasonable approach for this task: a new refund request can be grounded using previous refund responses rather than relying only on the LLM's general knowledge.

### Limitations

The dataset is still synthetic and therefore does not capture the full messiness of a real support inbox.

Real data may contain:

* Multiple intents in one email
* Much greater linguistic variation
* Incomplete information
* Long conversation histories
* Policy changes over time
* Different writing styles across support agents

The `required_elements` checklists are also authored as part of this project. They represent the project's definition of a good response rather than an actual company's QA rubric.

A properly licensed and anonymized support-ticket dataset could replace the synthetic dataset while keeping the same overall pipeline.

---

# 2. Generating Suggested Responses

## Approach: Retrieval-Augmented Few-Shot Prompting

The system uses **RAG + few-shot prompting** rather than fine-tuning or a fixed template system.

### Step 1 — Retrieve relevant historical examples

`TfidfRetriever` retrieves the top 3 historical emails using TF-IDF cosine similarity over the email subject/body.

### Step 2 — Build a grounded prompt

`ReplyGenerator` places the retrieved historical `(email, reply)` pairs into the prompt as few-shot examples together with the new incoming email.

The LLM is instructed to:

* Answer the customer's actual request
* Use facts from the incoming email
* Follow the communication style demonstrated by relevant examples
* Avoid inventing unsupported information

### Step 3 — Generate the suggested response

Gemini 2.5 Flash generates the final suggested reply.

The retrieved examples are retained in the evaluation report so that the output is inspectable rather than being an opaque generation.

---

## Why RAG + Few-Shot?

| Approach        | Decision   | Reason                                                                                                                                       |
| --------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Fine-tuning     | Not used   | 220 examples is too small for reliable fine-tuning and introduces retraining overhead whenever support policies change.                      |
| Static few-shot | Not used   | Fixed examples may be irrelevant to the current intent.                                                                                      |
| RAG + few-shot  | **Chosen** | Retrieves examples relevant to the current email, requires no retraining when examples change, and makes the grounding examples inspectable. |

### Why TF-IDF instead of embeddings?

TF-IDF keeps the retrieval system:

* Offline-capable
* Dependency-light
* Easy to inspect
* Reproducible

It works reasonably well in this narrow domain because related emails tend to share vocabulary such as "refund", "order", "subscription", etc.

Its main weakness is semantic paraphrasing. For example:

> "My package never came"

and

> "Where is my order?"

may have limited lexical overlap despite having similar intent.

The repository includes an `EmbeddingRetriever` stub showing where a semantic embedding-based retriever could replace TF-IDF later.

---

# 3. Evaluation — The Core of the Project

## What Does "Accurate" Mean for an Email Reply?

Exact-match accuracy is not appropriate for generative email responses.

Two responses can use completely different wording while both being excellent. Conversely, a response can contain many words from a reference answer while still failing to address the customer's actual request.

Therefore, this project defines response quality using four dimensions.

### 1. Coverage

Does the response satisfy the specific requirements of the incoming email?

Examples:

* Correct order number
* Refund timeframe
* Return instructions
* Confirmation that an action was completed

Coverage is evaluated against the example's `required_elements` checklist.

Each required element receives a true/false judgement, making the result auditable.

### 2. Relevance

Does the response directly address what the customer asked, rather than producing generic or unrelated text?

### 3. Completeness

Does the response provide a useful resolution or next step, rather than giving only a partial answer?

### 4. Tone

Is the response professional, clear, and appropriately empathetic?

Relevance, completeness, and tone are scored from 1–5 by an **LLM-as-judge** using the fixed rubric in:

```text
src/evaluator.py
```

The judge also provides a short rationale explaining its assessment.

---

## Aggregate Quality Score

The four dimensions are combined into a 0–100 score:

```text
overall_score =
    20 × (
        0.40 × coverage
      + 0.25 × relevance
      + 0.20 × completeness
      + 0.15 × tone
    )
```

All component scores are normalized to a 1–5 scale before aggregation.

Coverage receives the highest weight because a response that sounds professional but fails to perform or explain the requested action is still a poor support response.

Tone receives the lowest weight because it is generally easier to correct during final response editing than a missing business-critical action.

---

## Why Not Use Lexical Similarity as the Main Metric?

The system also calculates TF-IDF cosine similarity against the historical reply as a **diagnostic only**.

It is deliberately excluded from the quality score.

A reply should not receive a high score merely because it copies vocabulary from an old response. Likewise, a correctly written response should not be penalized simply because it uses different wording.

This distinction is important for generative systems.

---

# 4. Metric Validation

A scoring metric should itself be tested.

`eval/gold_labels.jsonl` contains 12 hand-authored evaluation cases covering six support categories.

The benchmark intentionally contains paired examples:

* Clearly good responses
* Clearly weak responses to the same underlying customer request

Examples include:

* A refund response that gives the order number, refund timeframe, and return instructions vs. a generic filler response.
* An order-status response with a concrete shipping timeframe vs. a vague "one or two weeks" response.
* A defective-product response offering a replacement/refund vs. a response that only suggests troubleshooting.
* A cancellation response confirming no future charges vs. a response that attempts to sell another plan.
* A sales response giving a concrete bulk discount vs. a generic sales message.
* A password-reset response with an action and fallback vs. a response that simply repeats the standard reset instruction.

The validation script runs the **same evaluator used by the main pipeline** against these examples and compares the automated scores with the benchmark's human-authored 1–5 ratings.

### Observed validation result

The current Gemini-based evaluator produced:

```text
Pearson r  = 0.987
Spearman ρ = 0.992
n = 12
```

This indicates that, on this small benchmark, the automated metric strongly separates the clearly good responses from the clearly weak ones.

### Important interpretation

**This is not "97.94% accuracy."**

The correlation result is a validation of the **evaluation metric**, not a measurement of how often the response generator produces correct answers in the real world.

Also, the benchmark is small and its human ratings were authored by the project developer. Therefore, this should be treated as a **sanity/plausibility check**, not as statistically representative human-evaluation research.

A production system should use a substantially larger benchmark labeled independently by multiple support-quality reviewers.

---

# 5. Evaluation Reports

The pipeline produces both detailed and aggregate reports.

### Per-response report

`reports/per_response_report.json` contains:

* Incoming email
* Generated reply
* Retrieved historical examples
* Coverage score
* Relevance score
* Completeness score
* Tone score
* Overall score
* Per-element coverage
* LLM judge rationale
* Diagnostic lexical similarity

This makes it possible to inspect **why a particular response received its score**.

### Overall report

`reports/overall_report.json` contains:

* Number of evaluated responses
* Mean score for each dimension
* Standard deviation
* Mean overall quality score
* Diagnostic lexical similarity
* Pass rate using `overall_score >= 70`

The pass rate is a simple operational signal and should not be interpreted as accuracy.

---

# 6. Current System-Level Evaluation

A live run of the pipeline evaluated 44 generated responses.

The resulting aggregate scores were:

```text
Responses evaluated:       44

Mean coverage:              4.81 / 5
Mean relevance:             5.00 / 5
Mean completeness:          4.86 / 5
Mean tone:                  5.00 / 5

Mean overall quality:       97.94 / 100
Pass rate (>=70):           97.7%
```

These results show that the current synthetic test set is relatively easy for the generator.

They **should not be interpreted as production accuracy**. The generated responses are evaluated on a controlled synthetic dataset whose intents and policies are relatively well-defined.

The 12-example gold benchmark is therefore especially important because it deliberately includes failure cases designed to test whether the evaluator can distinguish useful responses from superficially plausible ones.

---

# 7. AI Tool Disclosure

This project was developed with AI coding assistance.

Claude was used as a pair-programming tool during development to help:

* Design the project structure
* Implement and refine Python modules
* Debug the evaluation pipeline
* Develop tests
* Review generated outputs
* Draft and refine documentation

The final implementation was executed and tested locally, and the dataset and evaluation reports were generated from the repository's actual code.

The **response generation and LLM-based judging in the current implementation use Gemini 2.5 Flash** through the Google GenAI API.

The project also contains an offline mock backend so that the pipeline and tests can be exercised without an API key. The mock backend is intended for development/testing and is **not representative of real GenAI quality**.

---

# 8. Known Limitations and Future Improvements

### Current limitations

* **Synthetic dataset:** Real support data would provide more realistic language and edge cases.
* **TF-IDF retrieval:** Semantic embeddings would improve retrieval for paraphrased requests.
* **Small validation benchmark:** 12 examples are insufficient for statistically strong evaluation.
* **Self-authored ratings:** Independent human raters are needed for stronger metric validation.
* **Single-intent assumption:** Multi-intent emails are not currently modeled.
* **LLM-as-judge limitations:** The judge can inherit biases and blind spots from its underlying model.
* **Potential judge drift:** Changes to the judge model or prompt could change evaluation behavior.

### Next improvements

1. Replace TF-IDF with a production embedding model and compare retrieval quality.
2. Expand the gold benchmark substantially.
3. Use multiple independent human raters and measure inter-rater agreement.
4. Evaluate difficult multi-intent and ambiguous emails.
5. Add factuality/unsupported-claim detection.
6. Compare different LLMs as both generator and evaluator.
7. Add retrieval-quality metrics such as Recall@k.
8. Introduce an automated regression suite so model/prompt changes cannot silently degrade response quality.

---

# Design Summary

The project deliberately separates three concerns:

```text
                 Historical Emails
                        │
                        ▼
                 ┌─────────────┐
Incoming Email ─►│  Retriever  │
                 └──────┬──────┘
                        │
                 Top-k examples
                        │
                        ▼
                 ┌─────────────┐
                 │  GenAI LLM  │
                 └──────┬──────┘
                        │
                 Suggested Reply
                        │
                        ▼
                 ┌─────────────┐
                 │  Evaluator  │
                 │             │
                 │ Coverage    │
                 │ Relevance   │
                 │ Completeness│
                 │ Tone        │
                 └──────┬──────┘
                        │
                        ▼
              Per-response + Overall
                     Reports
```

The central design principle is:

> **For generative response systems, accuracy is not "does the output match a reference string?" — it is "does the response correctly address the customer's intent, satisfy the required actions, provide a useful resolution, and communicate appropriately?"**
