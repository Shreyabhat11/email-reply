"""
evaluator.py
------------
THE CORE OF THIS PROJECT: turning "is this a good suggested reply?" into a
measurable, explainable score.

## Why not exact match / plain string similarity?
A suggested reply is "correct" if it does the right *job*, not if it matches
the historical reply's wording. Two agents can resolve the same email in
different (both fine) phrasings. So:
  - Exact match is nearly useless (near-0% even for great replies).
  - Pure lexical similarity (BLEU/ROUGE/TF-IDF cosine to the one historical
    reply) rewards copying that one reply's wording and can penalize a
    perfectly good reply that's phrased differently, or reward a bad reply
    that happens to reuse the same words. We still compute it, but only as a
    *diagnostic* signal reported alongside the real score, not as the score.

## What we actually measure ("accurate" = 4 things a suggested reply needs)
1. Task/content coverage -- did the reply address the specific facts and
   required actions that THIS email calls for? (order number cited, refund
   timeframe given, etc.) This is checked against a per-example
   `required_elements` checklist that we authored when building the dataset
   -- an explicit, auditable ground truth that doesn't depend on wording.
2. Relevance -- is the reply actually about what the customer asked, not a
   generic/off-topic response?
3. Tone/professionalism -- does it read like a real, courteous support reply?
4. Completeness -- does it leave the customer with a clear next step /
   resolution rather than a half-answer?

(2)-(4) are judged by an LLM-as-judge with a fixed rubric (1-5 each) and a
required written rationale -- because these are exactly the qualities humans
would use to grade a reply, and no cheap automatic metric captures them well.
(1) is judged by the same LLM call against the explicit checklist, which
keeps it grounded in facts rather than vibes.

## Aggregate score
overall_score (0-100) = 100 * weighted_mean(
    coverage   x 0.40,   # did it do the actual job -- weighted highest
    relevance  x 0.25,
    completeness x 0.20,
    tone       x 0.15,
) / 5

Weights reflect that a reply which is polite and on-topic but misses the
customer's actual ask is a worse failure than one that's slightly curt but
handles everything correctly -- coverage is what gets escalated/complained
about; tone is comparatively easy to fix in a quick edit.

## Validating the metric against real quality
Automatic metrics can be self-consistent nonsense. `eval/validate_metric.py`
checks this: we hand-authored a small "gold" set of (email, candidate reply,
human 1-5 quality rating) pairs -- including deliberately bad candidates
(off-topic, missing the order number, rude tone) -- and compute Spearman/
Pearson correlation between our automatic overall_score and the human
ratings. See eval/gold_labels.jsonl and the README section "Validating the
metric" for the honest result and its limits (small n, single rater = us).
"""

import json
import re
import statistics as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.llm_client import get_llm


JUDGE_SYSTEM = """
You are a strict evaluator of customer-support email replies.

Evaluate the candidate reply against the incoming email, required elements,
and reference reply.

Return ONLY valid JSON.
Do not use markdown.
Do not explain your answer.
Do not add any text before or after the JSON.

Use exactly this schema:

{
  "relevance": 1,
  "completeness": 1,
  "tone": 1,
  "element_coverage": {
    "required element": true
  }
}

Scoring:
- relevance: 1 = irrelevant, 3 = partially relevant, 5 = directly addresses the request
- completeness: 1 = misses the main request, 3 = partially addresses it, 5 = fully addresses it
- tone: 1 = inappropriate, 3 = acceptable, 5 = professional and empathetic

For element_coverage:
- Use exactly the required elements provided.
- true only when the candidate actually satisfies that element.
- false when it is missing, vague, or unsupported.

Do not infer actions that the candidate does not explicitly communicate.
"""


JUDGE_PROMPT = """### RUBRIC

Evaluate the GENERATED REPLY against the INCOMING EMAIL.

Score these semantic dimensions from 1 to 5:

- relevance:
  1 = unrelated or off-topic
  3 = generally relevant but somewhat generic
  5 = directly addresses the customer's request

- tone:
  1 = rude, inappropriate, or unprofessional
  3 = acceptable but could be improved
  5 = professional, courteous, and appropriately empathetic

- completeness:
  1 = fails to address the request
  3 = partially addresses the request but misses useful information
  5 = gives a complete resolution or clear next step

Then evaluate EVERY REQUIRED ELEMENT independently.

For each required element:
- true = the generated reply clearly addresses that requirement
- false = the generated reply does not address it

Do NOT infer that an element was satisfied merely because the reply is
generally relevant.

IMPORTANT:
- Do not reward copied wording.
- Do not penalize different wording when the meaning is correct.
- Do not invent facts.
- Be strict about missing required information.

### INCOMING EMAIL

{email}

### REQUIRED ELEMENTS

{elements}

### GENERATED REPLY

{reply}

Return EXACTLY this JSON structure:

{{
  "relevance": <integer 1-5>,
  "tone": <integer 1-5>,
  "completeness": <integer 1-5>,
  "element_coverage": {{
    "<required element>": true,
    "<required element>": false
  }},
  "rationale": "<2-3 sentence explanation>"
}}
"""


def _clamp_score(value, minimum=1.0, maximum=5.0):
    """
    Safely constrain a judge score to the expected 1-5 range.

    Invalid values fall back to the midpoint instead of crashing the
    evaluation pipeline.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 3.0

    return max(minimum, min(maximum, value))


def _lexical_similarity(reply, reference):
    """
    Diagnostic-only TF-IDF cosine similarity.

    This is intentionally NOT part of the final quality score.
    """
    if not reply.strip() or not reference.strip():
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([reply, reference])

    return float(cosine_similarity(matrix[0], matrix[1])[0][0])


def _parse_judge_json(raw):
    """
    Parse the LLM judge output.

    Handles:
    - clean JSON
    - accidental ```json fences
    - JSON surrounded by explanatory text
    - malformed output

    Malformed output fails safely with neutral semantic scores and no
    fabricated element coverage.
    """
    if not raw:
        return {
            "relevance": 3.0,
            "tone": 3.0,
            "completeness": 3.0,
            "element_coverage": {},
            "rationale": "[empty judge output]",
        }

    cleaned = raw.strip()

    # Remove markdown fences if the model ignored the instruction.
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    # Last-resort extraction of the first JSON object.
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)

    if match:
        try:
            parsed = json.loads(match.group(0))

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

    return {
        "relevance": 3.0,
        "tone": 3.0,
        "completeness": 3.0,
        "element_coverage": {},
        "rationale": (
            f"[unparsable judge output: {cleaned[:300]}]"
        ),
    }



def _normalize_coverage(required_elements, coverage_map):
    """
    Return coverage ONLY for the explicitly required elements.

    This prevents the LLM from adding arbitrary keys and ensures that
    coverage is always calculated against our authored ground truth.
    """
    if not required_elements:
        return {}

    if not isinstance(coverage_map, dict):
        coverage_map = {}

    normalized = {}

    for element in required_elements:
        # Strict boolean check.
        # Strings such as "true" are intentionally NOT accepted.
        normalized[element] = coverage_map.get(element) is True

    return normalized


class ReplyEvaluator:
    """
    Evaluate a generated email reply using a structured LLM judge.
    """

    def __init__(self, model="gemini-2.5-flash"):
        self.llm = get_llm(model=model)

    def evaluate(
        self,
        incoming_email,
        required_elements,
        generated_reply,
        reference_reply=None,
    ):
        required_elements = required_elements or []

        elements_text = "\n".join(
            f"- {element}"
            for element in required_elements
        )

        if not elements_text:
            elements_text = "(none specified)"

        prompt = JUDGE_PROMPT.format(
            email=incoming_email,
            elements=elements_text,
            reply=generated_reply,
        )

        raw = self.llm.complete(
            JUDGE_SYSTEM,
            prompt,
            max_tokens=300,
            temperature=0.0,
        )

        parsed = _parse_judge_json(raw)

        # ---------------------------------------------------------------
        # Semantic dimensions
        # ---------------------------------------------------------------

        relevance = _clamp_score(
            parsed.get("relevance", 3.0)
        )

        tone = _clamp_score(
            parsed.get("tone", 3.0)
        )

        completeness = _clamp_score(
            parsed.get("completeness", 3.0)
        )

        # ---------------------------------------------------------------
        # Required-element coverage
        # ---------------------------------------------------------------

        coverage_map = _normalize_coverage(
            required_elements,
            parsed.get("element_coverage", {}),
        )

        if required_elements:
            satisfied = sum(
                coverage_map[element]
                for element in required_elements
            )

            coverage = (
                5.0 * satisfied / len(required_elements)
            )
        else:
            coverage = 5.0

        # ---------------------------------------------------------------
        # Aggregate score
        # ---------------------------------------------------------------

        overall_0_5 = (
            0.40 * coverage
            + 0.25 * relevance
            + 0.20 * completeness
            + 0.15 * tone
        )

        overall_100 = round(
            20.0 * overall_0_5,
            1,
        )

        result = {
            "scores": {
                "coverage": round(coverage, 2),
                "relevance": round(relevance, 2),
                "tone": round(tone, 2),
                "completeness": round(completeness, 2),
                "overall_score": overall_100,
            },
            "element_coverage": coverage_map,
            "rationale": parsed.get(
                "rationale",
                "",
            ),
        }

        # ---------------------------------------------------------------
        # Optional diagnostic metric
        # ---------------------------------------------------------------

        if reference_reply is not None:
            result[
                "diagnostic_lexical_similarity_to_historical_reply"
            ] = round(
                _lexical_similarity(
                    generated_reply,
                    reference_reply,
                ),
                3,
            )

        return result


def summarize(results):
    """
    Produce an overall report across multiple evaluated responses.
    """

    n = len(results)

    if n == 0:
        return {"n": 0}

    dimensions = [
        "coverage",
        "relevance",
        "tone",
        "completeness",
        "overall_score",
    ]

    output = {
        "n": n,
    }

    for dimension in dimensions:
        values = [
            result["scores"][dimension]
            for result in results
        ]

        output[f"mean_{dimension}"] = round(
            st.mean(values),
            2,
        )

        output[f"stdev_{dimension}"] = round(
            st.pstdev(values),
            2,
        ) if n > 1 else 0.0

    lexical_scores = [
        result[
            "diagnostic_lexical_similarity_to_historical_reply"
        ]
        for result in results
        if "diagnostic_lexical_similarity_to_historical_reply" in result
    ]

    if lexical_scores:
        output["mean_diagnostic_lexical_similarity"] = round(
            st.mean(lexical_scores),
            3,
        )

    output["pass_rate_overall_ge_70"] = round(
        sum(
            result["scores"]["overall_score"] >= 70
            for result in results
        ) / n,
        3,
    )

    return output