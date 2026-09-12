"""
llm_client.py
-------------
Thin wrapper around an LLM call, with two backends:

1. "gemini" (real): calls the Gemini API. Used whenever GEMINI_API_KEY
   is set in the environment. This is the intended way to run the system.
2. "mock" (offline fallback): a deterministic, template-based stand-in so the
   whole pipeline (generation + evaluation) can be demoed/tested with no
   network access and no API key. It is intentionally simple and clearly
   labeled -- it is NOT a substitute for real evaluation, only a way to prove
   the plumbing works end-to-end.

Both backends implement the same `.complete(system, prompt) -> str` interface
so the rest of the code never has to know which one is active.
"""
import os
import re

from dotenv import load_dotenv

load_dotenv()

class GeminiBackend:
    def __init__(self, model="gemini-2.5-flash"):
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def complete(self, system, prompt, max_tokens=300, temperature=0.0):
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                temperature=temperature,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0
                ),
            ),
        )

        return response.text.strip()

class MockBackend:
    """
    Deterministic offline stand-in. Two modes, chosen by the caller via the
    `task` kwarg baked into the prompt convention used here:
      - reply generation prompts contain '### INCOMING EMAIL' and few-shot
        examples; the mock stitches together the closest few-shot reply,
        lightly adapted (name/order-number swap) -- crude but self-consistent.
      - judge prompts contain '### RUBRIC'; the mock returns a heuristic
        keyword-overlap based score instead of a real semantic judgement.
    This backend exists ONLY so `python scripts/run_demo.py` works with no
    API key. Real accuracy numbers should be produced with GeminiBackend.
    """

    def complete(self, system, prompt, max_tokens=300, temperature=0.0):
        if "### RUBRIC" in prompt:
            return self._mock_judge(prompt)
        return self._mock_generate(prompt)

    # -- reply generation -------------------------------------------------
    def _mock_generate(self, prompt):
        # Pull the most similar few-shot reply out of the prompt and reuse it
        # near-verbatim (this is a deliberately weak baseline generator).
        examples = re.findall(r"REPLY:\s*(.*?)(?:\n\n---|\Z)", prompt, re.S)
        incoming = re.search(r"### INCOMING EMAIL\n(.*?)\n###", prompt, re.S)
        base = examples[0].strip() if examples else "Thanks for reaching out -- we'll look into this and follow up shortly."
        return base

    # -- judging ------------------------------------------------------------
    def _mock_judge(self, prompt: str) -> str:
        """
        Offline mock for the evaluator.

        This intentionally provides the same structured schema expected
        from the real LLM judge. It is only for local testing and is not
        intended to represent a high-quality semantic evaluator.
        """
        import json
        import re

        # Extract the candidate reply from the evaluation prompt.
        candidate_match = re.search(
            r"Candidate reply:\s*(.*?)(?:\n\s*Required elements:|\n\s*Reference reply:|\Z)",
            prompt,
            re.IGNORECASE | re.DOTALL,
        )

        candidate = (
            candidate_match.group(1).strip()
            if candidate_match
            else ""
        )

        # Extract required elements from the prompt.
        required_match = re.search(
            r"Required elements:\s*(.*?)(?:\n\s*Reference reply:|\n\s*Candidate reply:|\Z)",
            prompt,
            re.IGNORECASE | re.DOTALL,
        )

        required_text = (
            required_match.group(1).strip()
            if required_match
            else ""
        )

        # Basic offline approximation only.
        # The real LLM judge will perform semantic evaluation.
        required_elements = []

        for line in required_text.splitlines():
            line = line.strip("- •\t ")
            if line:
                required_elements.append(line)

        candidate_lower = candidate.lower()

        element_coverage = {}

        for element in required_elements:
            # Very simple lexical approximation for offline testing.
            words = re.findall(r"\b[a-zA-Z]{4,}\b", element.lower())

            if not words:
                element_coverage[element] = False
                continue

            matched = sum(
                1 for word in words
                if word in candidate_lower
            )

            element_coverage[element] = matched / len(words) >= 0.5

        covered = sum(element_coverage.values())
        total = len(element_coverage)

        if total:
            base_score = round(1 + 4 * (covered / total))
        else:
            base_score = 3

        return json.dumps(
            {
                "relevance": base_score,
                "completeness": base_score,
                "tone": 3,
                "element_coverage": element_coverage,
            }
        )


def get_llm(model="gemini-2.5-flash"):
    if os.environ.get("GEMINI_API_KEY"):
        try:
            return GeminiBackend(model=model)
        except Exception as e:
            print(f"[llm_client] Falling back to mock backend: {e}")
    else:
        print("[llm_client] No GEMINI _API_KEY found -> using offline mock backend "
              "(see README: this is for pipeline demo only, not real accuracy numbers).")
    return MockBackend()
