"""
generator.py
------------
Generates a suggested reply for a new incoming email.

Approach: RAG + few-shot prompting (no fine-tuning). We retrieve the k most
similar historical (email, reply) pairs from the training split and place
them in the prompt as few-shot examples, then ask the LLM to write a new
reply for the current email in the same house style.

Why this approach over the alternatives:
- Fine-tuning: would need far more data than 220 examples to avoid
  overfitting/memorizing, is slower to iterate on, and makes it hard to
  update behavior when policies change (refund window, discount tiers) --
  you'd have to retrain. Not justified at this scale.
- Pure prompting with a static handful of hand-picked few-shot examples:
  simpler, but doesn't adapt the examples to the *specific* incoming email,
  so style/content transfer is weaker for less common categories.
- RAG + few-shot (chosen): retrieval keeps the examples relevant to the
  specific email at hand, requires no training step, and updates instantly
  if the historical dataset changes (add/edit an example -> next call reflects
  it). Costs: retrieval quality bounds generation quality (garbage in,
  garbage out) and prompt length grows with k -- mitigated by keeping k small
  (3) and by TF-IDF being fast enough to be irrelevant to latency.
"""
from src.retriever import TfidfRetriever
from src.llm_client import get_llm

SYSTEM_PROMPT = (
    "You are a customer support agent for an e-commerce company. You write "
    "concise, warm, professional email replies. You always address the "
    "specific facts in the customer's email (order numbers, amounts, dates) "
    "and, where relevant, propose a concrete next step. You imitate the tone "
    "and structure of the example replies you're shown, but adapt content to "
    "the new email -- never copy an example's specific facts verbatim if they "
    "don't apply."
)

PROMPT_TEMPLATE = """Below are examples of past incoming emails and the replies our team sent, most similar first.

{examples}

Now write a reply to this new incoming email. Keep the same tone/style, address all the specific facts (order numbers, amounts, names) in the NEW email only, and keep it under ~120 words.

### INCOMING EMAIL
Subject: {subject}
{body}
### END

Write only the reply text, no preamble, no explanation."""

EXAMPLE_BLOCK = """---
EMAIL: {subject}
{body}
REPLY: {reply}
"""


class ReplyGenerator:
    def __init__(self, corpus_path, k=3, model="gemini-2.5-flash"):
        self.llm = get_llm(model="gemini-2.5-flash")
        self.retriever = TfidfRetriever(corpus_path)
        self.llm = get_llm(model=model)
        self.k = k

    def generate(self, subject, body, exclude_id=None):
        neighbors = self.retriever.retrieve(subject, body, k=self.k, exclude_id=exclude_id)
        examples_text = "\n".join(
            EXAMPLE_BLOCK.format(subject=r["subject"], body=r["incoming_email"], reply=r["sent_reply"])
            for r, _ in neighbors
        )
        prompt = PROMPT_TEMPLATE.format(examples=examples_text, subject=subject, body=body)
        reply = self.llm.complete(SYSTEM_PROMPT, prompt, max_tokens=400, temperature=0.4)
        return {
            "reply": reply,
            "retrieved_examples": [{"id": r["id"], "similarity": round(sim, 3)} for r, sim in neighbors],
        }
