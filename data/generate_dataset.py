"""
generate_dataset.py
--------------------
Builds the dataset of (incoming email, sent reply) pairs used to power the
RAG-based reply generator, plus per-example metadata used by the evaluator.

WHY SYNTHETIC + TEMPLATED (see README for full rationale):
Real customer-support mailboxes are private and full of PII, so a public,
inspectable dataset has to be built rather than scraped. We simulate a mid-size
e-commerce company's support inbox: 10 recurring intent categories, each with
several phrasing/scenario templates and randomized slot-filling (names, order
numbers, products, dates, complaint details, sentiment/tone, and occasional
typos/informality). This keeps the data:
  - Representative of a real, narrow-domain support setting (repeat intents,
    a house reply style, a fixed knowledge base of policies) -- exactly the
    setting where retrieval-augmented generation from historical replies is
    most defensible.
  - Free of any real PII or copyrighted text.
  - Auditable: every example is generated from a template we can inspect, and
    every example carries a `required_elements` checklist (facts/promises the
    *reply* must contain to be considered correct) that the evaluator uses.
    That checklist is authored by us as part of building the dataset -- it is
    the "ground truth" for factual/task coverage, independent of the exact
    wording of the historical reply.

Run:  python generate_dataset.py --n 220 --seed 7
Output: emails.jsonl  (one JSON object per line)
"""
import argparse
import json
import random

FIRST_NAMES = ["Alex", "Priya", "Jordan", "Wei", "Fatima", "Liam", "Sofia", "Noah",
               "Aisha", "Diego", "Emma", "Kenji", "Maria", "Sam", "Nia", "Omar",
               "Grace", "Lucas", "Ines", "Tariq"]
LAST_NAMES = ["Patel", "Nguyen", "Garcia", "Smith", "Kim", "Rossi", "Müller", "Khan",
              "Silva", "Johansson", "Cohen", "Dubois", "Ivanov", "Oduya", "Park"]
PRODUCTS = ["Aurora Desk Lamp", "TrailBlaze Backpack 40L", "SonicWave Earbuds Pro",
            "CloudNine Pillow", "IronGrip Yoga Mat", "PulseFit Smart Band",
            "EverBrew Coffee Maker", "NovaCharge Power Bank", "GlideStep Running Shoes",
            "BrightPath Desk Chair"]
AGENTS = ["Jamie", "Robin", "Casey", "Morgan", "Taylor"]

random.seed(0)  # placeholder, reseeded in main()


def order_number():
    return f"#{random.randint(100000, 999999)}"


def date_str():
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"2026-{m:02d}-{d:02d}"


def name():
    return random.choice(FIRST_NAMES), random.choice(LAST_NAMES)


def maybe_typo(text):
    """Light, controlled informality so emails don't look robotically clean."""
    if random.random() < 0.25:
        text = text.replace("I am", "Im").replace("do not", "dont").replace("cannot", "cant")
    if random.random() < 0.15:
        text = text.rstrip(".") + "..."
    return text


# ---------------------------------------------------------------------------
# Each category = (id, subject_templates, body_templates, reply_template,
#                   required_elements_template)
# {slots} are filled per-example. required_elements are short checklist
# strings the evaluator will check the reply against.
# ---------------------------------------------------------------------------

def cat_order_status(i):
    fn, ln = name()
    prod = random.choice(PRODUCTS)
    order = order_number()
    subj = random.choice([
        f"Where is my order {order}?",
        f"Order status - {prod}",
        f"Tracking info for {order}",
    ])
    body = maybe_typo(
        f"Hi, I ordered the {prod} on {date_str()} (order {order}) and it still says "
        f"'processing'. Can you tell me when it will ship? Thanks, {fn}"
    )
    agent = random.choice(AGENTS)
    ship_days = random.choice([1, 2, 3])
    reply = (
        f"Hi {fn},\n\nThanks for reaching out! I checked order {order} and it's on track to "
        f"ship within {ship_days} business day(s). You'll get a tracking email as soon as it "
        f"leaves the warehouse. Let us know if you don't see it by then.\n\nBest,\n{agent}"
    )
    required = [
        f"references order number {order}",
        "gives a concrete shipping timeframe",
        "offers a next step if the timeline is missed",
    ]
    return subj, body, reply, "order_status", required


def cat_refund(i):
    fn, ln = name()
    prod = random.choice(PRODUCTS)
    order = order_number()
    reason = random.choice(["arrived damaged", "wasn't what I expected", "I no longer need it",
                             "was the wrong size"])
    subj = f"Refund request for order {order}"
    body = maybe_typo(
        f"Hello, I'd like a refund for the {prod} from order {order} because it {reason}. "
        f"How do I proceed? - {fn}"
    )
    agent = random.choice(AGENTS)
    days = random.choice([5, 7, 10])
    reply = (
        f"Hi {fn},\n\nSorry to hear that! I've started a refund for order {order}; you should "
        f"see the amount back on your original payment method within {days} business days. "
        f"If it's not too much trouble, please drop the item at any courier point using the "
        f"prepaid label I've just emailed you.\n\nThanks for your patience,\n{agent}"
    )
    required = [
        f"references order number {order}",
        "confirms the refund is being processed",
        "gives a refund timeframe",
        "mentions return/shipping instructions",
    ]
    return subj, body, reply, "refund_request", required


def cat_defective(i):
    fn, ln = name()
    prod = random.choice(PRODUCTS)
    order = order_number()
    issue = random.choice(["stopped turning on after two days", "arrived with a cracked case",
                            "makes a rattling noise", "the battery won't hold a charge"])
    subj = f"{prod} is defective"
    body = maybe_typo(
        f"Hi team, the {prod} I got (order {order}) {issue}. This is disappointing given the "
        f"price. What can you do? {fn}"
    )
    agent = random.choice(AGENTS)
    reply = (
        f"Hi {fn},\n\nI'm really sorry about that experience with your {prod} -- that's not "
        f"the quality we aim for. I can send a free replacement right away, or process a full "
        f"refund for order {order}, whichever you'd prefer. Just reply and let me know which "
        f"you'd like.\n\nApologies again,\n{agent}"
    )
    required = [
        f"references order number {order}",
        "apologizes for the defect",
        "offers a concrete remedy (replacement or refund)",
        "asks the customer to confirm a choice",
    ]
    return subj, body, reply, "defective_product", required


def cat_address_change(i):
    fn, ln = name()
    order = order_number()
    subj = f"Need to change shipping address - {order}"
    body = maybe_typo(
        f"Hi, I just realized I gave the wrong address for order {order}. Can you update it "
        f"before it ships? New address is 42 Birch Lane. Thanks, {fn}"
    )
    agent = random.choice(AGENTS)
    shipped = random.random() < 0.4
    if shipped:
        reply = (
            f"Hi {fn},\n\nI just checked and order {order} has unfortunately already shipped, "
            f"so I'm not able to change the address on it directly. What I can do is contact "
            f"the courier to request a redirect to 42 Birch Lane, or arrange a reship once it's "
            f"returned to us. Let me know which you'd prefer.\n\nBest,\n{agent}"
        )
        required = [f"references order number {order}", "explains the order already shipped",
                    "offers an alternative (courier redirect or reship)"]
    else:
        reply = (
            f"Hi {fn},\n\nGood news -- order {order} hasn't shipped yet, so I've updated the "
            f"delivery address to 42 Birch Lane. You'll get a confirmation once it's on its way.\n\n"
            f"Best,\n{agent}"
        )
        required = [f"references order number {order}", "confirms whether the address was updated",
                    "mentions next shipping step"]
    return subj, body, reply, "address_change", required


def cat_cancel_subscription(i):
    fn, ln = name()
    plan = random.choice(["Monthly", "Annual", "Premium"])
    subj = f"Cancel my {plan} subscription"
    body = maybe_typo(
        f"Hi, please cancel my {plan} subscription. I don't want to be charged again. {fn}"
    )
    agent = random.choice(AGENTS)
    reply = (
        f"Hi {fn},\n\nI've gone ahead and canceled your {plan} subscription -- you won't be "
        f"billed again. You'll still have access until the end of your current billing period. "
        f"If you change your mind, just reply to this email and I can reactivate it any time.\n\n"
        f"Best,\n{agent}"
    )
    required = ["confirms the subscription was canceled", "confirms no future charges",
                "mentions access until end of current period"]
    return subj, body, reply, "cancel_subscription", required


def cat_billing_dispute(i):
    fn, ln = name()
    amount = random.choice([19.99, 49.00, 89.50, 129.99])
    subj = "Unrecognized charge on my card"
    body = maybe_typo(
        f"Hello, I see a charge of ${amount} from you that I don't recognize. Can you explain "
        f"what this is for? {fn}"
    )
    agent = random.choice(AGENTS)
    reply = (
        f"Hi {fn},\n\nThanks for flagging this. I looked into the ${amount} charge and it "
        f"matches [order/renewal] on your account -- I've attached the invoice for reference. "
        f"If this still doesn't look right to you, let me know and I'll open a dispute review "
        f"right away.\n\nBest,\n{agent}"
    )
    required = [f"mentions the amount ${amount}", "explains or investigates the charge",
                "offers next step if customer disagrees"]
    return subj, body, reply, "billing_dispute", required


def cat_password_reset(i):
    fn, ln = name()
    subj = "Can't log into my account"
    body = maybe_typo(
        f"Hi, I'm locked out of my account and the reset email never arrives. Can you help? {fn}"
    )
    agent = random.choice(AGENTS)
    reply = (
        f"Hi {fn},\n\nSorry for the trouble! I've manually triggered a password reset email to "
        f"the address on file -- please check spam/promotions too. If it still doesn't arrive "
        f"in 10 minutes, reply here and I'll reset it from our side directly.\n\nBest,\n{agent}"
    )
    required = ["acknowledges the login issue", "takes or describes a concrete action",
                "gives a fallback if the first step fails"]
    return subj, body, reply, "password_reset", required


def cat_feature_feedback(i):
    fn, ln = name()
    prod = random.choice(PRODUCTS)
    idea = random.choice(["a dark mode", "bulk export", "a size chart on the product page",
                           "faster checkout"])
    subj = f"Suggestion: {idea}"
    body = maybe_typo(
        f"Hey, I love the {prod} but it would be great if you added {idea}. Any plans for that? {fn}"
    )
    agent = random.choice(AGENTS)
    reply = (
        f"Hi {fn},\n\nThanks so much for the suggestion -- {idea} comes up a lot and I've "
        f"passed it straight to our product team as feedback. I can't promise a timeline, but "
        f"we do prioritize requests we hear often, so this genuinely helps.\n\nThanks again,\n{agent}"
    )
    required = ["thanks the customer for feedback", "confirms the suggestion was logged/passed on",
                "sets honest expectations on timeline"]
    return subj, body, reply, "feature_feedback", required


def cat_sales_inquiry(i):
    fn, ln = name()
    prod = random.choice(PRODUCTS)
    subj = f"Question about {prod} before I buy"
    qty = random.choice([5, 10, 20])
    body = maybe_typo(
        f"Hi, I'm considering ordering {qty} units of the {prod} for my team. Do you offer a "
        f"bulk discount? {fn}"
    )
    agent = random.choice(AGENTS)
    disc = random.choice([10, 15, 20])
    reply = (
        f"Hi {fn},\n\nGreat question! For orders of {qty}+ units of the {prod}, we can offer a "
        f"{disc}% bulk discount. I can send over a quote with that pricing applied -- just "
        f"confirm the exact quantity and shipping address and I'll get it right over.\n\nBest,\n{agent}"
    )
    required = [f"addresses bulk discount for {qty} units", "gives a concrete discount or next step",
                "asks for details needed to finalize (quantity/address)"]
    return subj, body, reply, "sales_inquiry", required


def cat_scheduling(i):
    fn, ln = name()
    topic = random.choice(["a product demo", "an onboarding call", "a renewal discussion"])
    subj = f"Scheduling {topic}"
    day = random.choice(["Tuesday", "Wednesday", "Thursday"])
    body = maybe_typo(
        f"Hi, could we set up {topic} sometime next {day}? Morning works best for me. {fn}"
    )
    agent = random.choice(AGENTS)
    time_ = random.choice(["10:00 AM", "10:30 AM", "11:00 AM"])
    reply = (
        f"Hi {fn},\n\nNext {day} morning works well -- does {time_} your time suit you for "
        f"{topic}? I'll send a calendar invite with a video link as soon as you confirm.\n\n"
        f"Best,\n{agent}"
    )
    required = [f"proposes a specific day/time near {day} morning", f"references {topic}",
                "mentions sending a calendar invite/confirmation"]
    return subj, body, reply, "scheduling", required


CATEGORY_FUNCS = [
    cat_order_status, cat_refund, cat_defective, cat_address_change,
    cat_cancel_subscription, cat_billing_dispute, cat_password_reset,
    cat_feature_feedback, cat_sales_inquiry, cat_scheduling,
]


def build_dataset(n, seed):
    random.seed(seed)
    rows = []
    for i in range(n):
        fn = CATEGORY_FUNCS[i % len(CATEGORY_FUNCS)]
        subj, body, reply, cat, required = fn(i)
        rows.append({
            "id": f"em_{i:04d}",
            "category": cat,
            "subject": subj,
            "incoming_email": body,
            "sent_reply": reply,
            "required_elements": required,
        })
    random.shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=220)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default="emails.jsonl")
    args = ap.parse_args()

    rows = build_dataset(args.n, args.seed)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} examples to {args.out}")
    from collections import Counter
    print(Counter(r["category"] for r in rows))


if __name__ == "__main__":
    main()
