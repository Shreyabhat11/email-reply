# Evaluation Report

## Overall system score

```json
{
  "n": 10,
  "mean_coverage": 0.0,
  "stdev_coverage": 0.0,
  "mean_relevance": 1.19,
  "stdev_relevance": 0.14,
  "mean_tone": 3.5,
  "stdev_tone": 0.0,
  "mean_completeness": 1.0,
  "stdev_completeness": 0.0,
  "mean_overall_score": 20.45,
  "stdev_overall_score": 0.69,
  "mean_diagnostic_lexical_similarity": 0.8,
  "pass_rate_overall_ge_70": 0.0
}
```

## Per-response scores

| id | category | overall | coverage | relevance | tone | completeness |
|---|---|---|---|---|---|---|
| em_0034 | cancel_subscription | 22.0 | 0.0 | 1.5 | 3.5 | 1.0 |
| em_0008 | sales_inquiry | 20.5 | 0.0 | 1.2 | 3.5 | 1.0 |
| em_0138 | sales_inquiry | 20.5 | 0.0 | 1.2 | 3.5 | 1.0 |
| em_0201 | refund_request | 19.5 | 0.0 | 1.0 | 3.5 | 1.0 |
| em_0108 | sales_inquiry | 20.5 | 0.0 | 1.2 | 3.5 | 1.0 |
| em_0044 | cancel_subscription | 21.0 | 0.0 | 1.3 | 3.5 | 1.0 |
| em_0151 | refund_request | 19.5 | 0.0 | 1.0 | 3.5 | 1.0 |
| em_0000 | order_status | 20.0 | 0.0 | 1.1 | 3.5 | 1.0 |
| em_0055 | billing_dispute | 20.5 | 0.0 | 1.2 | 3.5 | 1.0 |
| em_0208 | sales_inquiry | 20.5 | 0.0 | 1.2 | 3.5 | 1.0 |

## Sample detail (first 3)

### em_0034 (cancel_subscription) -- overall 22.0

**Incoming email:**
> Hi, please cancel my Annual subscription. I don't want to be charged again. Maria...

**Generated reply:**
> Hi Maria,

I've gone ahead and canceled your Annual subscription -- you won't be billed again. You'll still have access until the end of your current billing period. If you change your mind, just reply to this email and I can reactivate it any time.

Best,
Taylor

**Rationale:** Heuristic offline mock judge (keyword overlap only); run with ANTHROPIC_API_KEY set for a real semantic judgement.

### em_0008 (sales_inquiry) -- overall 20.5

**Incoming email:**
> Hi, I'm considering ordering 10 units of the NovaCharge Power Bank for my team. Do you offer a bulk discount? Diego

**Generated reply:**
> Hi Wei,

Great question! For orders of 5+ units of the NovaCharge Power Bank, we can offer a 15% bulk discount. I can send over a quote with that pricing applied -- just confirm the exact quantity and shipping address and I'll get it right over.

Best,
Jamie

**Rationale:** Heuristic offline mock judge (keyword overlap only); run with ANTHROPIC_API_KEY set for a real semantic judgement.

### em_0138 (sales_inquiry) -- overall 20.5

**Incoming email:**
> Hi, I'm considering ordering 5 units of the BrightPath Desk Chair for my team. Do you offer a bulk discount? Jordan

**Generated reply:**
> Hi Wei,

Great question! For orders of 20+ units of the BrightPath Desk Chair, we can offer a 10% bulk discount. I can send over a quote with that pricing applied -- just confirm the exact quantity and shipping address and I'll get it right over.

Best,
Morgan

**Rationale:** Heuristic offline mock judge (keyword overlap only); run with ANTHROPIC_API_KEY set for a real semantic judgement.
