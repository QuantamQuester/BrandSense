import requests
import json
import re
import pandas as pd

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"


def analyze_sentiment_batch(reviews: list) -> list:
    reviews_text = ""
    for i, r in enumerate(reviews):
        reviews_text += f"\n[{i+1}] Rating: {r['rating']}/5\nReview: {r['body'][:400]}\n"

    prompt = f"""Analyze these product reviews and return ONLY a JSON array.

For each review return:
- sentiment: "Positive", "Negative", or "Neutral"
- confidence: integer 0-100
- category: if Negative, one of ["Defective Piece", "Price Dissatisfaction", "Poor Customer Service", "Delivery / Packaging", "General Dissatisfaction"]. If not Negative, return ""
- key_issue: one short phrase (max 6 words) describing the main complaint or praise

Reviews:
{reviews_text}

Return ONLY a JSON array like:
[{{"sentiment":"Negative","confidence":92,"category":"Defective Piece","key_issue":"phone stopped working after week"}}, ...]

No explanation, no markdown, just the JSON array."""

    try:
        response = requests.post(
            ANTHROPIC_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": MODEL,
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        data = response.json()
        raw = data["content"][0]["text"].strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        results = json.loads(raw)

        for i, item in enumerate(results):
            item["id"] = reviews[i]["id"]

        return results

    except Exception:
        fallback = []
        for r in reviews:
            rating = r.get("rating", 3)
            if rating <= 2:
                sentiment = "Negative"
            elif rating == 3:
                sentiment = "Neutral"
            else:
                sentiment = "Positive"
            fallback.append({
                "id": r["id"],
                "sentiment": sentiment,
                "confidence": 70,
                "category": "",
                "key_issue": "rule-based fallback"
            })
        return fallback


def generate_brand_summary(brand: str, stats: dict) -> str:
    prompt = f"""Write a 3-sentence brand health summary for {brand} based on these metrics:

- Total Reviews Analyzed: {stats.get('total_reviews', 0)}
- Positive Reviews: {stats.get('positive_pct', 0):.1f}%
- Negative Reviews: {stats.get('negative_pct', 0):.1f}%
- Average Rating: {stats.get('avg_rating', 0):.2f}/5
- Top Negative Category: {stats.get('top_negative_cat', 'N/A')}
- Reputation Risk Score: {stats.get('risk_score', 0):.0f}/100
- Recent Trend: {stats.get('trend', 'stable')}

Write as a professional brand analyst. Be specific, concise, and actionable. No bullet points."""

    try:
        response = requests.post(
            ANTHROPIC_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": MODEL,
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=20
        )
        data = response.json()
        return data["content"][0]["text"].strip()
    except Exception:
        return (
            f"{brand} has a {stats.get('positive_pct', 0):.0f}% positive sentiment rate "
            f"with an average rating of {stats.get('avg_rating', 0):.2f}/5. "
            f"Key concerns center around {stats.get('top_negative_cat', 'product quality')}."
        )


def compute_reputation_risk_score(df_brand) -> float:
    if len(df_brand) == 0:
        return 0.0

    total = len(df_brand)
    neg_count = (df_brand["sentiment_label"] == "Negative").sum()
    neg_ratio = neg_count / total

    avg_impact = df_brand["impact_score"].mean()

    try:
        latest_date = df_brand["date"].max()
        recent = df_brand[df_brand["date"] >= latest_date - pd.Timedelta(days=90)]
        recent_neg_ratio = (recent["sentiment_label"] == "Negative").sum() / max(len(recent), 1)
        trend_multiplier = 1.2 if recent_neg_ratio > neg_ratio * 1.3 else 1.0
    except Exception:
        trend_multiplier = 1.0

    score = (neg_ratio * 50) + (avg_impact * 0.5)
    score = score * trend_multiplier

    return round(min(score, 100), 1)