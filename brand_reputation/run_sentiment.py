"""
run_sentiment.py
────────────────
ONE-TIME script. Run this BEFORE launching the dashboard.

What it does:
  1. Loads items.csv + reviews.csv
  2. Cleans all review text properly
  3. Runs RoBERTa sentiment model on every review body
  4. Classifies negative reviews into sub-categories
  5. Saves final scored CSV → data/reviews_scored.csv

How to run:
  python run_sentiment.py

Time estimate:
  ~20-40 minutes on CPU for full dataset (~67k reviews)
  Use --brand flag to process only one brand first (faster for testing):
  python run_sentiment.py --brand Samsung

After this finishes, launch the dashboard:
  streamlit run app.py
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ITEMS_PATH   = "data/20191226-items.csv"
REVIEWS_PATH = "data/20191226-reviews.csv"
OUTPUT_PATH  = "data/reviews_scored.csv"
BATCH_SIZE   = 64   # increase to 64 if you have more RAM
MAX_TOKENS   = 512  # RoBERTa hard limit

# Label mapping from Cardiff RoBERTa model
LABEL_MAP = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral":  "Neutral",
}

# ── Negative sub-category keywords ───────────────────────────────────────────
CATEGORY_KEYWORDS = {
    "Defective Piece": [
        "defect", "broken", "broke", "damage", "fault", "malfunction",
        "dead", "crack", "shatter", "stop working", "stopped working",
        "doesn't work", "not working", "burn", "overheat", "explode",
        "hardware", "screen problem", "battery drain", "battery dead",
        "screen crack", "won't turn on", "dead on arrival", "doa",
    ],
    "Price Dissatisfaction": [
        "expensive", "overpriced", "price", "cost", "cheap",
        "worth", "money", "pricey", "costly", "ripoff", "rip off",
        "pay", "afford", "budget", "value", "not worth", "waste of money",
        "too much", "highway robbery",
    ],
    "Poor Customer Service": [
        "customer service", "support", "warranty", "representative",
        "helpline", "customer care", "service center", "repair",
        "replacement", "response", "unresponsive", "rude", "staff",
        "no help", "useless support", "terrible service",
    ],
    "Delivery / Packaging": [
        "delivery", "shipping", "shipped", "package", "arrived",
        "late", "delay", "carrier", "courier", "box", "damaged box",
        "return", "refund", "seller", "amazon", "wrong item",
        "not delivered", "missing", "packaging",
    ],
}


# ── Text Cleaning ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Clean review text before feeding to RoBERTa.
    Steps:
      - Remove HTML tags
      - Remove URLs
      - Remove excessive punctuation (keep basic sentence punctuation)
      - Normalize whitespace
      - Strip leading/trailing whitespace
    Note: Do NOT remove emojis — RoBERTa understands them.
    Note: Do NOT lowercase — RoBERTa is case-sensitive (CAPS = emphasis).
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove HTML entities like &amp; &nbsp; etc.
    text = re.sub(r"&[a-zA-Z]+;", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove repeated punctuation (e.g. "!!!!" → "!")
    text = re.sub(r"([!?.]){2,}", r"\1", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def truncate_to_tokens(text: str, tokenizer, max_tokens: int = 512) -> str:
    """
    Truncate text to max_tokens using the model's actual tokenizer.
    This prevents RoBERTa from crashing on very long reviews.
    """
    tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) <= max_tokens:
        return text
    # Decode truncated tokens back to string
    truncated = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
    return truncated


# ── Negative Category Classification ─────────────────────────────────────────
def classify_negative_category(text: str) -> str:
    """
    Given a negative review text, classify into sub-category
    using keyword frequency scoring.
    """
    text_lower = text.lower()
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        scores[category] = sum(1 for kw in keywords if kw in text_lower)

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General Dissatisfaction"


# ── Impact Score ──────────────────────────────────────────────────────────────
def compute_impact_score(row) -> float:
    """
    0-100 score for negative reviews only.
    Higher = more damaging to brand.
    """
    if row["sentiment_label"] != "Negative":
        return 0.0

    score = 0.0

    # Confidence from RoBERTa (higher confidence negative = more certain damage)
    score += row.get("confidence", 50) * 0.3     # max 30 pts

    # Helpful votes (amplifies reach)
    helpful = min(row.get("helpfulVotes", 0), 50)
    score += (helpful / 50) * 25                  # max 25 pts

    # Verified buyer = more trustworthy
    if row.get("verified", False):
        score += 20                               # 20 pts

    # Category severity weight
    cat_weight = {
        "Defective Piece":        15,
        "Poor Customer Service":  12,
        "Price Dissatisfaction":   8,
        "Delivery / Packaging":    6,
        "General Dissatisfaction": 4,
    }
    score += cat_weight.get(row.get("negative_category", ""), 4)

    return round(min(score, 100), 2)


# ── Main Pipeline ─────────────────────────────────────────────────────────────
def main(brand_filter=None):
    print("=" * 60)
    print("  BrandSense — RoBERTa Sentiment Scoring Pipeline")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────
    print("\n[1/6] Loading data...")
    items   = pd.read_csv(ITEMS_PATH)
    reviews = pd.read_csv(REVIEWS_PATH)
    print(f"      items.csv:   {len(items):,} products")
    print(f"      reviews.csv: {len(reviews):,} reviews")

    # ── 2. Clean items ────────────────────────────────────────
    print("\n[2/6] Cleaning and merging data...")
    items = items.dropna(subset=["brand"])
    items["brand"] = items["brand"].str.strip().str.title()
    items["price"] = items["price"].replace(0, np.nan)

    # ── 3. Clean reviews ──────────────────────────────────────
    reviews = reviews.dropna(subset=["body"])
    reviews["body"]         = reviews["body"].astype(str)
    reviews["date"]         = pd.to_datetime(reviews["date"], errors="coerce")
    reviews                 = reviews.dropna(subset=["date"])
    reviews["verified"]     = reviews["verified"].astype(bool)
    reviews["helpfulVotes"] = reviews["helpfulVotes"].fillna(0).astype(int)
    reviews["rating"]       = reviews["rating"].clip(1, 5)

    # ── 4. Merge ──────────────────────────────────────────────
    df = reviews.merge(
        items[["asin", "brand", "title", "price"]],
        on="asin", how="inner"
    )
    df = df.rename(columns={"title_x": "review_title", "title_y": "product_title"})
    if "rating_x" in df.columns:
        df = df.rename(columns={"rating_x": "rating"})
    if "rating_y" in df.columns:
        df = df.drop(columns=["rating_y"])

    # Remove duplicate reviews
    before = len(df)
    df = df.drop_duplicates(subset=["body"])
    print(f"      Removed {before - len(df):,} duplicate reviews")

    # Filter by brand if specified
    if brand_filter:
        df = df[df["brand"].str.lower() == brand_filter.lower()].copy()
        print(f"      Filtered to brand '{brand_filter}': {len(df):,} reviews")

    # Add time columns
    df["year"]       = df["date"].dt.year
    df["month"]      = df["date"].dt.month
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    print(f"      Final dataset: {len(df):,} reviews")

    # ── 5. Clean text for RoBERTa ─────────────────────────────
    print("\n[3/6] Cleaning review text...")
    df["clean_body"] = df["body"].apply(clean_text)

    # Drop reviews where clean text is empty
    df = df[df["clean_body"].str.len() > 5].copy()
    print(f"      Reviews after text cleaning: {len(df):,}")

    # ── 6. Load RoBERTa model ─────────────────────────────────
    print(f"\n[4/6] Loading RoBERTa model: {MODEL_NAME}")
    print("      (First run will download ~500MB model — takes a few minutes)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    print(f"      Running on: {device.upper()}")

    # ── 7. Run RoBERTa in batches ─────────────────────────────
    print(f"\n[5/6] Running sentiment analysis (batch size={BATCH_SIZE})...")
    print(f"      Total reviews to score: {len(df):,}")
    print(f"      Estimated time on CPU: ~{len(df)//500} minutes")
    print()

    sentiments   = []
    confidences  = []

    texts = df["clean_body"].tolist()
    total = len(texts)

    for i in range(0, total, BATCH_SIZE):
        batch_texts = texts[i: i + BATCH_SIZE]

        # Truncate each text to 512 tokens
        batch_texts = [truncate_to_tokens(t, tokenizer, MAX_TOKENS) for t in batch_texts]

        # Tokenize
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Inference
        with torch.no_grad():
            outputs = model(**encoded)

        scores = softmax(outputs.logits.cpu().numpy(), axis=1)

        for score_row in scores:
            label_idx   = int(np.argmax(score_row))
            raw_label   = model.config.id2label[label_idx].lower()
            sentiment   = LABEL_MAP.get(raw_label, "Neutral")
            confidence  = round(float(score_row[label_idx]) * 100, 1)
            sentiments.append(sentiment)
            confidences.append(confidence)

        # Progress
        done = min(i + BATCH_SIZE, total)
        pct  = done / total * 100
        bar  = "█" * int(pct // 2) + "░" * (50 - int(pct // 2))
        print(f"\r      [{bar}] {pct:.1f}% ({done:,}/{total:,})", end="", flush=True)

    print("\n      ✅ Sentiment scoring complete!")

    df["sentiment_label"] = sentiments
    df["confidence"]      = confidences

    # ── 8. Classify negative categories ──────────────────────
    print("\n[6/6] Classifying negative review categories...")
    df["negative_category"] = df.apply(
        lambda row: classify_negative_category(row["clean_body"])
        if row["sentiment_label"] == "Negative" else "",
        axis=1
    )

    # Compute impact scores
    df["impact_score"] = df.apply(compute_impact_score, axis=1)

    # ── 9. Save ───────────────────────────────────────────────
    keep_cols = [
        "asin", "brand", "product_title", "review_title", "clean_body",
        "rating", "date", "year", "month", "year_month",
        "verified", "helpfulVotes",
        "sentiment_label", "confidence", "negative_category", "impact_score",
        "price",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df[keep_cols].to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'='*60}")
    print(f"  ✅ Done! Scored CSV saved to: {OUTPUT_PATH}")
    print(f"  Total reviews scored: {len(df):,}")
    print(f"\n  Sentiment breakdown:")
    for label, count in df["sentiment_label"].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {label:10s}: {count:6,} ({pct:.1f}%)")
    print(f"\n  Now run:  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--brand",
        type=str,
        default=None,
        help="Filter to one brand only (e.g. --brand Samsung). Processes all brands if omitted."
    )
    args = parser.parse_args()
    main(brand_filter=args.brand)
