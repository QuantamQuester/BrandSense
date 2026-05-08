import pandas as pd
import numpy as np
import re
import os
import streamlit as st

ITEMS_PATH   = "data/20191226-items.csv"
REVIEWS_PATH = "data/20191226-reviews.csv"
SCORED_PATH  = "data/reviews_scored.csv"

CATEGORY_KEYWORDS = {
    "Defective Piece": [
        "defect","broken","broke","damage","fault","malfunction","dead","crack",
        "stop working","stopped working","doesn't work","not working","burn",
        "overheat","explode","battery drain","battery dead","dead on arrival","doa",
    ],
    "Price Dissatisfaction": [
        "expensive","overpriced","price","cost","worth","money","pricey","costly",
        "ripoff","rip off","not worth","waste of money","too much",
    ],
    "Poor Customer Service": [
        "customer service","support","warranty","representative","customer care",
        "service center","repair","replacement","unresponsive","rude","no help",
    ],
    "Delivery / Packaging": [
        "delivery","shipping","shipped","package","arrived","late","delay",
        "carrier","damaged box","return","refund","wrong item","not delivered",
    ],
}

RETURN_KEYWORDS = {
    "Completed Return":      ["returned it","sent it back","got my refund","money back"],
    "Exchange / Replacement":["got a replacement","replacement sent","exchanged it","swapped it"],
    "Return Attempted":      ["want to return","trying to return","asked for refund","return request"],
    "Return Failed":         ["couldn't return","denied return","no refund","won't refund"],
}

POSITIVE_KEYWORDS = [
    "love","excellent","amazing","perfect","great","fantastic","awesome","best",
    "recommend","outstanding","superb","wonderful","brilliant","impressive",
    "satisfied","happy","pleased","delighted","quality","reliable","smooth",
]


def scored_data_exists() -> bool:
    return os.path.exists(SCORED_PATH)


@st.cache_data(show_spinner=False)
def load_scored_data() -> pd.DataFrame:
    df = pd.read_csv(SCORED_PATH)
    df["date"]              = pd.to_datetime(df["date"], errors="coerce")
    df["verified"]          = df["verified"].astype(bool)
    df["negative_category"] = df["negative_category"].fillna("")
    df["confidence"]        = df["confidence"].fillna(0)
    df["impact_score"]      = df["impact_score"].fillna(0)
    if "year_month" not in df.columns:
        df["year_month"] = df["date"].dt.to_period("M").astype(str)
    if "year" not in df.columns:
        df["year"] = df["date"].dt.year
    if "month" not in df.columns:
        df["month"] = df["date"].dt.month

    # Add positive keyword score if not present
    if "positive_keywords" not in df.columns:
        df["positive_keywords"] = df["clean_body"].apply(_count_positive_keywords)

    # Add return signals if not present
    if "return_mentioned" not in df.columns:
        df = _add_return_signals(df)

    # Add pros/cons extraction
    if "pros" not in df.columns:
        df["pros"]  = df.apply(lambda r: _extract_pros(r), axis=1)
        df["cons"]  = df.apply(lambda r: _extract_cons(r), axis=1)

    return df.reset_index(drop=True)


def _count_positive_keywords(text: str) -> int:
    if not isinstance(text, str):
        return 0
    t = text.lower()
    return sum(1 for kw in POSITIVE_KEYWORDS if kw in t)


def _add_return_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return_mentioned"] = False
    df["return_type"]      = ""
    df["return_reason"]    = ""
    for ret_type, keywords in RETURN_KEYWORDS.items():
        mask = df["clean_body"].str.lower().apply(
            lambda t: any(kw in str(t) for kw in keywords)
        )
        df.loc[mask & ~df["return_mentioned"], "return_mentioned"] = True
        df.loc[mask & (df["return_type"] == ""), "return_type"]   = ret_type
    # return_reason = negative_category for return reviews
    df["return_reason"] = df.apply(
        lambda r: r["negative_category"]
        if r["return_mentioned"] and r.get("negative_category","") != "" else "",
        axis=1
    )
    return df


def _extract_pros(row) -> str:
    if row.get("sentiment_label") != "Positive":
        return ""
    text = str(row.get("clean_body", ""))
    found = [kw for kw in POSITIVE_KEYWORDS if kw in text.lower()]
    return ", ".join(found[:4]) if found else "Generally positive"


def _extract_cons(row) -> str:
    if row.get("sentiment_label") != "Negative":
        return ""
    cat = row.get("negative_category", "General issue")
    return cat if cat else "General issue"


def get_available_brands(df: pd.DataFrame) -> list:
    return sorted(df["brand"].dropna().unique().tolist())


def get_brand_df(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    return df[df["brand"].str.lower() == brand.lower()].copy()


def get_models(df: pd.DataFrame, brand: str) -> list:
    b = get_brand_df(df, brand)
    return sorted(b["product_title"].dropna().unique().tolist())


def compute_reputation_risk_score(df_brand: pd.DataFrame) -> float:
    if len(df_brand) == 0:
        return 0.0
    total     = len(df_brand)
    neg       = df_brand[df_brand["sentiment_label"] == "Negative"]
    neg_ratio = len(neg) / total
    avg_neg_conf = neg["confidence"].mean() / 100 if len(neg) > 0 else 0
    try:
        latest  = df_brand["date"].max()
        recent  = df_brand[df_brand["date"] >= latest - pd.Timedelta(days=90)]
        rec_neg = (recent["sentiment_label"] == "Negative").sum() / max(len(recent), 1)
        trend   = 1.25 if rec_neg > neg_ratio * 1.3 else 1.0
    except Exception:
        trend = 1.0
    score = (neg_ratio * 55) + (avg_neg_conf * 30) + (df_brand["impact_score"].mean() * 0.15)
    return round(min(score * trend, 100), 1)


def get_purchase_recommendation(df_model: pd.DataFrame) -> dict:
    """
    Generate a purchase recommendation for a specific phone model.
    Returns dict with verdict, score, pros, cons, summary.
    """
    if len(df_model) == 0:
        return {"verdict": "Insufficient Data", "color": "#6b7280", "score": 0}

    total = len(df_model)
    pos   = (df_model["sentiment_label"] == "Positive").sum()
    neg   = (df_model["sentiment_label"] == "Negative").sum()
    neu   = (df_model["sentiment_label"] == "Neutral").sum()
    pos_pct = pos / total * 100
    neg_pct = neg / total * 100
    avg_conf = df_model["confidence"].mean()
    avg_rating = df_model["rating"].mean()
    verified_pos = df_model[(df_model["verified"] == True) &
                            (df_model["sentiment_label"] == "Positive")]
    verified_neg = df_model[(df_model["verified"] == True) &
                            (df_model["sentiment_label"] == "Negative")]

    # Score 0-100
    score = (pos_pct * 0.5) + (avg_rating / 5 * 30) + (avg_conf / 100 * 20)

    # Top pros from positive reviews
    pos_reviews = df_model[df_model["sentiment_label"] == "Positive"]["clean_body"]
    all_pros = []
    for text in pos_reviews:
        found = [kw for kw in POSITIVE_KEYWORDS if kw in str(text).lower()]
        all_pros.extend(found)
    from collections import Counter
    top_pros = [kw for kw, _ in Counter(all_pros).most_common(5)]

    # Top cons from negative reviews
    neg_reviews = df_model[df_model["sentiment_label"] == "Negative"]
    top_cons_raw = neg_reviews["negative_category"].value_counts().head(3).index.tolist()

    # Verdict
    if pos_pct >= 65 and avg_rating >= 4.0:
        verdict = "✅ Highly Recommended"
        color   = "#22c55e"
        emoji   = "🟢"
    elif pos_pct >= 50 and avg_rating >= 3.5:
        verdict = "👍 Recommended"
        color   = "#4ade80"
        emoji   = "🟢"
    elif neg_pct >= 40 or avg_rating < 2.5:
        verdict = "❌ Not Recommended"
        color   = "#ef4444"
        emoji   = "🔴"
    elif neg_pct >= 25:
        verdict = "⚠️ Mixed Reviews — Proceed with Caution"
        color   = "#f59e0b"
        emoji   = "🟡"
    else:
        verdict = "🤔 Neutral — Average Product"
        color   = "#94a3b8"
        emoji   = "⚪"

    return {
        "verdict":      verdict,
        "color":        color,
        "emoji":        emoji,
        "score":        round(score, 1),
        "pos_pct":      round(pos_pct, 1),
        "neg_pct":      round(neg_pct, 1),
        "avg_rating":   round(avg_rating, 2),
        "total":        total,
        "verified_pos": len(verified_pos),
        "verified_neg": len(verified_neg),
        "pros":         top_pros if top_pros else ["Good performance", "Reliable"],
        "cons":         top_cons_raw if top_cons_raw else ["No major issues found"],
    }
