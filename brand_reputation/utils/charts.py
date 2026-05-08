import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud

POS  = "#22c55e"
NEG  = "#ef4444"
NEU  = "#f59e0b"
PRI  = "#6366f1"
SEC  = "#a855f7"
TXT  = "#f1f5f9"
GRID = "#334155"
CARD = "#1e293b"

CAT_COLORS = {
    "Defective Piece":        "#ef4444",
    "Price Dissatisfaction":  "#f97316",
    "Poor Customer Service":  "#a855f7",
    "Delivery / Packaging":   "#3b82f6",
    "General Dissatisfaction":"#6b7280",
}

BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TXT, family="Inter, sans-serif"),
    margin=dict(l=10, r=10, t=40, b=10),
)


# ── SHARED ────────────────────────────────────────────────────────────────────

def sentiment_donut(df: pd.DataFrame, title="Sentiment Split") -> go.Figure:
    counts = df["sentiment_label"].value_counts()
    color_map = {"Positive": POS, "Neutral": NEU, "Negative": NEG}
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.65,
        marker_colors=[color_map.get(l, PRI) for l in counts.index],
        textinfo="label+percent", textfont_size=13,
    ))
    fig.update_layout(
        **BASE, showlegend=False, height=300, title=title,
        annotations=[dict(text=f"{len(df):,}<br>Reviews",
                          x=0.5, y=0.5, font_size=15, font_color=TXT, showarrow=False)],
    )
    return fig


def sentiment_trend(df: pd.DataFrame) -> go.Figure:
    monthly = (df.groupby(["year_month","sentiment_label"])
                 .size().reset_index(name="count")
                 .sort_values("year_month"))
    pivot = (monthly.pivot(index="year_month", columns="sentiment_label", values="count")
                    .fillna(0).reset_index())
    fig = go.Figure()
    for col, color in [("Positive",POS),("Neutral",NEU),("Negative",NEG)]:
        if col in pivot.columns:
            fig.add_trace(go.Scatter(x=pivot["year_month"], y=pivot[col],
                name=col, mode="lines", line=dict(color=color, width=2.5)))
    fig.update_layout(**BASE, title="Sentiment Trend Over Time (RoBERTa)",
                      height=320, legend=dict(orientation="h", y=1.08))
    fig.update_xaxes(gridcolor=GRID, tickangle=-45, nticks=20)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def rating_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["rating"].value_counts().sort_index()
    colors = [NEG, NEG, NEU, POS, POS]
    fig = go.Figure(go.Bar(
        x=[f"{'⭐'*int(r)}" for r in counts.index],
        y=counts.values, marker_color=colors[:len(counts)],
        text=counts.values, textposition="outside",
    ))
    fig.update_layout(**BASE, title="Star Rating Distribution", height=280)
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def monthly_volume(df: pd.DataFrame) -> go.Figure:
    monthly = (df.groupby("year_month").size()
                 .reset_index(name="count").sort_values("year_month"))
    neg_m = (df[df["sentiment_label"]=="Negative"]
               .groupby("year_month").size().reset_index(name="neg_count"))
    pos_m = (df[df["sentiment_label"]=="Positive"]
               .groupby("year_month").size().reset_index(name="pos_count"))
    monthly = (monthly.merge(neg_m, on="year_month", how="left")
                       .merge(pos_m, on="year_month", how="left").fillna(0))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=monthly["year_month"], y=monthly["count"],
                         name="Total", marker_color="rgba(99,102,241,0.5)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=monthly["year_month"], y=monthly["neg_count"],
                             name="Negative", mode="lines",
                             line=dict(color=NEG, width=2)), secondary_y=True)
    fig.add_trace(go.Scatter(x=monthly["year_month"], y=monthly["pos_count"],
                             name="Positive", mode="lines",
                             line=dict(color=POS, width=2)), secondary_y=True)
    fig.update_layout(**BASE, title="Monthly Volume — Positive vs Negative Trend",
                      height=320, legend=dict(orientation="h", y=1.08))
    fig.update_xaxes(gridcolor=GRID, tickangle=-45, nticks=20)
    fig.update_yaxes(title_text="Total Reviews", secondary_y=False, gridcolor=GRID)
    fig.update_yaxes(title_text="Sentiment Count", secondary_y=True, gridcolor=GRID)
    return fig


def verified_vs_unverified(df: pd.DataFrame) -> go.Figure:
    grp = (df.groupby(["verified","sentiment_label"]).size()
             .reset_index(name="count"))
    grp["label"] = grp["verified"].map({True:"Verified Buyer", False:"Unverified"})
    fig = px.bar(grp, x="label", y="count", color="sentiment_label",
                 color_discrete_map={"Positive":POS,"Neutral":NEU,"Negative":NEG},
                 barmode="group", title="Verified vs Unverified Buyer Sentiment")
    fig.update_layout(**BASE, height=280)
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def confidence_distribution(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for label, color in [("Positive",POS),("Neutral",NEU),("Negative",NEG)]:
        s = df[df["sentiment_label"]==label]["confidence"]
        if len(s):
            fig.add_trace(go.Histogram(x=s, name=label, nbinsx=20,
                                       marker_color=color, opacity=0.75))
    fig.update_layout(**BASE, title="RoBERTa Confidence Score Distribution",
                      barmode="overlay", height=280,
                      legend=dict(orientation="h", y=1.08))
    fig.update_xaxes(title_text="Confidence (%)", gridcolor=GRID, range=[0,100])
    fig.update_yaxes(title_text="Count", gridcolor=GRID)
    return fig


def rating_vs_sentiment(df: pd.DataFrame) -> go.Figure:
    cross = pd.crosstab(df["rating"], df["sentiment_label"]).reset_index()
    fig = go.Figure()
    for col, color in [("Positive",POS),("Neutral",NEU),("Negative",NEG)]:
        if col in cross.columns:
            fig.add_trace(go.Bar(x=cross["rating"], y=cross[col],
                                 name=col, marker_color=color))
    fig.update_layout(**BASE, title="Star Rating vs RoBERTa Sentiment",
                      barmode="stack", height=300,
                      legend=dict(orientation="h", y=1.08))
    fig.update_xaxes(title_text="Star Rating", gridcolor=GRID)
    fig.update_yaxes(title_text="Count", gridcolor=GRID)
    return fig


def generate_wordcloud(df: pd.DataFrame, sentiment: str = "Negative"):
    subset = df[df["sentiment_label"]==sentiment]["clean_body"].dropna()
    if len(subset) == 0:
        return None
    text = " ".join(subset.tolist())
    stopwords = {
        "phone","samsung","apple","motorola","nokia","google","the","and","for",
        "this","that","with","have","from","not","but","are","was","its","been",
        "very","just","get","got","one","use","used","also","will","would","can",
        "had","has","it","is","in","to","of","a","i","my","me","we","you","your",
    }
    cmap = {"Negative":"Reds","Positive":"Greens","Neutral":"Blues"}
    wc = WordCloud(width=800, height=380, background_color="#0f172a",
                   colormap=cmap.get(sentiment,"viridis"), stopwords=stopwords,
                   max_words=80, prefer_horizontal=0.85, collocations=False).generate(text)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    fig.patch.set_facecolor("#0f172a")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


# ── COMPANY DASHBOARD CHARTS ──────────────────────────────────────────────────

def negative_category_bar(df: pd.DataFrame) -> go.Figure:
    neg = df[df["sentiment_label"]=="Negative"]
    if len(neg) == 0:
        return go.Figure()
    cat = neg["negative_category"].value_counts().reset_index()
    cat.columns = ["category","count"]
    fig = go.Figure(go.Bar(
        x=cat["count"], y=cat["category"], orientation="h",
        marker_color=[CAT_COLORS.get(c, PRI) for c in cat["category"]],
        text=cat["count"], textposition="outside",
    ))
    fig.update_layout(**BASE, title="Negative Review Sub-Categories", height=300)
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, categoryorder="total ascending")
    return fig


def impact_score_histogram(df: pd.DataFrame) -> go.Figure:
    neg = df[df["sentiment_label"]=="Negative"]
    if len(neg) == 0:
        return go.Figure()
    fig = go.Figure(go.Histogram(x=neg["impact_score"], nbinsx=20,
                                  marker_color=NEG, opacity=0.85))
    fig.update_layout(**BASE, title="Business Impact Score Distribution", height=260)
    fig.update_xaxes(title_text="Impact Score (0-100)", gridcolor=GRID)
    fig.update_yaxes(title_text="Count", gridcolor=GRID)
    return fig


def review_alert_chart(df: pd.DataFrame) -> go.Figure:
    daily = (df[df["sentiment_label"]=="Negative"]
               .groupby("date").size().reset_index(name="neg_count")
               .sort_values("date"))
    if len(daily) == 0:
        return go.Figure()
    daily["rolling_7d"] = daily["neg_count"].rolling(7, min_periods=1).mean()
    mean_val  = daily["rolling_7d"].mean()
    threshold = mean_val * 2
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["rolling_7d"],
                             name="7-day Avg Negative",
                             line=dict(color=NEG, width=2),
                             fill="tozeroy", fillcolor="rgba(239,68,68,0.1)"))
    fig.add_hline(y=threshold, line_dash="dash", line_color=NEU,
                  annotation_text="⚠️ Review Threshold (2× avg)",
                  annotation_position="top left",
                  annotation_font_color=NEU)
    fig.update_layout(**BASE, title="Review Early Warning — 7-Day Rolling Negative Rate",
                      height=320)
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def return_type_chart(df: pd.DataFrame) -> go.Figure:
    ret = df[df["return_mentioned"]==True]
    if len(ret) == 0:
        return go.Figure()
    counts = ret["return_type"].value_counts().reset_index()
    counts.columns = ["type","count"]
    colors = {"Completed Return":NEG,"Exchange / Replacement":SEC,
              "Return Attempted":NEU,"Return Failed":"#f97316"}
    fig = go.Figure(go.Bar(
        x=counts["count"], y=counts["type"], orientation="h",
        marker_color=[colors.get(t, PRI) for t in counts["type"]],
        text=counts["count"], textposition="outside",
    ))
    fig.update_layout(**BASE, title="Return Signal Breakdown", height=260)
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def return_trend(df: pd.DataFrame) -> go.Figure:
    ret = df[df["return_mentioned"]==True].copy()
    if len(ret) == 0:
        return go.Figure()
    monthly = ret.groupby("year_month").size().reset_index(name="returns")
    total_m = df.groupby("year_month").size().reset_index(name="total")
    monthly = monthly.merge(total_m, on="year_month", how="left")
    monthly["return_rate"] = (monthly["returns"]/monthly["total"]*100).round(2)
    monthly = monthly.sort_values("year_month")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=monthly["year_month"], y=monthly["returns"],
                         name="Return Mentions",
                         marker_color="rgba(239,68,68,0.6)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=monthly["year_month"], y=monthly["return_rate"],
                             name="Return Rate %", mode="lines+markers",
                             line=dict(color="#f97316", width=2)), secondary_y=True)
    fig.update_layout(**BASE, title="Monthly Return Mentions & Rate",
                      height=300, legend=dict(orientation="h", y=1.08))
    fig.update_xaxes(gridcolor=GRID, tickangle=-45, nticks=20)
    fig.update_yaxes(title_text="Return Mentions", secondary_y=False, gridcolor=GRID)
    fig.update_yaxes(title_text="Return Rate %",   secondary_y=True,  gridcolor=GRID)
    return fig


# ── CUSTOMER DASHBOARD CHARTS ─────────────────────────────────────────────────

def model_sentiment_bar(df_model: pd.DataFrame, model_name: str) -> go.Figure:
    counts = df_model["sentiment_label"].value_counts()
    color_map = {"Positive":POS,"Neutral":NEU,"Negative":NEG}
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=[color_map.get(l, PRI) for l in counts.index],
        text=counts.values, textposition="outside",
    ))
    fig.update_layout(**BASE, title=f"Review Breakdown — {model_name[:40]}", height=280)
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def positive_negative_split_gauge(pos_pct: float, neg_pct: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=pos_pct,
        domain={"x":[0,0.45],"y":[0,1]},
        title={"text":"Positive %","font":{"color":TXT,"size":13}},
        gauge={
            "axis":{"range":[0,100],"tickcolor":TXT},
            "bar":{"color":POS},
            "bgcolor":CARD,"bordercolor":GRID,
            "steps":[{"range":[0,50],"color":"#052e16"},{"range":[50,100],"color":"#14532d"}],
        },
        number={"font":{"color":POS,"size":28}},
    ))
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=neg_pct,
        domain={"x":[0.55,1],"y":[0,1]},
        title={"text":"Negative %","font":{"color":TXT,"size":13}},
        gauge={
            "axis":{"range":[0,100],"tickcolor":TXT},
            "bar":{"color":NEG},
            "bgcolor":CARD,"bordercolor":GRID,
            "steps":[{"range":[0,25],"color":"#052e16"},{"range":[25,100],"color":"#450a0a"}],
        },
        number={"font":{"color":NEG,"size":28}},
    ))
    fig.update_layout(**BASE, height=240)
    return fig


def top_positive_reviews_chart(df_model: pd.DataFrame) -> go.Figure:
    pos = df_model[df_model["sentiment_label"]=="Positive"]
    if len(pos) == 0:
        return go.Figure()
    yearly = (pos.groupby("year").size().reset_index(name="count")
                 .sort_values("year"))
    fig = go.Figure(go.Bar(
        x=yearly["year"], y=yearly["count"],
        marker_color=POS, text=yearly["count"], textposition="outside",
    ))
    fig.update_layout(**BASE, title="Positive Reviews by Year", height=260)
    fig.update_xaxes(gridcolor=GRID, title_text="Year")
    fig.update_yaxes(gridcolor=GRID, title_text="Count")
    return fig


def top_negative_reviews_chart(df_model: pd.DataFrame) -> go.Figure:
    neg = df_model[df_model["sentiment_label"]=="Negative"]
    if len(neg) == 0:
        return go.Figure()
    cat = neg["negative_category"].value_counts().reset_index()
    cat.columns = ["category","count"]
    fig = go.Figure(go.Bar(
        x=cat["count"], y=cat["category"], orientation="h",
        marker_color=[CAT_COLORS.get(c, NEG) for c in cat["category"]],
        text=cat["count"], textposition="outside",
    ))
    fig.update_layout(**BASE, title="What Customers Complain About", height=280)
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, categoryorder="total ascending")
    return fig


def brand_model_heatmap(df_brand: pd.DataFrame) -> go.Figure:
    top_models = df_brand["product_title"].value_counts().head(12).index.tolist()
    sub = df_brand[df_brand["product_title"].isin(top_models)]
    pivot = sub.groupby(["product_title","sentiment_label"]).size().unstack(fill_value=0)
    pivot["pos_pct"] = (pivot.get("Positive",0) / pivot.sum(axis=1) * 100).round(1)
    pivot = pivot.sort_values("pos_pct", ascending=True)
    short_names = [n[:35]+"..." if len(n)>35 else n for n in pivot.index]
    fig = go.Figure(go.Bar(
        x=pivot["pos_pct"], y=short_names, orientation="h",
        marker_color=[POS if v >= 60 else NEU if v >= 40 else NEG
                      for v in pivot["pos_pct"]],
        text=[f"{v:.0f}%" for v in pivot["pos_pct"]],
        textposition="outside",
    ))
    fig.update_layout(**BASE, title="Positive Review % by Model", height=400)
    fig.update_xaxes(gridcolor=GRID, title_text="Positive %", range=[0,110])
    fig.update_yaxes(gridcolor=GRID)
    return fig


# ── RETURN CHARTS (4 charts for t6 tab) ──────────────────────────────────────

def return_type_donut(df: pd.DataFrame) -> go.Figure:
    """Donut chart of return types — Completed / Attempted / Failed / Exchange."""
    ret = df[df["return_mentioned"] == True]
    if len(ret) == 0:
        return go.Figure()
    counts = ret["return_type"].value_counts()
    colors_map = {
        "Completed Return":       "#ef4444",
        "Return Attempted":       "#f97316",
        "Return Failed":          "#a855f7",
        "Exchange / Replacement": "#3b82f6",
    }
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.60,
        marker_colors=[colors_map.get(l, PRI) for l in counts.index],
        textinfo="label+percent",
        textfont_size=12,
    ))
    fig.update_layout(
        **BASE,
        showlegend=False,
        height=280,
        title="Return Type Breakdown",
        annotations=[dict(
            text=f"{len(ret):,}<br>Returns",
            x=0.5, y=0.5,
            font_size=14, font_color=TXT, showarrow=False,
        )],
    )
    return fig


def return_by_category(df: pd.DataFrame) -> go.Figure:
    """Which complaint category causes most returns."""
    ret = df[(df["return_mentioned"] == True) & (df["return_reason"] != "")]
    if len(ret) == 0:
        # fallback: use negative_category if return_reason empty
        ret = df[(df["return_mentioned"] == True) & (df["negative_category"] != "")]
        if len(ret) == 0:
            return go.Figure()
        col_use = "negative_category"
    else:
        col_use = "return_reason"

    cat_counts = ret[col_use].value_counts().reset_index()
    cat_counts.columns = ["category", "count"]
    fig = go.Figure(go.Bar(
        x=cat_counts["count"],
        y=cat_counts["category"],
        orientation="h",
        marker_color=[CAT_COLORS.get(c, PRI) for c in cat_counts["category"]],
        text=cat_counts["count"],
        textposition="outside",
    ))
    fig.update_layout(
        **BASE,
        title="Return Reasons by Complaint Category",
        height=280,
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, categoryorder="total ascending")
    return fig


def return_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Monthly return mentions and return rate % over time."""
    ret = df[df["return_mentioned"] == True].copy()
    if len(ret) == 0:
        return go.Figure()
    monthly     = ret.groupby("year_month").size().reset_index(name="return_count")
    total_monthly = df.groupby("year_month").size().reset_index(name="total")
    monthly = monthly.merge(total_monthly, on="year_month", how="left")
    monthly["return_rate"] = (monthly["return_count"] / monthly["total"] * 100).round(2)
    monthly = monthly.sort_values("year_month")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=monthly["year_month"], y=monthly["return_count"],
        name="Return Mentions", marker_color="rgba(239,68,68,0.6)",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=monthly["year_month"], y=monthly["return_rate"],
        name="Return Rate %", mode="lines+markers",
        line=dict(color="#f97316", width=2), marker=dict(size=4),
    ), secondary_y=True)
    fig.update_layout(
        **BASE,
        title="Monthly Return Mentions & Rate Over Time",
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(gridcolor=GRID, tickangle=-45, nticks=20)
    fig.update_yaxes(title_text="Return Mentions", secondary_y=False, gridcolor=GRID)
    fig.update_yaxes(title_text="Return Rate %",   secondary_y=True,  gridcolor=GRID)
    return fig


def return_by_product(df: pd.DataFrame) -> go.Figure:
    """Top 10 products with most return mentions."""
    ret = df[df["return_mentioned"] == True]
    if len(ret) == 0 or "product_title" not in ret.columns:
        return go.Figure()
    prod = ret["product_title"].value_counts().head(10).reset_index()
    prod.columns = ["product", "returns"]
    prod["product"] = prod["product"].apply(
        lambda x: str(x)[:45] + "..." if len(str(x)) > 45 else str(x)
    )
    fig = go.Figure(go.Bar(
        x=prod["returns"],
        y=prod["product"],
        orientation="h",
        marker_color=NEG,
        text=prod["returns"],
        textposition="outside",
    ))
    fig.update_layout(
        **BASE,
        title="Top 10 Products with Most Return Mentions",
        height=360,
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, categoryorder="total ascending")
    return fig
