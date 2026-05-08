import streamlit as st
import pandas as pd
import numpy as np
import os, sys

st.set_page_config(
    page_title="BrandSense — Reputation Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import (
    load_scored_data, scored_data_exists,
    get_available_brands, get_brand_df, get_models,
    compute_reputation_risk_score, get_purchase_recommendation,
)
from utils.charts import (
    sentiment_donut, sentiment_trend, rating_distribution,
    monthly_volume, verified_vs_unverified, confidence_distribution,
    rating_vs_sentiment, generate_wordcloud,
    negative_category_bar, impact_score_histogram,
    review_alert_chart, return_type_chart, return_trend,
    model_sentiment_bar, positive_negative_split_gauge,
    top_positive_reviews_chart, top_negative_reviews_chart,
    brand_model_heatmap,
    return_type_donut, return_by_category, return_trend_chart, return_by_product,
)
try:
    from utils.report_generator import (
        generate_company_report, generate_customer_report, REPORTLAB_AVAILABLE
    )
except Exception:
    REPORTLAB_AVAILABLE = False

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f172a;
    color: #f1f5f9;
}
.main .block-container { padding: 1.5rem 2rem; }
.metric-card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 12px; padding: 1.2rem 1.4rem; text-align: center;
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem; font-weight: 700; line-height: 1.1;
}
.metric-label {
    font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;
    letter-spacing: 0.08em; margin-top: 0.3rem;
}
.section-header {
    font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem;
    font-weight: 600; color: #f1f5f9; border-bottom: 1px solid #334155;
    padding-bottom: 0.5rem; margin: 1.5rem 0 1rem 0;
}
.review-card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.75rem;
}
.review-card.negative { border-left: 4px solid #ef4444; }
.review-card.positive { border-left: 4px solid #22c55e; }
.review-card.neutral  { border-left: 4px solid #f59e0b; }
.alert-red    { background: #450a0a; border-left: 4px solid #ef4444; border-radius: 8px; padding: 1rem 1.2rem; margin: 0.5rem 0; }
.alert-yellow { background: #422006; border-left: 4px solid #f59e0b; border-radius: 8px; padding: 1rem 1.2rem; margin: 0.5rem 0; }
.alert-green  { background: #052e16; border-left: 4px solid #22c55e; border-radius: 8px; padding: 1rem 1.2rem; margin: 0.5rem 0; }
.verdict-box {
    border-radius: 12px; padding: 1.5rem 2rem; margin: 1rem 0; text-align: center;
}
.pros-cons-card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 10px; padding: 1rem 1.2rem;
}
[data-testid="stSidebar"] { background: #1e293b; border-right: 1px solid #334155; }
.stTabs [data-baseweb="tab-list"] { background: #1e293b; border-radius: 8px; padding: 2px; gap: 4px; }
.stTabs [data-baseweb="tab"] { color: #94a3b8; border-radius: 6px; font-size: 0.85rem; }
.stTabs [aria-selected="true"] { background: #334155 !important; color: #f1f5f9 !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Check scored data ─────────────────────────────────────────────────────────
if not scored_data_exists():
    st.markdown("""
    <div style='text-align:center; padding:3rem 2rem;'>
      <div style='font-family:Space Grotesk; font-size:2.5rem; font-weight:700;
                  background:linear-gradient(135deg,#6366f1,#a855f7);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        BrandSense
      </div>
      <div style='color:#64748b; font-size:1rem; margin-top:0.5rem;'>
        Brand Reputation & Sentiment Intelligence Platform
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.error("⚠️ Scored data not found. Run the sentiment pipeline first.")
    st.markdown("""
    ### How to generate scored data
    ```bash
    # Samsung only — faster (~10 min on CPU)
    python run_sentiment.py --brand Samsung

    # All brands (~30-40 min on CPU)
    python run_sentiment.py

    # Or use Google Colab GPU (~5 min) — upload colab notebook
    ```
    After finishing, refresh this page.
    """)
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading BrandSense..."):
    df_master = load_scored_data()

brands = get_available_brands(df_master)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:0.5rem 0 0.2rem 0;'>
      <div style='font-family:Space Grotesk; font-size:1.3rem; font-weight:700;
                  background:linear-gradient(135deg,#6366f1,#a855f7);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        📊 BrandSense
      </div>
      <div style='font-size:0.7rem; color:#64748b;'>Reputation Intelligence Platform</div>
    </div>
    <hr style='border-color:#334155; margin:0.6rem 0;'>
    """, unsafe_allow_html=True)

    # ── DASHBOARD MODE ───────────────────────────────────────────────────────
    st.markdown("#### 👤 Dashboard Mode")
    mode = st.radio(
        "",
        ["🏢 Company Dashboard", "🛒 Customer Dashboard"],
        label_visibility="collapsed",
    )
    is_company = (mode == "🏢 Company Dashboard")

    st.markdown("<hr style='border-color:#334155; margin:0.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("#### 🎛️ Filters")

    brand_select = st.selectbox(
        "🏷️ Brand", brands,
        index=brands.index("Samsung") if "Samsung" in brands else 0,
    )

    years = sorted(df_master["year"].dropna().unique().astype(int).tolist())
    year_range = st.select_slider("📅 Year Range", options=years,
                                   value=(min(years), max(years)))

    only_verified = st.checkbox("✅ Verified Buyers Only", value=False)
    min_confidence = st.slider("🎯 Min AI Confidence (%)", 0, 100, 0)

    st.markdown("""
    <hr style='border-color:#334155;'>
    <div style='font-size:0.68rem; color:#475569; text-align:center;'>
      Model: cardiffnlp/twitter-roberta-base<br>
      Data: Amazon Mobile Reviews 2005–2019
    </div>
    """, unsafe_allow_html=True)

# ── Apply filters ─────────────────────────────────────────────────────────────
df_brand = get_brand_df(df_master, brand_select)
df_brand = df_brand[
    (df_brand["year"] >= year_range[0]) &
    (df_brand["year"] <= year_range[1])
]
if only_verified:
    df_brand = df_brand[df_brand["verified"] == True]
if min_confidence > 0:
    df_brand = df_brand[df_brand["confidence"] >= min_confidence]

df_negative = df_brand[df_brand["sentiment_label"] == "Negative"].copy()
df_positive = df_brand[df_brand["sentiment_label"] == "Positive"].copy()

# ── KPIs ─────────────────────────────────────────────────────────────────────
total     = len(df_brand)
pos_count = (df_brand["sentiment_label"] == "Positive").sum()
neg_count = (df_brand["sentiment_label"] == "Negative").sum()
neu_count = (df_brand["sentiment_label"] == "Neutral").sum()
pos_pct   = pos_count / total * 100 if total else 0
neg_pct   = neg_count / total * 100 if total else 0
avg_rating    = df_brand["rating"].mean() if total else 0
avg_confidence= df_brand["confidence"].mean() if total else 0
risk_score    = compute_reputation_risk_score(df_brand)
top_neg_cat   = (df_negative["negative_category"].value_counts().idxmax()
                 if len(df_negative) > 0 else "N/A")
star_sentiment = pd.cut(df_brand["rating"], bins=[0,2,3,5],
                         labels=["Negative","Neutral","Positive"])
disagree_rate  = (star_sentiment != df_brand["sentiment_label"]).mean()*100 if total else 0

if risk_score >= 60:
    risk_color="#ef4444"; risk_status="🔴 HIGH RISK";  alert_cls="alert-red"
elif risk_score >= 35:
    risk_color="#f59e0b"; risk_status="🟡 MODERATE";   alert_cls="alert-yellow"
else:
    risk_color="#22c55e"; risk_status="🟢 HEALTHY";    alert_cls="alert-green"


# ══════════════════════════════════════════════════════════════════════════════
#  COMPANY DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if is_company:
    # Header
    st.markdown(f"""
    <div style='display:flex; align-items:center; justify-content:space-between;
                padding:0.8rem 0 0.5rem 0; border-bottom:1px solid #334155; margin-bottom:1rem;'>
      <div>
        <div style='font-size:0.7rem; color:#6366f1; font-weight:600;
                    text-transform:uppercase; letter-spacing:0.1em;'>
          🏢 COMPANY INTELLIGENCE DASHBOARD
        </div>
        <div style='font-family:Space Grotesk; font-size:1.8rem; font-weight:700;'>{brand_select}</div>
        <div style='color:#64748b; font-size:0.82rem;'>
          {year_range[0]}–{year_range[1]} &nbsp;·&nbsp;
          {"Verified only" if only_verified else "All reviewers"} &nbsp;·&nbsp;
          {total:,} reviews &nbsp;·&nbsp;
          Avg RoBERTa confidence: <span style='color:#6366f1'>{avg_confidence:.1f}%</span>
        </div>
      </div>
      <div style='text-align:right;'>
        <div style='font-size:0.7rem; color:#64748b; text-transform:uppercase;'>Reputation Risk</div>
        <div style='font-family:Space Grotesk; font-size:2rem; font-weight:700;
                    color:{risk_color};'>{risk_score:.0f}/100</div>
        <div style='font-size:0.82rem; color:{risk_color};'>{risk_status}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    kpis = [
        (f"{total:,}",             "Total Reviews",         "#6366f1"),
        (f"{pos_pct:.1f}%",        "Positive (RoBERTa)",    "#22c55e"),
        (f"{neg_pct:.1f}%",        "Negative (RoBERTa)",    "#ef4444"),
        (f"{avg_rating:.2f} ⭐",   "Avg Star Rating",       "#f59e0b"),
        (f"{avg_confidence:.1f}%", "Avg AI Confidence",     "#a855f7"),
        (f"{disagree_rate:.1f}%",  "Rating vs AI Mismatch", "#3b82f6"),
    ]
    for col,(val,label,color) in zip([k1,k2,k3,k4,k5,k6],kpis):
        col.markdown(f"""<div class='metric-card'>
          <div class='metric-value' style='color:{color};'>{val}</div>
          <div class='metric-label'>{label}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    t1,t2,t3,t4,t5,t6,t7 = st.tabs([
        "📊 Overview","🤖 RoBERTa Analysis","⚠️ Review Monitor",
        "🔍 Review Explorer","☁️ Word Cloud","🔄 Returns","📄 Report",
    ])

    # ── TAB 1: Overview
    with t1:
        c1,c2 = st.columns([1,2])
        with c1:
            st.plotly_chart(sentiment_donut(df_brand), use_container_width=True)
            st.plotly_chart(rating_distribution(df_brand), use_container_width=True)
        with c2:
            st.plotly_chart(sentiment_trend(df_brand), use_container_width=True)
            st.plotly_chart(monthly_volume(df_brand), use_container_width=True)

        c3,c4 = st.columns(2)
        with c3:
            st.plotly_chart(negative_category_bar(df_brand), use_container_width=True)
        with c4:
            st.plotly_chart(verified_vs_unverified(df_brand), use_container_width=True)
        st.plotly_chart(impact_score_histogram(df_brand), use_container_width=True)

    # ── TAB 2: RoBERTa Analysis
    with t2:
        st.markdown("<div class='section-header'>🤖 RoBERTa Sentiment Deep Dive</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#1e293b; border:1px solid #334155; border-radius:10px;
                    padding:1rem 1.2rem; margin-bottom:1rem; font-size:0.85rem; color:#94a3b8;'>
          <b style='color:#6366f1;'>Model:</b> cardiffnlp/twitter-roberta-base-sentiment-latest &nbsp;·&nbsp;
          <b style='color:#6366f1;'>What it does:</b> Reads actual review TEXT — not star ratings — to
          classify sentiment. The mismatch chart shows where AI and star ratings disagree,
          revealing hidden brand issues invisible to traditional monitoring.
        </div>
        """, unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            st.plotly_chart(confidence_distribution(df_brand), use_container_width=True)
        with c2:
            st.plotly_chart(rating_vs_sentiment(df_brand), use_container_width=True)

        # Mismatch reviews
        st.markdown("<div class='section-header'>⚡ Hidden Problems — High Stars but Negative Text</div>", unsafe_allow_html=True)
        mismatch = df_brand[(df_brand["rating"]>=4)&(df_brand["sentiment_label"]=="Negative")].nlargest(8,"confidence")
        if len(mismatch)==0:
            st.info("No high-star negative reviews found.")
        else:
            for _,row in mismatch.iterrows():
                st.markdown(f"""
                <div class='review-card negative'>
                  <div style='display:flex; justify-content:space-between; margin-bottom:0.3rem;'>
                    <span style='font-weight:600;'>{str(row.get('review_title',''))[:70]}</span>
                    <span style='color:#a855f7; font-size:0.8rem;'>AI confidence: {row['confidence']:.0f}%</span>
                  </div>
                  <div style='font-size:0.78rem; color:#94a3b8; margin-bottom:0.4rem;'>
                    {'⭐'*int(row['rating'])} Star &nbsp;·&nbsp; AI: <span style='color:#ef4444;'>Negative</span>
                    &nbsp;·&nbsp; {str(row['date'])[:10]}
                  </div>
                  <div style='font-size:0.87rem; color:#cbd5e1; line-height:1.5;'>
                    {str(row['clean_body'])[:300]}{'...' if len(str(row['clean_body']))>300 else ''}
                  </div>
                </div>""", unsafe_allow_html=True)

    # ── TAB 3: Review Monitor
    with t3:
        st.markdown("<div class='section-header'>⚠️ Reputation Review Early Warning</div>", unsafe_allow_html=True)

        daily_neg = (df_brand[df_brand["sentiment_label"]=="Negative"]
                       .groupby("date").size().reset_index(name="count")
                       .sort_values("date"))
        if len(daily_neg) > 7:
            rolling      = daily_neg["count"].rolling(7, min_periods=1).mean()
            overall_mean = rolling.mean()
            recent_mean  = rolling.tail(30).mean()
            spike_ratio  = recent_mean / overall_mean if overall_mean > 0 else 1
            if spike_ratio >= 2.0:
                msg = f"🚨 REVIEW ALERT: Negative reviews are {spike_ratio:.1f}× above normal. Immediate action required."
                cls = "alert-red"
            elif spike_ratio >= 1.4:
                msg = f"⚠️ WARNING: Negative review rate is {spike_ratio:.1f}× above baseline. Monitor closely."
                cls = "alert-yellow"
            else:
                msg = f"✅ STABLE: Brand is within normal range ({spike_ratio:.1f}× baseline)."
                cls = "alert-green"
            st.markdown(f"<div class='{cls}'>{msg}</div>", unsafe_allow_html=True)

        st.plotly_chart(review_alert_chart(df_brand), use_container_width=True)

        st.markdown("<div class='section-header'>🔥 Highest Business Impact Negative Reviews</div>", unsafe_allow_html=True)
        top_impact = df_negative.nlargest(10,"impact_score")
        for _,row in top_impact.iterrows():
            st.markdown(f"""
            <div class='review-card negative'>
              <div style='display:flex; justify-content:space-between; margin-bottom:0.3rem;'>
                <span style='font-weight:600;'>{str(row.get('review_title',''))[:75]}</span>
                <span style='color:#ef4444; font-weight:700;'>Impact: {row['impact_score']:.0f}/100</span>
              </div>
              <div style='font-size:0.78rem; color:#94a3b8; margin-bottom:0.4rem;'>
                {'⭐'*int(row['rating'])} &nbsp;·&nbsp;
                <span style='color:#f97316;'>{row['negative_category']}</span> &nbsp;·&nbsp;
                {'✅ Verified' if row['verified'] else '❓'} &nbsp;·&nbsp;
                AI: <span style='color:#a855f7;'>{row['confidence']:.0f}%</span> &nbsp;·&nbsp;
                {str(row['date'])[:10]}
              </div>
              <div style='font-size:0.87rem; color:#cbd5e1; line-height:1.5;'>
                {str(row['clean_body'])[:280]}{'...' if len(str(row['clean_body']))>280 else ''}
              </div>
            </div>""", unsafe_allow_html=True)

    # ── TAB 4: Review Explorer
    with t4:
        st.markdown("<div class='section-header'>🔍 Review Explorer</div>", unsafe_allow_html=True)
        f1,f2,f3,f4 = st.columns(4)
        with f1:
            sent_filter = st.multiselect("Sentiment", ["Positive","Neutral","Negative"], default=["Negative"])
        with f2:
            cat_filter = st.multiselect("Category", options=sorted(df_negative["negative_category"].unique().tolist()), default=[])
        with f3:
            search_term = st.text_input("🔎 Search", placeholder="battery, screen...")
        with f4:
            min_cf = st.slider("Min Confidence", 0, 100, 50)

        view_df = df_brand.copy()
        if sent_filter:
            view_df = view_df[view_df["sentiment_label"].isin(sent_filter)]
        if cat_filter:
            view_df = view_df[view_df["negative_category"].isin(cat_filter)]
        if search_term:
            mask = (view_df["clean_body"].str.contains(search_term,case=False,na=False)|
                    view_df["review_title"].str.contains(search_term,case=False,na=False))
            view_df = view_df[mask]
        view_df = view_df[view_df["confidence"]>=min_cf].sort_values("confidence",ascending=False)
        st.markdown(f"<div style='color:#64748b; font-size:0.83rem; margin-bottom:0.8rem;'>Showing {min(len(view_df),25):,} of {len(view_df):,} reviews</div>", unsafe_allow_html=True)
        for _,row in view_df.head(25).iterrows():
            s   = str(row.get("sentiment_label","neutral")).lower()
            cls = "positive" if s=="positive" else "negative" if s=="negative" else "neutral"
            conf_color = "#22c55e" if row["confidence"]>=80 else "#f59e0b" if row["confidence"]>=60 else "#ef4444"
            cat_txt = f"· <span style='color:#f97316;'>{row['negative_category']}</span>" if row.get("negative_category") else ""
            st.markdown(f"""
            <div class='review-card {cls}'>
              <div style='display:flex; justify-content:space-between; margin-bottom:0.3rem;'>
                <span style='font-weight:600;'>{str(row.get('review_title',''))[:70]}</span>
                <span style='color:{conf_color}; font-size:0.78rem; font-weight:600;'>
                  {row['sentiment_label']} · {row['confidence']:.0f}%
                </span>
              </div>
              <div style='font-size:0.78rem; color:#94a3b8; margin-bottom:0.4rem;'>
                {'⭐'*int(row['rating'])} &nbsp; {cat_txt} &nbsp;·&nbsp;
                {'✅' if row['verified'] else '❓'} &nbsp;·&nbsp; {str(row['date'])[:10]}
              </div>
              <div style='font-size:0.87rem; color:#cbd5e1; line-height:1.5;'>
                {str(row['clean_body'])[:320]}{'...' if len(str(row['clean_body']))>320 else ''}
              </div>
            </div>""", unsafe_allow_html=True)

    # ── TAB 5: Word Cloud
    with t5:
        st.markdown("<div class='section-header'>☁️ Word Cloud by Sentiment</div>", unsafe_allow_html=True)
        st.markdown("<div style='color:#94a3b8; font-size:0.83rem; margin-bottom:1rem;'>Generated from cleaned review text. Toggle between Positive and Negative to see what words customers use most.</div>", unsafe_allow_html=True)
        wc_sent = st.selectbox("Sentiment", ["Negative","Positive","Neutral"])
        fig_wc  = generate_wordcloud(df_brand, sentiment=wc_sent)
        if fig_wc:
            st.pyplot(fig_wc, use_container_width=True)
        else:
            st.info(f"Not enough {wc_sent} reviews for word cloud.")

    # ── TAB 6: Returns
    with t6:
        st.markdown("<div class='section-header'>🔄 Product Return Signal Analysis</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#1e293b; border:1px solid #334155; border-radius:10px;
                    padding:1rem 1.2rem; margin-bottom:1rem; font-size:0.85rem; color:#94a3b8;'>
          <b style='color:#6366f1;'>How this works:</b> Amazon review data does not contain
          actual transaction or return records. This analysis detects <b>return signals</b> —
          keywords like &quot;returned it&quot;, &quot;sent it back&quot;, &quot;asked for refund&quot;, &quot;got a replacement&quot;
          — in review text to estimate return intent and return outcomes.
          This is the same approach used in industry when transaction data is unavailable.
        </div>
        """, unsafe_allow_html=True)

        df_ret        = df_brand[df_brand["return_mentioned"] == True]
        total_ret     = len(df_ret)
        ret_rate      = total_ret / len(df_brand) * 100 if len(df_brand) > 0 else 0
        completed_ret = (df_ret["return_type"] == "Completed Return").sum()
        failed_ret    = (df_ret["return_type"] == "Return Failed").sum()
        exchange_ret  = (df_ret["return_type"] == "Exchange / Replacement").sum()

        rk1,rk2,rk3,rk4 = st.columns(4)
        ret_kpis = [
            (f"{total_ret:,}",          "Return Mentions",         "#ef4444"),
            (f"{ret_rate:.1f}%",        "Est. Return Signal Rate", "#f97316"),
            (f"{completed_ret:,}",      "Confirmed Returns",       "#a855f7"),
            (f"{exchange_ret:,}",       "Exchanges / Replacements","#3b82f6"),
        ]
        for col,(val,label,color) in zip([rk1,rk2,rk3,rk4], ret_kpis):
            col.markdown(f"""<div class='metric-card'>
              <div class='metric-value' style='color:{color};'>{val}</div>
              <div class='metric-label'>{label}</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        rc1,rc2 = st.columns(2)
        with rc1:
            st.plotly_chart(return_type_donut(df_brand), use_container_width=True)
        with rc2:
            st.plotly_chart(return_by_category(df_brand), use_container_width=True)

        st.plotly_chart(return_trend_chart(df_brand), use_container_width=True)
        st.plotly_chart(return_by_product(df_brand), use_container_width=True)

        # Return reviews cards
        st.markdown("<div class='section-header'>📝 Reviews Mentioning Returns</div>", unsafe_allow_html=True)
        ret_filter_opts = ["All"] + sorted(df_ret["return_type"].dropna().unique().tolist()) if len(df_ret) > 0 else ["All"]
        ret_filter = st.selectbox("Filter by Return Type", ret_filter_opts, key="ret_type_filter")
        show_ret = df_ret if ret_filter == "All" else df_ret[df_ret["return_type"] == ret_filter]
        show_ret = show_ret.sort_values("date", ascending=False)
        st.markdown(f"<div style='color:#64748b; font-size:0.83rem; margin-bottom:0.8rem;'>Showing {min(len(show_ret),15):,} of {len(show_ret):,} return-related reviews</div>", unsafe_allow_html=True)

        if len(show_ret) == 0:
            st.info("No return-related reviews found for this brand/filter.")
        else:
            type_colors = {
                "Completed Return":      "#ef4444",
                "Return Attempted":      "#f97316",
                "Return Failed":         "#a855f7",
                "Exchange / Replacement":"#3b82f6",
            }
            for _,row in show_ret.head(15).iterrows():
                type_color = type_colors.get(row["return_type"], "#6b7280")
                st.markdown(f"""
                <div class='review-card negative'>
                  <div style='display:flex; justify-content:space-between; margin-bottom:0.3rem;'>
                    <span style='font-weight:600;'>{str(row.get("review_title",""))[:70]}</span>
                    <span style='color:{type_color}; font-size:0.8rem; font-weight:600;'>{row["return_type"]}</span>
                  </div>
                  <div style='font-size:0.78rem; color:#94a3b8; margin-bottom:0.4rem;'>
                    {"⭐"*int(row["rating"])} &nbsp;·&nbsp;
                    <span style='color:#f97316;'>{row.get("negative_category","")}</span> &nbsp;·&nbsp;
                    {"✅ Verified" if row["verified"] else "❓"} &nbsp;·&nbsp; {str(row["date"])[:10]}
                  </div>
                  <div style='font-size:0.87rem; color:#cbd5e1; line-height:1.5;'>
                    {str(row["clean_body"])[:320]}{"..." if len(str(row["clean_body"])) > 320 else ""}
                  </div>
                </div>""", unsafe_allow_html=True)

    # ── TAB 7: Report
    with t7:
        st.markdown("<div class='section-header'>📄 Company Intelligence Report</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:#1e293b; border:1px solid #334155; border-radius:10px;
                    padding:1rem 1.2rem; margin-bottom:1rem; font-size:0.85rem; color:#94a3b8;'>
          Generate a complete PDF brand intelligence report for <b style='color:#6366f1;'>{brand_select}</b>
          including reputation risk score, sentiment breakdown, negative categories,
          highest impact reviews, and return analysis.
        </div>
        """, unsafe_allow_html=True)

        # Summary stats table
        summary = pd.DataFrame({
            "Metric": [
                "Total Reviews","Positive (RoBERTa)","Negative (RoBERTa)",
                "Neutral (RoBERTa)","Average Star Rating","Average AI Confidence",
                "Rating vs AI Mismatch","Top Negative Category",
                "Verified Buyer %","Reputation Risk Score",
            ],
            "Value": [
                f"{total:,}",
                f"{pos_count:,} ({pos_pct:.1f}%)",
                f"{neg_count:,} ({neg_pct:.1f}%)",
                f"{neu_count:,} ({100-pos_pct-neg_pct:.1f}%)",
                f"{avg_rating:.2f} / 5.0",
                f"{avg_confidence:.1f}%",
                f"{disagree_rate:.1f}%",
                top_neg_cat,
                f"{df_brand['verified'].mean()*100:.1f}%",
                f"{risk_score:.1f} / 100 — {risk_status}",
            ]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        col1,col2 = st.columns(2)
        with col1:
            if REPORTLAB_AVAILABLE:
                if "comp_pdf_ready" not in st.session_state:
                    st.session_state["comp_pdf_ready"] = False
                if st.button("📄 Generate PDF Report", type="primary", key="gen_comp_pdf"):
                    with st.spinner("Generating company intelligence report..."):
                        try:
                            pdf_bytes = generate_company_report(
                                brand_select, df_brand, risk_score,
                                top_neg_cat, pos_pct, neg_pct, avg_rating
                            )
                            st.session_state["comp_pdf_bytes"] = pdf_bytes
                            st.session_state["comp_pdf_ready"] = True
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            import traceback; st.code(traceback.format_exc())
                if st.session_state.get("comp_pdf_ready"):
                    st.download_button(
                        label="⬇️ Download Company Report PDF",
                        data=st.session_state["comp_pdf_bytes"],
                        file_name=f"{brand_select}_company_report.pdf",
                        mime="application/pdf",
                        key="dl_comp_pdf",
                    )
                    st.success("✅ Report ready!")
            else:
                st.warning("Install reportlab: pip install reportlab")
        with col2:
            csv = df_brand[["date","rating","sentiment_label","confidence",
                            "negative_category","impact_score","verified",
                            "review_title","clean_body"]].to_csv(index=False)
            st.download_button("⬇️ Download Data as CSV", csv,
                               f"{brand_select}_data.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOMER DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown(f"""
    <div style='display:flex; align-items:center; justify-content:space-between;
                padding:0.8rem 0 0.5rem 0; border-bottom:1px solid #334155; margin-bottom:1rem;'>
      <div>
        <div style='font-size:0.7rem; color:#22c55e; font-weight:600;
                    text-transform:uppercase; letter-spacing:0.1em;'>
          🛒 CUSTOMER BUYING GUIDE
        </div>
        <div style='font-family:Space Grotesk; font-size:1.8rem; font-weight:700;'>
          Should You Buy This Phone?
        </div>
        <div style='color:#64748b; font-size:0.82rem;'>
          AI-powered purchase guidance based on {total:,} real customer reviews for {brand_select}
        </div>
      </div>
      <div style='text-align:right;'>
        <div style='font-size:0.7rem; color:#64748b;'>Overall Brand Sentiment</div>
        <div style='font-family:Space Grotesk; font-size:2rem; font-weight:700; color:#22c55e;'>{pos_pct:.0f}% 👍</div>
        <div style='font-size:0.82rem; color:#94a3b8;'>Positive reviews</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Model selector — prominent
    st.markdown("<div class='section-header'>🔍 Select a Phone Model to Analyze</div>", unsafe_allow_html=True)
    models = get_models(df_master, brand_select)
    if not models:
        st.warning("No models found for this brand.")
        st.stop()

    selected_model = st.selectbox("📱 Choose Phone Model", models)
    df_model = df_brand[df_brand["product_title"] == selected_model].copy()

    # Get recommendation
    rec = get_purchase_recommendation(df_model)

    # ── VERDICT BANNER ───────────────────────────────────────────────────────
    st.markdown(f"""
    <div class='verdict-box' style='background: {rec["color"]}18; border: 2px solid {rec["color"]};'>
      <div style='font-size:2.2rem; margin-bottom:0.3rem;'>{rec.get("emoji","")}</div>
      <div style='font-family:Space Grotesk; font-size:1.6rem; font-weight:700;
                  color:{rec["color"]};'>{rec["verdict"]}</div>
      <div style='color:#94a3b8; font-size:0.9rem; margin-top:0.4rem;'>
        Based on {rec["total"]:,} customer reviews &nbsp;·&nbsp;
        {rec["pos_pct"]:.0f}% positive &nbsp;·&nbsp;
        {rec["neg_pct"]:.0f}% negative &nbsp;·&nbsp;
        ⭐ {rec["avg_rating"]:.2f}/5.0
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PROS AND CONS ────────────────────────────────────────────────────────
    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown("""<div style='background:#052e16; border:1px solid #22c55e;
                        border-radius:10px; padding:1rem 1.2rem;'>
          <div style='color:#22c55e; font-weight:700; font-size:0.9rem; margin-bottom:0.6rem;'>
            ✅ WHAT CUSTOMERS LOVE
          </div>""", unsafe_allow_html=True)
        for pro in rec["pros"]:
            st.markdown(f"<div style='color:#cbd5e1; font-size:0.85rem; padding:0.2rem 0;'>• {pro.title()}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with pc2:
        st.markdown("""<div style='background:#450a0a; border:1px solid #ef4444;
                        border-radius:10px; padding:1rem 1.2rem;'>
          <div style='color:#ef4444; font-weight:700; font-size:0.9rem; margin-bottom:0.6rem;'>
            ❌ COMMON COMPLAINTS
          </div>""", unsafe_allow_html=True)
        for con in rec["cons"]:
            st.markdown(f"<div style='color:#cbd5e1; font-size:0.85rem; padding:0.2rem 0;'>• {con}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CUSTOMER TABS ────────────────────────────────────────────────────────
    ct1,ct2,ct3,ct4,ct5 = st.tabs([
        "📊 Sentiment Overview","💚 Positive Reviews",
        "❤️ Negative Reviews","🏷️ All Brand Models","📄 Buying Guide",
    ])

    # ── CT1: Sentiment Overview
    with ct1:
        c1,c2 = st.columns([1,1])
        with c1:
            st.plotly_chart(model_sentiment_bar(df_model, selected_model[:30]),
                            use_container_width=True)
        with c2:
            st.plotly_chart(positive_negative_split_gauge(rec["pos_pct"], rec["neg_pct"]),
                            use_container_width=True)

        st.plotly_chart(rating_vs_sentiment(df_model), use_container_width=True)

        # Verified vs Unverified summary
        v_pos = rec["verified_pos"]
        v_neg = rec["verified_neg"]
        vm1,vm2,vm3,vm4 = st.columns(4)
        for col,(val,label,color) in zip([vm1,vm2,vm3,vm4],[
            (f"{rec['total']:,}","Reviews Analyzed","#6366f1"),
            (f"{rec['avg_rating']:.2f} ⭐","Avg Star Rating","#f59e0b"),
            (f"{v_pos:,}","Verified Positive","#22c55e"),
            (f"{v_neg:,}","Verified Negative","#ef4444"),
        ]):
            col.markdown(f"""<div class='metric-card'>
              <div class='metric-value' style='color:{color};'>{val}</div>
              <div class='metric-label'>{label}</div></div>""", unsafe_allow_html=True)

    # ── CT2: Positive Reviews
    with ct2:
        st.markdown("<div class='section-header'>💚 What Customers Love About This Phone</div>", unsafe_allow_html=True)

        if len(df_positive[df_positive["product_title"]==selected_model]) == 0:
            st.info("No positive reviews found for this model.")
        else:
            pos_model = df_positive[df_positive["product_title"]==selected_model]
            st.plotly_chart(top_positive_reviews_chart(pos_model), use_container_width=True)

            # Sort options
            sort_pos = st.selectbox("Sort positive reviews by", ["Confidence","Date (newest)","Helpful Votes"])
            if sort_pos == "Confidence":
                pos_model = pos_model.sort_values("confidence", ascending=False)
            elif sort_pos == "Date (newest)":
                pos_model = pos_model.sort_values("date", ascending=False)
            else:
                pos_model = pos_model.sort_values("helpfulVotes", ascending=False)

            st.markdown(f"<div style='color:#64748b; font-size:0.83rem; margin-bottom:0.8rem;'>{len(pos_model):,} positive reviews found</div>", unsafe_allow_html=True)

            for _,row in pos_model.head(20).iterrows():
                st.markdown(f"""
                <div class='review-card positive'>
                  <div style='display:flex; justify-content:space-between; margin-bottom:0.3rem;'>
                    <span style='font-weight:600;'>{str(row.get('review_title',''))[:70]}</span>
                    <span style='color:#22c55e; font-size:0.78rem;'>AI: {row['confidence']:.0f}% confident</span>
                  </div>
                  <div style='font-size:0.78rem; color:#94a3b8; margin-bottom:0.4rem;'>
                    {'⭐'*int(row['rating'])} &nbsp;·&nbsp;
                    {'✅ Verified' if row['verified'] else '❓'} &nbsp;·&nbsp;
                    👍 {int(row['helpfulVotes'])} helpful &nbsp;·&nbsp; {str(row['date'])[:10]}
                  </div>
                  <div style='font-size:0.87rem; color:#cbd5e1; line-height:1.5;'>
                    {str(row['clean_body'])[:320]}{'...' if len(str(row['clean_body']))>320 else ''}
                  </div>
                </div>""", unsafe_allow_html=True)

    # ── CT3: Negative Reviews
    with ct3:
        st.markdown("<div class='section-header'>❤️ What Customers Complain About</div>", unsafe_allow_html=True)

        neg_model = df_negative[df_negative["product_title"]==selected_model]
        if len(neg_model) == 0:
            st.success("🎉 No significant negative reviews found for this model!")
        else:
            st.plotly_chart(top_negative_reviews_chart(neg_model), use_container_width=True)

            cat_filter_c = st.selectbox("Filter by complaint type",
                ["All"] + sorted(neg_model["negative_category"].unique().tolist()))
            show_neg = neg_model if cat_filter_c=="All" else neg_model[neg_model["negative_category"]==cat_filter_c]
            show_neg = show_neg.sort_values("confidence", ascending=False)

            st.markdown(f"<div style='color:#64748b; font-size:0.83rem; margin-bottom:0.8rem;'>{len(show_neg):,} negative reviews</div>", unsafe_allow_html=True)

            for _,row in show_neg.head(20).iterrows():
                st.markdown(f"""
                <div class='review-card negative'>
                  <div style='display:flex; justify-content:space-between; margin-bottom:0.3rem;'>
                    <span style='font-weight:600;'>{str(row.get('review_title',''))[:70]}</span>
                    <span style='color:#ef4444; font-size:0.78rem;'>AI: {row['confidence']:.0f}%</span>
                  </div>
                  <div style='font-size:0.78rem; color:#94a3b8; margin-bottom:0.4rem;'>
                    {'⭐'*int(row['rating'])} &nbsp;·&nbsp;
                    <span style='color:#f97316;'>{row['negative_category']}</span> &nbsp;·&nbsp;
                    {'✅ Verified' if row['verified'] else '❓'} &nbsp;·&nbsp; {str(row['date'])[:10]}
                  </div>
                  <div style='font-size:0.87rem; color:#cbd5e1; line-height:1.5;'>
                    {str(row['clean_body'])[:320]}{'...' if len(str(row['clean_body']))>320 else ''}
                  </div>
                </div>""", unsafe_allow_html=True)

    # ── CT4: All Brand Models
    with ct4:
        st.markdown(f"<div class='section-header'>🏷️ All {brand_select} Models — Sentiment Ranking</div>", unsafe_allow_html=True)
        st.markdown("<div style='color:#94a3b8; font-size:0.83rem; margin-bottom:1rem;'>Compare all models by positive review percentage. Green = safe to buy, Red = avoid.</div>", unsafe_allow_html=True)
        st.plotly_chart(brand_model_heatmap(df_brand), use_container_width=True)

        # Model quick comparison table
        model_stats = []
        for m in df_brand["product_title"].unique():
            dm = df_brand[df_brand["product_title"]==m]
            if len(dm) < 5:
                continue
            p = (dm["sentiment_label"]=="Positive").mean()*100
            n = (dm["sentiment_label"]=="Negative").mean()*100
            r = dm["rating"].mean()
            verdict = "✅ Buy" if p>=65 else "⚠️ Maybe" if p>=50 else "❌ Avoid"
            model_stats.append({"Model":m[:40],"Positive%":f"{p:.0f}%",
                                  "Negative%":f"{n:.0f}%","Avg Rating":f"{r:.1f}⭐",
                                  "Reviews":len(dm),"Verdict":verdict})
        if model_stats:
            stats_df = pd.DataFrame(model_stats).sort_values("Positive%", ascending=False)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # ── CT5: Buying Guide PDF
    with ct5:
        st.markdown("<div class='section-header'>📄 Personal Buying Guide</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:#1e293b; border:1px solid #334155; border-radius:10px;
                    padding:1rem 1.2rem; margin-bottom:1rem; font-size:0.85rem; color:#94a3b8;'>
          Generate a complete PDF buying guide for <b style='color:#22c55e;'>{selected_model}</b>
          including the purchase verdict, pros & cons, sentiment analysis, and
          sample positive/negative reviews from real customers.
        </div>
        """, unsafe_allow_html=True)

        if REPORTLAB_AVAILABLE:
            if "cust_pdf_ready" not in st.session_state:
                st.session_state["cust_pdf_ready"] = False
            if st.button("📄 Generate Buying Guide PDF", type="primary", key="gen_cust_pdf"):
                with st.spinner("Generating your personalised buying guide..."):
                    try:
                        pdf_bytes = generate_customer_report(
                            brand_select, selected_model, rec, df_model
                        )
                        st.session_state["cust_pdf_bytes"] = pdf_bytes
                        st.session_state["cust_pdf_ready"] = True
                        st.session_state["cust_pdf_name"] = selected_model.replace(" ","_").replace("/","_")[:50]
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")
                        import traceback; st.code(traceback.format_exc())
            if st.session_state.get("cust_pdf_ready"):
                st.download_button(
                    label="⬇️ Download Buying Guide PDF",
                    data=st.session_state["cust_pdf_bytes"],
                    file_name=f"{st.session_state.get('cust_pdf_name','guide')}_buying_guide.pdf",
                    mime="application/pdf",
                    key="dl_cust_pdf",
                )
                st.success("✅ Buying guide ready!")
        else:
            st.warning("Install reportlab: pip install reportlab")