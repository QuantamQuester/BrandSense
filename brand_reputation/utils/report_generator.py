"""
report_generator.py
Beautiful, well-structured PDF reports for BrandSense.
Uses WHITE background with dark text for maximum readability.
Includes full return signal analysis in both reports.
"""
import io
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.colors import HexColor, white, black
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table,
        TableStyle, HRFlowable, PageBreak, KeepTogether,
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

W, H = A4

C_PURPLE   = HexColor("#5B5FCF")
C_DARK     = HexColor("#1e1b4b")
C_GREEN    = HexColor("#16a34a")
C_RED      = HexColor("#dc2626")
C_ORANGE   = HexColor("#ea580c")
C_YELLOW   = HexColor("#ca8a04")
C_GRAY     = HexColor("#6b7280")
C_LIGHT_BG = HexColor("#f8fafc")
C_BORDER   = HexColor("#e2e8f0")
C_TEXT     = HexColor("#1e293b")
C_SUBTEXT  = HexColor("#475569")
C_ROW_ALT  = HexColor("#f1f5f9")
C_HDR_BG   = HexColor("#5B5FCF")
C_PALE_GRN = HexColor("#f0fdf4")
C_PALE_RED = HexColor("#fff8f8")
C_PALE_YLW = HexColor("#fefce8")


def S(name, **kw):
    d = dict(fontName="Helvetica", fontSize=9, textColor=C_TEXT,
             leading=14, spaceAfter=0)
    d.update(kw)
    return ParagraphStyle(name, **d)


def risk_col(s):
    return C_RED if s >= 60 else C_YELLOW if s >= 35 else C_GREEN


def risk_lbl(s):
    return "HIGH RISK" if s >= 60 else "MODERATE" if s >= 35 else "HEALTHY"


def divider(color=None, space=6):
    return HRFlowable(width="100%", thickness=1,
                      color=color or C_BORDER, spaceAfter=space, spaceBefore=3)


def logo_bar():
    t = Table([[Paragraph("BrandSense", ParagraphStyle(
        "_lb", fontName="Helvetica-Bold", fontSize=26,
        textColor=white, alignment=TA_CENTER, leading=30,
    ))]], colWidths=[W - 4*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), C_PURPLE),
        ("ROWPADDING", (0,0),(-1,-1), 16),
    ]))
    return t


def section_hdr(text):
    return Paragraph(text, ParagraphStyle(
        "_sh", fontName="Helvetica-Bold", fontSize=12,
        textColor=C_DARK, spaceBefore=16, spaceAfter=7,
        borderPad=4,
    ))


def metric_grid(pairs, cols=4):
    """Pairs: list of (label, value, hex_color)"""
    rows, row = [], []
    for label, value, color in pairs:
        inner = Table([
            [Paragraph(str(value), ParagraphStyle(
                "_mv", fontName="Helvetica-Bold", fontSize=14,
                textColor=HexColor(color), leading=16, alignment=TA_CENTER,
            ))],
            [Paragraph(str(label), ParagraphStyle(
                "_mk", fontSize=7.5, textColor=C_SUBTEXT,
                leading=10, alignment=TA_CENTER,
            ))],
        ], colWidths=["100%"])
        inner.setStyle(TableStyle([
            ("ALIGN",      (0,0),(-1,-1), "CENTER"),
            ("ROWPADDING", (0,0),(-1,-1), 2),
        ]))
        row.append(inner)
        if len(row) == cols:
            rows.append(row); row = []
    while len(row) < cols:
        row.append(Paragraph("", ParagraphStyle("_e")))
    if row:
        rows.append(row)
    cw = [(W - 4*cm) / cols] * cols
    t  = Table(rows, colWidths=cw)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,-1), C_LIGHT_BG),
        ("GRID",        (0,0),(-1,-1), 0.5, C_BORDER),
        ("ROWPADDING",  (0,0),(-1,-1), 10),
        ("ALIGN",       (0,0),(-1,-1), "CENTER"),
        ("VALIGN",      (0,0),(-1,-1), "MIDDLE"),
    ]))
    return t


def data_table(headers, rows_data, col_widths, alt=True):
    data = [headers] + rows_data
    t = Table(data, colWidths=col_widths, repeatRows=1)
    cmds = [
        ("BACKGROUND",    (0,0),(-1,0),  C_HDR_BG),
        ("TEXTCOLOR",     (0,0),(-1,0),  white),
        ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 8.5),
        ("LEADING",       (0,0),(-1,-1), 12),
        ("ROWPADDING",    (0,0),(-1,-1), 7),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("RIGHTPADDING",  (0,0),(-1,-1), 8),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("GRID",          (0,0),(-1,-1), 0.4, C_BORDER),
        ("BACKGROUND",    (0,1),(-1,-1), white),
    ]
    if alt:
        for i in range(1, len(rows_data)+1, 2):
            cmds.append(("BACKGROUND", (0,i),(-1,i), C_ROW_ALT))
    t.setStyle(TableStyle(cmds))
    return t


def review_card(row, number, mode="negative"):
    rv = str(row.get("clean_body",""))[:260]
    if len(str(row.get("clean_body",""))) > 260:
        rv += "..."
    title = str(row.get("review_title","No Title"))[:65]
    acc   = f"AI: {row.get('confidence',0):.0f}%"
    meta  = (f"{'★'*int(row.get('rating',0))}  |  "
             f"{'Verified' if row.get('verified',False) else 'Unverified'}  |  "
             f"{acc}  |  {str(row.get('date',''))[:10]}")
    if mode == "negative":
        meta += f"  |  {row.get('negative_category','')}"
        top_color  = C_RED
        bg_color   = C_PALE_RED
    else:
        top_color  = C_GREEN
        bg_color   = C_PALE_GRN

    hdr_txt = Paragraph(
        f"<b>#{number}</b>  {title}",
        ParagraphStyle("_rh", fontSize=9, fontName="Helvetica-Bold",
                       textColor=C_DARK, leading=12)
    )
    conf_txt = Paragraph(
        acc,
        ParagraphStyle("_rc", fontSize=9, fontName="Helvetica-Bold",
                       textColor=top_color, alignment=TA_RIGHT, leading=12)
    )
    meta_txt = Paragraph(
        meta,
        ParagraphStyle("_rm", fontSize=7.5, textColor=C_SUBTEXT, leading=11)
    )
    body_txt = Paragraph(
        f'"{rv}"',
        ParagraphStyle("_rb", fontSize=8.5, textColor=C_TEXT,
                       leading=13, fontName="Helvetica-Oblique")
    )
    card = Table([
        [hdr_txt,   conf_txt],
        [meta_txt,  Paragraph("", ParagraphStyle("_e"))],
        [body_txt,  Paragraph("", ParagraphStyle("_e"))],
    ], colWidths=["72%","28%"])
    card.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), bg_color),
        ("LINEABOVE",    (0,0),(-1,0), 2.5, top_color),
        ("LINEBELOW",    (0,1),(-1,1), 0.4, C_BORDER),
        ("ROWPADDING",   (0,0),(-1,-1), 6),
        ("LEFTPADDING",  (0,0),(-1,-1), 10),
        ("RIGHTPADDING", (0,0),(-1,-1), 10),
        ("VALIGN",       (0,0),(-1,-1), "TOP"),
        ("SPAN",         (0,1),(1,1)),
        ("SPAN",         (0,2),(1,2)),
    ]))
    return card


def footer_block():
    return [
        Spacer(1, 0.8*cm),
        divider(C_PURPLE, 4),
        Paragraph(
            f"BrandSense Intelligence Platform  |  "
            f"Model: cardiffnlp/twitter-roberta-base-sentiment  |  "
            f"Amazon Mobile Reviews Dataset  |  "
            f"{datetime.now().strftime('%d %B %Y')}",
            ParagraphStyle("_ft", fontSize=7, textColor=C_GRAY,
                           alignment=TA_CENTER, leading=11)
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# COMPANY REPORT
# ══════════════════════════════════════════════════════════════════════════════
def generate_company_report(brand, df_brand, risk_score,
                             top_neg_cat, pos_pct, neg_pct, avg_rating):
    if not REPORTLAB_AVAILABLE:
        raise ImportError("pip install reportlab")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             rightMargin=2*cm, leftMargin=2*cm,
                             topMargin=2*cm, bottomMargin=2*cm)
    E = []

    # Cover
    E += [
        Spacer(1, 0.8*cm),
        logo_bar(),
        Spacer(1, 0.4*cm),
        Paragraph("Company Intelligence Report", ParagraphStyle(
            "_ct", fontSize=11, textColor=C_SUBTEXT,
            alignment=TA_CENTER, spaceAfter=4)),
        Paragraph("Brand Reputation & Sentiment Analysis",
                   ParagraphStyle("_cs", fontSize=9, textColor=C_GRAY,
                                  alignment=TA_CENTER, spaceAfter=16)),
        divider(C_PURPLE, 10),
        Paragraph(brand, ParagraphStyle("_cb", fontName="Helvetica-Bold",
                   fontSize=20, textColor=C_PURPLE, alignment=TA_CENTER)),
        Spacer(1, 0.2*cm),
        Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}  |  "
                   f"Model: cardiffnlp/twitter-roberta-base-sentiment",
                   ParagraphStyle("_cg", fontSize=8, textColor=C_GRAY,
                                  alignment=TA_CENTER)),
        Spacer(1, 0.5*cm),
    ]

    # Risk banner
    rc = risk_col(risk_score)
    rl = risk_lbl(risk_score)
    rt = Table([[
        Paragraph("REPUTATION RISK SCORE",
                   ParagraphStyle("_rk", fontSize=8, textColor=C_SUBTEXT)),
        Paragraph(f"{risk_score:.0f} / 100",
                   ParagraphStyle("_rv", fontName="Helvetica-Bold", fontSize=22,
                                  textColor=rc, alignment=TA_CENTER)),
        Paragraph(f"Status: {rl}",
                   ParagraphStyle("_rl", fontName="Helvetica-Bold", fontSize=10,
                                  textColor=rc, alignment=TA_RIGHT)),
    ]], colWidths=["33%","34%","33%"])
    rt.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), C_LIGHT_BG),
        ("LINEABOVE",  (0,0),(-1,0), 4, rc),
        ("ROWPADDING", (0,0),(-1,-1), 14),
        ("VALIGN",     (0,0),(-1,-1), "MIDDLE"),
        ("GRID",       (0,0),(-1,-1), 0.5, C_BORDER),
    ]))
    E += [rt, Spacer(1, 0.6*cm)]

    # Key metrics
    total_r  = len(df_brand)
    pos_c    = int((df_brand["sentiment_label"]=="Positive").sum()) if "sentiment_label" in df_brand.columns else 0
    neg_c    = int((df_brand["sentiment_label"]=="Negative").sum()) if "sentiment_label" in df_brand.columns else 0
    neu_c    = int((df_brand["sentiment_label"]=="Neutral").sum())  if "sentiment_label" in df_brand.columns else 0
    avg_conf = df_brand["confidence"].mean() if "confidence" in df_brand.columns else 0
    ver_pct  = df_brand["verified"].mean()*100 if "verified" in df_brand.columns else 0

    E.append(section_hdr("Key Performance Metrics"))
    E.append(metric_grid([
        ("Total Reviews",   f"{total_r:,}",        "#5B5FCF"),
        ("Avg Star Rating", f"{avg_rating:.2f}/5",  "#ca8a04"),
        ("Positive Rate",   f"{pos_pct:.1f}%",      "#16a34a"),
        ("Negative Rate",   f"{neg_pct:.1f}%",      "#dc2626"),
        ("AI Confidence",   f"{avg_conf:.1f}%",     "#7c3aed"),
        ("Verified Buyers", f"{ver_pct:.1f}%",      "#0369a1"),
        ("Positive Count",  f"{pos_c:,}",           "#16a34a"),
        ("Negative Count",  f"{neg_c:,}",           "#dc2626"),
    ], cols=4))
    E.append(Spacer(1, 0.5*cm))

    # Sentiment table
    E.append(section_hdr("Sentiment Breakdown"))
    sent_rows = [
        ["Positive", f"{pos_c:,}", f"{pos_pct:.1f}%", "Customers satisfied with product"],
        ["Neutral",  f"{neu_c:,}", f"{100-pos_pct-neg_pct:.1f}%", "Mixed or factual feedback"],
        ["Negative", f"{neg_c:,}", f"{neg_pct:.1f}%", "Customers dissatisfied"],
    ]
    cw = [3*cm, 2.5*cm, 2.5*cm, None]
    cw[-1] = W - 4*cm - sum(cw[:-1])
    E.append(data_table(["Sentiment","Count","Percentage","Description"], sent_rows, cw))
    E.append(Spacer(1, 0.5*cm))

    # Negative categories
    if "negative_category" in df_brand.columns:
        E.append(section_hdr("Negative Review Categories"))
        neg_df     = df_brand[df_brand["sentiment_label"]=="Negative"] if "sentiment_label" in df_brand.columns else df_brand
        cat_counts = neg_df["negative_category"].value_counts()
        total_neg  = len(neg_df)
        sev_map = {
            "Defective Piece":        "Critical",
            "Poor Customer Service":  "High",
            "Price Dissatisfaction":  "Medium",
            "Delivery / Packaging":   "Medium",
            "Camera Issues":          "Medium",
            "Battery Issues":         "High",
            "Performance Issues":     "Medium",
            "General Dissatisfaction":"Low",
        }
        cat_rows = []
        for cat, count in cat_counts.items():
            cat_rows.append([
                cat, str(count),
                f"{count/max(total_neg,1)*100:.1f}%",
                sev_map.get(cat, "Low"),
            ])
        cw2 = [6*cm, 2*cm, 2.8*cm, None]
        cw2[-1] = W - 4*cm - sum(cw2[:-1])
        E.append(data_table(["Category","Count","% of Negatives","Severity"], cat_rows, cw2))
        E.append(Spacer(1, 0.5*cm))

    # Return analysis
    if "return_mentioned" in df_brand.columns:
        E.append(section_hdr("Return Signal Analysis"))
        E.append(Paragraph(
            "Return signals detected via NLP keyword analysis on review text — "
            "phrases like 'returned it', 'asked for refund', 'got a replacement'. "
            "Industry-standard approach when transaction data is unavailable.",
            ParagraphStyle("_rb2", fontSize=9, textColor=C_SUBTEXT, leading=14, spaceAfter=8)
        ))
        df_ret   = df_brand[df_brand["return_mentioned"]==True]
        ret_tot  = len(df_ret)
        ret_rate = ret_tot / max(len(df_brand),1) * 100
        comp_r   = int((df_ret["return_type"]=="Completed Return").sum())
        exch_r   = int((df_ret["return_type"]=="Exchange / Replacement").sum())
        att_r    = int((df_ret["return_type"]=="Return Attempted").sum())
        fail_r   = int((df_ret["return_type"]=="Return Failed").sum())

        E.append(metric_grid([
            ("Return Mentions",   f"{ret_tot:,}",   "#dc2626"),
            ("Est. Return Rate",  f"{ret_rate:.1f}%","#ea580c"),
            ("Confirmed Returns", f"{comp_r:,}",    "#7c3aed"),
            ("Exchanges",         f"{exch_r:,}",    "#0369a1"),
        ], cols=4))
        E.append(Spacer(1, 0.3*cm))

        if ret_tot > 0:
            ret_rows = []
            for rtype, count in df_ret["return_type"].value_counts().items():
                ret_rows.append([rtype, str(count), f"{count/ret_tot*100:.1f}%"])
            cw3 = [7*cm, 2.5*cm, None]
            cw3[-1] = W - 4*cm - sum(cw3[:-1])
            E.append(data_table(["Return Type","Count","% of Returns"], ret_rows, cw3))
            E.append(Spacer(1, 0.4*cm))

            # Sample return reviews
            E.append(Paragraph("Sample Return-Related Reviews", ParagraphStyle(
                "_srr", fontName="Helvetica-Bold", fontSize=10,
                textColor=C_DARK, spaceBefore=10, spaceAfter=6)))
            for _, row in df_ret.sort_values("confidence",ascending=False).head(3).iterrows():
                rv = str(row.get("clean_body",""))[:200]
                if len(str(row.get("clean_body",""))) > 200:
                    rv += "..."
                E.append(KeepTogether([
                    Paragraph(
                        f"<b>{row.get('return_type','')}</b>  |  "
                        f"{'★'*int(row.get('rating',0))}  |  "
                        f"{'Verified' if row.get('verified',False) else 'Unverified'}  |  "
                        f"AI: {row.get('confidence',0):.0f}%",
                        ParagraphStyle("_srrm", fontSize=8, textColor=C_SUBTEXT,
                                       leading=11, spaceBefore=4)
                    ),
                    Paragraph(f'"{rv}"',
                               ParagraphStyle("_srrb", fontSize=8.5, textColor=C_TEXT,
                                              leading=13, fontName="Helvetica-Oblique",
                                              leftIndent=10, spaceAfter=8)),
                ]))

    # Page break
    E.append(PageBreak())

    # High impact reviews
    E.append(section_hdr("Highest Business Impact Negative Reviews"))
    E.append(Paragraph(
        "Scored using RoBERTa confidence, helpful votes, verified buyer status, "
        "and complaint category severity. Higher = greater brand damage potential.",
        ParagraphStyle("_hip", fontSize=9, textColor=C_SUBTEXT, leading=14, spaceAfter=10)
    ))
    if "impact_score" in df_brand.columns:
        top_neg = df_brand[df_brand["sentiment_label"]=="Negative"].nlargest(8,"impact_score")
        for i, (_, row) in enumerate(top_neg.iterrows(), 1):
            E.append(review_card(row, i, "negative"))
            E.append(Spacer(1, 0.2*cm))

    E += footer_block()
    doc.build(E)
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOMER BUYING GUIDE
# ══════════════════════════════════════════════════════════════════════════════
def generate_customer_report(brand, model_name, recommendation, df_model):
    if not REPORTLAB_AVAILABLE:
        raise ImportError("pip install reportlab")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             rightMargin=2*cm, leftMargin=2*cm,
                             topMargin=2*cm, bottomMargin=2*cm)
    E = []

    verdict = recommendation.get("verdict","N/A")
    score   = recommendation.get("score", 0)
    vcolor  = HexColor(recommendation.get("color","#5B5FCF"))
    pos_p   = recommendation.get("pos_pct",0)
    neg_p   = recommendation.get("neg_pct",0)
    r       = recommendation.get("avg_rating",0)

    # Cover
    E += [
        Spacer(1, 0.8*cm),
        logo_bar(),
        Spacer(1, 0.4*cm),
        Paragraph("Customer Buying Guide", ParagraphStyle(
            "_ct2", fontSize=11, textColor=C_SUBTEXT,
            alignment=TA_CENTER, spaceAfter=4)),
        Paragraph("AI-Powered Purchase Recommendation",
                   ParagraphStyle("_cs2", fontSize=9, textColor=C_GRAY,
                                  alignment=TA_CENTER, spaceAfter=16)),
        divider(C_PURPLE, 10),
        Paragraph(f"{brand}  —  {model_name[:55]}",
                   ParagraphStyle("_cb2", fontName="Helvetica-Bold", fontSize=16,
                                  textColor=C_PURPLE, alignment=TA_CENTER)),
        Spacer(1, 0.2*cm),
        Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}",
                   ParagraphStyle("_cg2", fontSize=8, textColor=C_GRAY,
                                  alignment=TA_CENTER)),
        Spacer(1, 0.5*cm),
    ]

    # Verdict banner
    vt = Table([[
        Paragraph("PURCHASE VERDICT",
                   ParagraphStyle("_vl", fontSize=8, textColor=C_SUBTEXT)),
        Paragraph(verdict,
                   ParagraphStyle("_vv", fontName="Helvetica-Bold", fontSize=15,
                                  textColor=vcolor, alignment=TA_CENTER)),
        Paragraph(f"Score: {score:.0f}/100",
                   ParagraphStyle("_vs", fontSize=9, textColor=C_SUBTEXT,
                                  alignment=TA_RIGHT)),
    ]], colWidths=["25%","50%","25%"])
    vt.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), C_LIGHT_BG),
        ("LINEABOVE",  (0,0),(-1,0), 4, vcolor),
        ("ROWPADDING", (0,0),(-1,-1), 16),
        ("VALIGN",     (0,0),(-1,-1), "MIDDLE"),
        ("GRID",       (0,0),(-1,-1), 0.5, C_BORDER),
    ]))
    E += [vt, Spacer(1, 0.5*cm)]

    # Summary metrics
    E.append(section_hdr("Product Sentiment Summary"))
    E.append(metric_grid([
        ("Reviews Analyzed",  f"{recommendation.get('total',0):,}",       "#5B5FCF"),
        ("Avg Star Rating",   f"{r:.2f}/5",                               "#ca8a04"),
        ("Positive Rate",     f"{pos_p:.1f}%",                            "#16a34a"),
        ("Negative Rate",     f"{neg_p:.1f}%",                            "#dc2626"),
        ("Verified Positive", f"{recommendation.get('verified_pos',0):,}","#16a34a"),
        ("Verified Negative", f"{recommendation.get('verified_neg',0):,}","#dc2626"),
    ], cols=3))
    E.append(Spacer(1, 0.5*cm))

    # Pros and Cons
    E.append(section_hdr("Pros & Cons Summary"))
    pros = recommendation.get("pros",[])
    cons = recommendation.get("cons",[])

    pros_rows = [[Paragraph("WHAT CUSTOMERS LOVE", ParagraphStyle(
        "_ph", fontName="Helvetica-Bold", fontSize=9,
        textColor=C_GREEN, leading=12))]]
    for p in pros:
        pros_rows.append([Paragraph(
            f"  +  {p.title()}",
            ParagraphStyle("_pi", fontSize=9, textColor=C_TEXT, leading=13,
                           leftIndent=6))])

    cons_rows = [[Paragraph("COMMON COMPLAINTS", ParagraphStyle(
        "_ch", fontName="Helvetica-Bold", fontSize=9,
        textColor=C_RED, leading=12))]]
    for c in cons:
        cons_rows.append([Paragraph(
            f"  -  {c}",
            ParagraphStyle("_ci", fontSize=9, textColor=C_TEXT, leading=13,
                           leftIndent=6))])

    p_t = Table(pros_rows, colWidths=["100%"])
    p_t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(0,0), C_PALE_GRN),
        ("BACKGROUND",  (0,1),(-1,-1), white),
        ("ROWPADDING",  (0,0),(-1,-1), 7),
        ("LEFTPADDING", (0,0),(-1,-1), 10),
        ("GRID",        (0,0),(-1,-1), 0.4, HexColor("#bbf7d0")),
        ("LINEABOVE",   (0,0),(-1,0), 3, C_GREEN),
    ]))
    c_t = Table(cons_rows, colWidths=["100%"])
    c_t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(0,0), C_PALE_RED),
        ("BACKGROUND",  (0,1),(-1,-1), white),
        ("ROWPADDING",  (0,0),(-1,-1), 7),
        ("LEFTPADDING", (0,0),(-1,-1), 10),
        ("GRID",        (0,0),(-1,-1), 0.4, HexColor("#fecaca")),
        ("LINEABOVE",   (0,0),(-1,0), 3, C_RED),
    ]))
    pc = Table([[p_t, Spacer(0.3*cm, 1), c_t]], colWidths=["48%","4%","48%"])
    pc.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP")]))
    E += [pc, Spacer(1, 0.5*cm)]

    # Return signals for this model
    if "return_mentioned" in df_model.columns:
        df_ret = df_model[df_model["return_mentioned"]==True]
        if len(df_ret) > 0:
            ret_rate = len(df_ret)/max(len(df_model),1)*100
            E.append(section_hdr("Return Signal Analysis"))
            E.append(Paragraph(
                f"Of {len(df_model):,} reviews analyzed, {len(df_ret):,} "
                f"({ret_rate:.1f}%) mention product returns or exchanges.",
                ParagraphStyle("_rn", fontSize=9, textColor=C_SUBTEXT,
                               leading=14, spaceAfter=8)))
            ret_rows = []
            for rtype, count in df_ret["return_type"].value_counts().items():
                ret_rows.append([rtype, str(count), f"{count/len(df_ret)*100:.1f}%"])
            cw_r = [7*cm, 2.5*cm, None]
            cw_r[-1] = W - 4*cm - sum(cw_r[:-1])
            E.append(data_table(["Return Type","Count","% of Returns"], ret_rows, cw_r))
            E.append(Spacer(1, 0.4*cm))

    # Page break
    E.append(PageBreak())

    # Positive reviews
    E.append(section_hdr("Top Positive Reviews"))
    E.append(Paragraph(
        "Reviews where RoBERTa detected positive sentiment from review text.",
        ParagraphStyle("_pr", fontSize=9, textColor=C_SUBTEXT, leading=14, spaceAfter=8)
    ))
    pos_revs = df_model[df_model["sentiment_label"]=="Positive"].nlargest(5,"confidence")
    if len(pos_revs)==0:
        E.append(Paragraph("No positive reviews found for this model.",
                            ParagraphStyle("_npr", fontSize=9, textColor=C_GRAY)))
    else:
        for i,(_, row) in enumerate(pos_revs.iterrows(), 1):
            E.append(review_card(row, i, "positive"))
            E.append(Spacer(1, 0.2*cm))

    # Negative reviews
    E += [Spacer(1, 0.3*cm)]
    E.append(section_hdr("Top Negative Reviews"))
    E.append(Paragraph(
        "Reviews where RoBERTa detected negative sentiment. "
        "Read these carefully before purchasing.",
        ParagraphStyle("_nr2", fontSize=9, textColor=C_SUBTEXT, leading=14, spaceAfter=8)
    ))
    neg_revs = df_model[df_model["sentiment_label"]=="Negative"].nlargest(5,"confidence")
    if len(neg_revs)==0:
        E.append(Paragraph(
            "No significant negative reviews — this is a good sign!",
            ParagraphStyle("_nnr", fontSize=9, textColor=C_GREEN,
                           fontName="Helvetica-Bold")))
    else:
        for i,(_, row) in enumerate(neg_revs.iterrows(), 1):
            E.append(review_card(row, i, "negative"))
            E.append(Spacer(1, 0.2*cm))

    # Final advice
    E += [Spacer(1, 0.4*cm)]
    E.append(section_hdr("Final Recommendation"))
    if pos_p >= 65 and r >= 4.0:
        advice = (f"With {pos_p:.0f}% positive reviews and an average rating of "
                  f"{r:.2f}/5, this product has a strong track record among real customers. "
                  f"The majority of verified buyers report satisfaction. This is a safe purchase.")
    elif neg_p >= 40:
        advice = (f"With {neg_p:.0f}% negative reviews, this product shows significant quality concerns. "
                  f"We recommend researching alternatives before purchasing. If you proceed, "
                  f"ensure the seller offers a full return policy.")
    else:
        advice = (f"This product has mixed feedback ({pos_p:.0f}% positive, {neg_p:.0f}% negative). "
                  f"Performance may vary. Review the specific complaints listed above before deciding. "
                  f"It may still suit your needs if the reported issues are not relevant to your use case.")

    adv_t = Table([[Paragraph(advice, ParagraphStyle(
        "_adv", fontSize=9.5, textColor=C_TEXT, leading=16, alignment=TA_JUSTIFY
    ))]], colWidths=["100%"])
    adv_t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,-1), C_LIGHT_BG),
        ("ROWPADDING",  (0,0),(-1,-1), 14),
        ("LEFTPADDING", (0,0),(-1,-1), 14),
        ("GRID",        (0,0),(-1,-1), 0.5, C_BORDER),
        ("LINEABOVE",   (0,0),(-1,0), 3, vcolor),
    ]))
    E.append(adv_t)
    E += footer_block()

    doc.build(E)
    buf.seek(0)
    return buf.read()
