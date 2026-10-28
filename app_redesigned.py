import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="FinSight | Financial Intelligence", page_icon="💠", layout="wide", initial_sidebar_state="collapsed")

REQUIRED_COLUMNS = ["expense_id", "employee_id", "expense_date", "merchant", "category", "amount", "currency", "description"]
DEFAULT_LIMITS = {"Travel": 15000, "Meals": 2500, "Accommodation": 12000, "Office Supplies": 5000, "Software": 20000, "Transport": 4000, "Other": 5000}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
.stApp{background:#F4F7FC;color:#172033;font-family:'DM Sans',sans-serif}
.block-container{max-width:1440px;padding-top:1.7rem;padding-bottom:3rem}
h1,h2,h3{font-family:'Manrope',sans-serif;letter-spacing:-.03em}
.hero{background:radial-gradient(circle at 82% 15%,#344b83 0,transparent 28%),radial-gradient(circle at 70% 110%,#174b52 0,transparent 35%),linear-gradient(125deg,#10172A,#1B2B4A);border-radius:28px;padding:3.2rem;color:white;min-height:340px;margin-bottom:1.5rem;box-shadow:0 18px 45px #18243e20;position:relative;overflow:hidden}
.hero h1{color:white;font-size:clamp(2.6rem,5vw,4.2rem);line-height:1.06;margin:.3rem 0 1rem}
.hero p{color:#C7D1E5;font-size:1.05rem;line-height:1.8;max-width:680px}
.eyebrow{color:#91B3FF;font-size:.76rem;font-weight:800;letter-spacing:.18em}
.chip{display:inline-block;border:1px solid #ffffff35;background:#ffffff10;border-radius:999px;padding:.45rem .75rem;margin:.4rem .3rem 0 0;color:#E4EAFA;font-size:.8rem}
.kicker{color:#71809A;text-transform:uppercase;font-size:.74rem;font-weight:800;letter-spacing:.15em;margin-bottom:.35rem}
.title{font-size:1.65rem;font-weight:800;color:#172033;margin-bottom:.25rem}
.sub{color:#748198;margin-bottom:1.1rem}
.card{background:white;border:1px solid #E6EBF4;border-radius:19px;padding:1.2rem;min-height:165px;box-shadow:0 5px 20px #1c2d4e09}
.icon{display:inline-flex;width:43px;height:43px;border-radius:14px;align-items:center;justify-content:center;font-size:1.25rem;margin-bottom:.8rem}
.card h3{font-size:1rem;margin:0 0 .4rem;color:#1B2740}.card p{color:#71809A;font-size:.88rem;line-height:1.6;margin:0}
.panel{background:white;border:1px solid #E6EBF4;border-radius:18px;padding:1.1rem 1.2rem;margin:.5rem 0 1rem}
.stat{background:white;border:1px solid #E6EBF4;border-radius:16px;padding:1rem 1.1rem;min-height:110px}
.stat-label{color:#6F7D94;font-size:.78rem;font-weight:700}.stat-value{font-family:'Manrope',sans-serif;font-size:1.65rem;font-weight:800;color:#172033;margin-top:.35rem}.stat-note{color:#8290A5;font-size:.75rem;margin-top:.15rem}
.blue{border-top:4px solid #3978F6}.green{border-top:4px solid #11A88A}.red{border-top:4px solid #E55367}.purple{border-top:4px solid #845EF7}
div.stButton>button{border-radius:12px;border:0;padding:.72rem 1rem;font-weight:700}
div.stButton>button[kind="primary"]{background:linear-gradient(100deg,#3978F6,#6B5CF6);color:white;box-shadow:0 8px 18px #3978f638}
div[data-testid="stMetric"]{background:white;border:1px solid #E6EBF4;border-radius:15px;padding:1rem}
.footer{text-align:center;color:#8793A7;font-size:.78rem;padding:1.2rem .3rem}
/* Improve readability of dashboard text */
.stApp {
    color: #172033 !important;
}

/* Main dashboard labels */
.stApp label,
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] p {
    color: #344054 !important;
    opacity: 1 !important;
}

/* Keep Audit settings readable in the dark sidebar */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #F1F5F9 !important;
    opacity: 1 !important;
}

/* Metric headings */
.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricLabel"] p {
    color: #64748B !important;
    opacity: 1 !important;
}

/* Metric values */
.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricValue"] p {
    color: #172033 !important;
    opacity: 1 !important;
}

/* Dashboard paragraphs and descriptions */
.stApp .stMarkdown p,
.stApp .stMarkdown li {
    color: #475569;
}

/* Tables */
.stApp [data-testid="stDataFrame"],
.stApp [data-testid="stTable"] {
    color: #172033 !important;
}

/* Chart labels and axis text */
.stApp .stPlotlyChart svg text {
    fill: #475569 !important;
}

/* Sidebar controls */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
    color: #F8FAFC !important;
}
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "home"

def navigate(page):
    st.session_state.page = page
    st.rerun()

@st.cache_data
def load_sample():
    return pd.read_csv(Path(__file__).parent / "sample_expenses.csv")

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    aliases = {"expenseid":"expense_id","employeeid":"employee_id","date":"expense_date","transaction_date":"expense_date","vendor":"merchant","expense_amount":"amount","expense_category":"category"}
    return df.rename(columns={c: aliases.get(c,c) for c in df.columns})

def analyse_expenses(raw, policy_limits, anomaly_fraction):
    df = normalize_columns(raw)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    df = df[REQUIRED_COLUMNS].copy()
    df["expense_date"] = pd.to_datetime(df["expense_date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    for col in ["employee_id","merchant","category"]:
        df[col] = df[col].astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str)
    invalid = df["expense_date"].isna() | df["amount"].isna() | (df["amount"] < 0)
    df["review_reasons"] = ""
    df["risk_score"] = 0
    df["duplicate_flag"] = False
    df["policy_flag"] = False
    df["anomaly_flag"] = False
    df["data_quality_flag"] = invalid
    df["duplicate_flag"] = df.duplicated(subset=["employee_id","expense_date","merchant","amount"], keep=False) & ~invalid
    df["policy_limit"] = df["category"].map(policy_limits).fillna(policy_limits.get("Other",5000))
    df["policy_flag"] = (df["amount"] > df["policy_limit"]) & ~invalid
    valid_idx = df.index[~invalid]
    if len(valid_idx) >= 10 and df.loc[valid_idx,"amount"].nunique() >= 3:
        model_data = df.loc[valid_idx,["amount"]].copy()
        model_data["log_amount"] = np.log1p(model_data["amount"])
        model = IsolationForest(n_estimators=100, contamination=anomaly_fraction, random_state=42)
        df.loc[valid_idx,"anomaly_flag"] = model.fit_predict(model_data) == -1
    for idx,row in df.iterrows():
        reasons, score = [], 0
        if row["data_quality_flag"]:
            reasons.append("Invalid or missing date/amount"); score += 25
        if row["duplicate_flag"]:
            reasons.append("Possible duplicate claim"); score += 40
        if row["policy_flag"]:
            reasons.append(f"Amount exceeds {row['category']} policy limit"); score += 30
        if row["anomaly_flag"]:
            reasons.append("Unusual amount pattern (Isolation Forest)"); score += 20
        df.at[idx,"review_reasons"] = "; ".join(reasons) if reasons else "No rule triggered"
        df.at[idx,"risk_score"] = min(score,100)
    df["review_status"] = np.where(df["risk_score"] > 0,"Needs review","No flag")
    return df.sort_values(["risk_score","amount"],ascending=[False,False]).reset_index(drop=True)

# PAGE 1 — Landing screen
if st.session_state.page == "home":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">AI-POWERED FINANCIAL OPERATIONS</div>
      <h1>See the story<br>behind every expense.</h1>
      <p>One intelligent workspace to monitor spend, surface unusual claims, and help audit teams focus on what needs a closer look.</p>
      <span class="chip">✦ Expense intelligence</span><span class="chip">◈ Policy monitoring</span><span class="chip">↗ Anomaly screening</span>
    </div>""", unsafe_allow_html=True)
    intro, action = st.columns([1.4, .8], vertical_alignment="center")
    with intro:
        st.markdown('<div class="kicker">YOUR AUDIT WORKSPACE</div><div class="title">From raw expenses to clearer decisions.</div><div class="sub">Review spending trends, apply configurable policy limits, and prioritise claims for human review—all in one place.</div>', unsafe_allow_html=True)
    with action:
        st.markdown('<div class="panel"><span style="color:#3978F6;font-size:.76rem;font-weight:800;letter-spacing:.1em">READY TO EXPLORE</span><h3 style="margin:.6rem 0;color:#172033">Launch the workspace</h3><p style="color:#71809A">Open the interactive audit dashboard and explore sample expense data.</p></div>', unsafe_allow_html=True)
        if st.button("Launch Audit Dashboard  →", type="primary", use_container_width=True):
            navigate("analysis")
    st.markdown('<div class="kicker">PLATFORM CAPABILITIES</div><div class="title">A smarter way to review expenses</div><div class="sub">Built to support audit judgement—not replace it.</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    cards = [("◈","Spend visibility","Explore expense totals and category-level patterns.","#EAF1FF"),("✓","Policy monitoring","Compare claims with configurable category limits.","#DDF8F0"),("◎","Anomaly screening","Use Isolation Forest to surface unusual amounts.","#F0EAFE"),("↗","Review prioritisation","Combine rule-based flags into a transparent score.","#FFF1D8")]
    for col,(icon,title,desc,bg) in zip(cols,cards):
        with col:
            st.markdown(f'<div class="card"><div class="icon" style="background:{bg}">{icon}</div><h3>{title}</h3><p>{desc}</p></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    tech = st.columns(4)
    for col,label in zip(tech,["PYTHON","PANDAS","SCIKIT-LEARN","STREAMLIT"]):
        with col:
            st.markdown(f'<div class="panel" style="text-align:center;color:#53627B;font-size:.8rem;font-weight:800;letter-spacing:.08em">{label}</div>',unsafe_allow_html=True)
    st.markdown('<div class="footer">FINSIGHT • FINANCIAL INTELLIGENCE DEMO<br>Flags indicate claims for review, not proof of fraud.</div>',unsafe_allow_html=True)

# PAGE 2 — Existing analysis workspace
else:
    head, back = st.columns([3,1])
    with head:
        st.markdown('<div class="kicker">FINANCIAL INTELLIGENCE / AUDIT WORKSPACE</div><div class="title">Expense Audit Dashboard</div><div class="sub">Monitor spending, inspect risk indicators, and export claims that need follow-up.</div>',unsafe_allow_html=True)
    with back:
        if st.button("← Back to Home",use_container_width=True):
            navigate("home")
    with st.sidebar:
        st.markdown("## ⚙️ Audit settings")
        limits = {}
        for category,default in DEFAULT_LIMITS.items():
            limits[category] = st.number_input(f"{category} limit (₹)",min_value=0,value=int(default),step=500,key=f"limit_{category}")
        contamination = st.slider("Expected anomaly fraction (ML)",min_value=0.01,max_value=0.20,value=0.05,step=0.01,help="Screening threshold, not a fraud probability.")
    up, helpcol = st.columns([1.2,1])
    with up:
        uploaded = st.file_uploader("Upload an expense CSV",type=["csv"])
    with helpcol:
        with st.expander("Required CSV columns"):
            st.write("Your CSV must contain:")
            st.code(", ".join(REQUIRED_COLUMNS))
            st.caption("Dates must be parseable and amount numeric. Use synthetic or authorised data only.")
    if uploaded:
        try:
            raw_df = pd.read_csv(uploaded); source_name = uploaded.name
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}"); st.stop()
    else:
        try:
            raw_df = load_sample(); source_name = "sample_expenses.csv"
            st.info("Showing sample data. Upload your own CSV to analyse it.")
        except FileNotFoundError:
            st.error("sample_expenses.csv was not found. Keep it beside app.py, or upload your own CSV."); st.stop()
    st.caption(f"Data source: **{source_name}**")
    try:
        results = analyse_expenses(raw_df,limits,contamination)
    except ValueError as exc:
        st.error(str(exc)); st.stop()
    flagged = results[results["risk_score"] > 0]
    total_spend = results.loc[results["amount"].notna() & (results["amount"] >= 0),"amount"].sum()
    flagged_spend = flagged.loc[flagged["amount"].notna(),"amount"].sum()
    flag_rate = len(flagged)/len(results)*100 if len(results) else 0
    anomaly_count = int(results["anomaly_flag"].sum())
    metrics = st.columns(4)
    metric_data = [("TOTAL EXPENSE CLAIMS",f"{len(results):,}","All uploaded claims","blue"),("TOTAL SPEND",f"₹{total_spend:,.0f}","Valid non-negative amounts","green"),("NEEDS REVIEW",f"{len(flagged):,}",f"{flag_rate:.1f}% of claims","red"),("POTENTIAL ANOMALIES",f"{anomaly_count:,}","Isolation Forest flags","purple")]
    for col,(label,value,note,cls) in zip(metrics,metric_data):
        with col:
            st.markdown(f'<div class="stat {cls}"><div class="stat-label">{label}</div><div class="stat-value">{value}</div><div class="stat-note">{note}</div></div>',unsafe_allow_html=True)
    extras = st.columns(3)
    extras[0].metric("Policy limit breaches",f"{int(results['policy_flag'].sum()):,}")
    extras[1].metric("Possible duplicate claims",f"{int(results['duplicate_flag'].sum()):,}")
    extras[2].metric("Flagged spend",f"₹{flagged_spend:,.0f}")
    left,right = st.columns(2)
    palette = {"Travel":"#3978F6","Meals":"#F59E0B","Accommodation":"#845EF7","Office Supplies":"#11A88A","Software":"#E55367","Transport":"#23A6D5","Other":"#93A1B5"}
    with left:
        st.markdown('<div class="panel"><div class="kicker">SPENDING PATTERNS</div><div class="title" style="font-size:1.2rem">Spend by category</div>',unsafe_allow_html=True)
        spend = results.groupby("category",dropna=False)["amount"].sum().reset_index().sort_values("amount",ascending=False)
        fig = px.bar(spend,x="category",y="amount",color="category",color_discrete_map=palette,labels={"category":"Expense category","amount":"Amount (₹)"},text_auto=".2s")
        fig.update_layout(height=370,margin=dict(l=10,r=10,t=15,b=10),showlegend=False,plot_bgcolor="white",paper_bgcolor="white",font=dict(family="DM Sans",color="#53627B"),xaxis_title=None,yaxis_title="Amount (₹)")
        fig.update_traces(marker_line_width=0,textposition="outside")
        st.plotly_chart(fig,use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="kicker">AUDIT SIGNALS</div><div class="title" style="font-size:1.2rem">Why claims are flagged</div>',unsafe_allow_html=True)
        reasons = pd.Series([r for value in flagged["review_reasons"] for r in value.split("; ") if r]).value_counts().rename_axis("reason").reset_index(name="count")
        if reasons.empty:
            st.success("No claims were flagged with the current settings.")
        else:
            fig2 = px.bar(reasons,x="count",y="reason",orientation="h",color="reason",color_discrete_map={"Invalid or missing date/amount":"#E55367","Possible duplicate claim":"#F59E0B","Unusual amount pattern (Isolation Forest)":"#845EF7","Amount exceeds Travel policy limit":"#3978F6"},labels={"count":"Number of claims","reason":"Review reason"},text="count")
            fig2.update_layout(height=370,margin=dict(l=10,r=10,t=15,b=10),showlegend=False,plot_bgcolor="white",paper_bgcolor="white",font=dict(family="DM Sans",color="#53627B"),xaxis_title="Number of claims",yaxis_title=None)
            fig2.update_traces(textposition="outside")
            st.plotly_chart(fig2,use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)
    st.markdown('<div class="kicker">INVESTIGATION QUEUE</div><div class="title">Transactions requiring review</div><div class="sub">Filter the queue to focus on claims relevant to your audit.</div>',unsafe_allow_html=True)
    f1,f2,f3 = st.columns(3)
    with f1:
        selected_category = st.selectbox("Filter by category",["All"]+sorted(flagged["category"].dropna().unique().tolist()),key="audit_category_filter")
    with f2:
        min_risk = st.slider("Minimum risk score",min_value=0,max_value=100,value=0,step=10,key="audit_risk_filter")
    with f3:
        reasons_list = ["All"]+sorted({r.strip() for val in flagged["review_reasons"].dropna() for r in val.split(";") if r.strip()})
        selected_reason = st.selectbox("Filter by review reason",reasons_list,key="audit_reason_filter")
    filtered = flagged.copy()
    if selected_category != "All": filtered = filtered[filtered["category"] == selected_category]
    filtered = filtered[filtered["risk_score"] >= min_risk]
    if selected_reason != "All": filtered = filtered[filtered["review_reasons"].str.contains(selected_reason,regex=False,na=False)]
    show_cols = ["expense_id","employee_id","expense_date","merchant","category","amount","policy_limit","risk_score","review_reasons"]
    st.caption(f"Showing {len(filtered)} of {len(flagged)} flagged transactions.")
    if filtered.empty:
        st.info("No flagged transactions match these filters.")
    else:
        shown = filtered[show_cols].copy()
        shown["expense_date"] = shown["expense_date"].dt.strftime("%Y-%m-%d")
        st.dataframe(shown,use_container_width=True,hide_index=True)
        st.download_button("⬇️ Download filtered audit results (CSV)",data=filtered.to_csv(index=False).encode("utf-8"),file_name="filtered_expenses_audit_report.csv",mime="text/csv",key="download_filtered_audit")
    with st.expander("View all analysed transactions"):
        all_shown = results.copy()
        all_shown["expense_date"] = all_shown["expense_date"].dt.strftime("%Y-%m-%d")
        st.dataframe(all_shown,use_container_width=True,hide_index=True)
    st.download_button("⬇️ Download all flagged transactions (CSV)",data=flagged.to_csv(index=False).encode("utf-8"),file_name="flagged_expenses_audit_report.csv",mime="text/csv")
    st.markdown('<div class="footer">Duplicate matching + configurable category limits + Isolation Forest screening.<br>Risk scores prioritise review; they are not calibrated fraud probabilities. Validate claims against receipts and policy.</div>',unsafe_allow_html=True)
