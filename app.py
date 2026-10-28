import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Expense Intelligence", page_icon="🔎", layout="wide")

REQUIRED_COLUMNS = [
    "expense_id", "employee_id", "expense_date", "merchant",
    "category", "amount", "currency", "description"
]

DEFAULT_LIMITS = {
    "Travel": 15000,
    "Meals": 2500,
    "Accommodation": 12000,
    "Office Supplies": 5000,
    "Software": 20000,
    "Transport": 4000,
    "Other": 5000,
}

st.title("🔎 AI-Powered Financial Audit & Expense Intelligence")
st.caption("A decision-support demo that screens expense claims for review. Flags are not proof of fraud.")

with st.sidebar:
    st.header("Audit settings")
    st.write("Adjust category limits for this demo.")
    limits = {}
    for category, default in DEFAULT_LIMITS.items():
        limits[category] = st.number_input(
            f"{category} limit (₹)", min_value=0, value=int(default), step=500, key=f"limit_{category}"
        )
    contamination = st.slider(
        "Expected anomaly fraction (ML)", min_value=0.01, max_value=0.20,
        value=0.05, step=0.01,
        help="Isolation Forest setting; it is a screening threshold, not a fraud probability."
    )

@st.cache_data
def load_sample():
    sample_path = Path(__file__).parent / "sample_expenses.csv"
    return pd.read_csv(sample_path)

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    aliases = {
        "expenseid": "expense_id", "employeeid": "employee_id",
        "date": "expense_date", "transaction_date": "expense_date",
        "vendor": "merchant", "expense_amount": "amount",
        "expense_category": "category"
    }
    df = df.rename(columns={c: aliases.get(c, c) for c in df.columns})
    return df

def analyse_expenses(raw, policy_limits, anomaly_fraction):
    df = normalize_columns(raw)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    df = df[REQUIRED_COLUMNS].copy()
    df["expense_date"] = pd.to_datetime(df["expense_date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["employee_id"] = df["employee_id"].astype(str).str.strip()
    df["merchant"] = df["merchant"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str)

    invalid = df["expense_date"].isna() | df["amount"].isna() | (df["amount"] < 0)
    df["review_reasons"] = ""
    df["risk_score"] = 0
    df["duplicate_flag"] = False
    df["policy_flag"] = False
    df["anomaly_flag"] = False
    df["data_quality_flag"] = invalid

    # Duplicate candidates: same employee, date, merchant and amount.
    duplicate_key = ["employee_id", "expense_date", "merchant", "amount"]
    df["duplicate_flag"] = df.duplicated(subset=duplicate_key, keep=False) & ~invalid

    # Policy limit checks by category.
    df["policy_limit"] = df["category"].map(policy_limits).fillna(policy_limits.get("Other", 5000))
    df["policy_flag"] = (df["amount"] > df["policy_limit"]) & ~invalid

    # Isolation Forest: analyse transaction amounts and their log-transformed values
    valid_idx = df.index[~invalid]

    if len(valid_idx) >= 10 and df.loc[valid_idx, "amount"].nunique() >= 3:
        model_data = df.loc[valid_idx, ["amount"]].copy()

        # Log transformation reduces the influence of very large amounts
        model_data["log_amount"] = np.log1p(model_data["amount"])

        model = IsolationForest(
            n_estimators=100,
            contamination=anomaly_fraction,
            random_state=42
        )

        predictions = model.fit_predict(model_data)
        df.loc[valid_idx, "anomaly_flag"] = predictions == -1

    for idx, row in df.iterrows():
        reasons = []
        score = 0

        if row["data_quality_flag"]:
            reasons.append("Invalid or missing date/amount")
            score += 25

        if row["duplicate_flag"]:
            reasons.append("Possible duplicate claim")
            score += 40

        if row["policy_flag"]:
            reasons.append(f"Amount exceeds {row['category']} policy limit")
            score += 30

        if row["anomaly_flag"]:
            reasons.append("Unusual amount pattern (Isolation Forest)")
            score += 20

        df.at[idx, "review_reasons"] = "; ".join(reasons) if reasons else "No rule triggered"
        df.at[idx, "risk_score"] = min(score, 100)

    df["review_status"] = np.where(df["risk_score"] > 0, "Needs review", "No flag")
    return df.sort_values(
        ["risk_score", "amount"],
        ascending=[False, False]
    ).reset_index(drop=True)
  

uploaded = st.file_uploader("Upload an expense CSV", type=["csv"])
if uploaded:
    try:
        raw_df = pd.read_csv(uploaded)
        source_name = uploaded.name
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
else:
    raw_df = load_sample()
    source_name = "sample_expenses.csv"
    st.info("Showing sample data. Upload your own CSV to analyse it.")

st.caption(f"Data source: **{source_name}**")
with st.expander("Required CSV columns and example"):
    st.write("Your CSV must contain these columns (case-insensitive):")
    st.code(", ".join(REQUIRED_COLUMNS))
    st.write("Dates should be parseable dates; amount should be numeric. Use synthetic or authorised data only.")

try:
    results = analyse_expenses(raw_df, limits, contamination)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

flagged = results[results["risk_score"] > 0]
total_spend = results.loc[results["amount"].notna() & (results["amount"] >= 0), "amount"].sum()
flagged_spend = flagged.loc[flagged["amount"].notna(), "amount"].sum()
flag_rate = (len(flagged) / len(results) * 100) if len(results) else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Expense claims", f"{len(results):,}")
m2.metric("Total spend", f"₹{total_spend:,.0f}")
m3.metric("Claims flagged", f"{len(flagged):,}", f"{flag_rate:.1f}% of claims")
m4.metric("Flagged spend", f"₹{flagged_spend:,.0f}")

left, right = st.columns(2)

with left:
    st.subheader("Spend by Category")

    spend_by_category = (
        results.groupby("category", dropna=False)["amount"]
        .sum()
        .reset_index()
        .sort_values("amount", ascending=False)
    )

    fig = px.bar(
        spend_by_category,
        x="category",
        y="amount",
        title="Total Spending by Category",
        labels={
            "category": "Expense Category",
            "amount": "Amount (₹)"
        },
        text_auto=".2s"
    )

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Expense Category",
        yaxis_title="Amount (₹)"
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Audit Review Reasons")

    reason_counts = pd.Series(
        [
            reason
            for value in flagged["review_reasons"]
            for reason in value.split("; ")
            if reason
        ]
    ).value_counts().rename_axis("reason").reset_index(name="count")

    if reason_counts.empty:
        st.success("No claims were flagged with the current settings.")
    else:
        fig = px.bar(
            reason_counts,
            x="count",
            y="reason",
            orientation="h",
            title="Reasons for Flagging",
            labels={
                "count": "Number of Claims",
                "reason": "Review Reason"
            },
            text="count"
        )

        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=60, b=20),
            yaxis={"categoryorder": "total ascending"}
        )

        st.plotly_chart(fig, use_container_width=True)

st.subheader("Transactions Requiring Review")

# Filters for the audit review table
filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    categories = ["All"] + sorted(
        flagged["category"].dropna().unique().tolist()
    )
    selected_category = st.selectbox(
        "Filter by category",
        categories,
        key="audit_category_filter"
    )

with filter_col2:
    min_risk = st.slider(
        "Minimum risk score",
        min_value=0,
        max_value=100,
        value=0,
        step=10,
        key="audit_risk_filter"
    )

with filter_col3:
    review_options = ["All"] + sorted({
        reason.strip()
        for value in flagged["review_reasons"].dropna()
        for reason in value.split(";")
        if reason.strip()
    })

    selected_reason = st.selectbox(
        "Filter by review reason",
        review_options,
        key="audit_reason_filter"
    )

# Apply selected filters
filtered_flagged = flagged.copy()

if selected_category != "All":
    filtered_flagged = filtered_flagged[
        filtered_flagged["category"] == selected_category
    ]

filtered_flagged = filtered_flagged[
    filtered_flagged["risk_score"] >= min_risk
]

if selected_reason != "All":
    filtered_flagged = filtered_flagged[
        filtered_flagged["review_reasons"].str.contains(
            selected_reason,
            regex=False,
            na=False
        )
    ]

show_cols = [
    "expense_id",
    "employee_id",
    "expense_date",
    "merchant",
    "category",
    "amount",
    "policy_limit",
    "risk_score",
    "review_reasons"
]

st.caption(
    f"Showing {len(filtered_flagged)} of {len(flagged)} flagged transactions."
)

if filtered_flagged.empty:
    st.info("No flagged transactions match these filters.")
else:
    st.dataframe(
        filtered_flagged[show_cols],
        use_container_width=True,
        hide_index=True
    )

    filtered_csv = filtered_flagged.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download filtered audit results (CSV)",
        data=filtered_csv,
        file_name="filtered_expenses_audit_report.csv",
        mime="text/csv",
        key="download_filtered_audit"
    )

st.subheader("All analysed transactions")
with st.expander("View full results"):
    st.dataframe(results, use_container_width=True, hide_index=True)

csv_bytes = flagged.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download flagged transactions (CSV)",
    data=csv_bytes,
    file_name="flagged_expenses_audit_report.csv",
    mime="text/csv",
)

st.divider()
st.caption(
    "Methodology: deterministic duplicate matching + configurable category limits + Isolation Forest on amount. "
    "The risk score is a transparent prioritisation score, not a calibrated probability of fraud. "
    "Review flagged claims with supporting receipts and company policy before taking action."
)
