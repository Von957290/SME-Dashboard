# SME Discrete Manufacturing Dashboard
# Foundations of Business Analytics — final project dashboard
# Install: python -m pip install streamlit pandas numpy plotly scikit-learn
# Run:    python -m streamlit run app.py

import io
import os
import math
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report,
                              confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(page_title="SME Manufacturing Analytics", page_icon="🏭", layout="wide")

# ------------------------------- Styling ------------------------------------
CSS = """
<style>
:root{--navy:#10233f;--blue:#2563eb;--cyan:#0891b2;--ink:#172033;--muted:#64748b;--page:#f3f6fb;--line:#dbe4f0}
.stApp{background:var(--page);color:var(--ink)}
[data-testid="stHeader"]{background:rgba(243,246,251,.92)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#10233f 0%,#183b63 100%)}
[data-testid="stSidebar"] *{color:#f8fafc}
[data-testid="stSidebar"] [data-baseweb="select"]>div,[data-testid="stSidebar"] input{background:#fff!important;color:#172033!important;border-radius:9px!important}
[data-testid="stFileUploader"] section{background:#fff;border:1px dashed #7c9bc8;border-radius:12px}
[data-testid="stMetric"]{background:#fff;border:1px solid #e2e8f0;padding:17px 19px;border-radius:15px;box-shadow:0 5px 18px rgba(15,35,65,.06)}
[data-testid="stMetricLabel"]{color:#64748b;font-weight:650}[data-testid="stMetricValue"]{color:#10233f;font-weight:750}
[data-testid="stPlotlyChart"],[data-testid="stDataFrame"]{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:7px;box-shadow:0 4px 14px rgba(15,35,65,.04)}
h1,h2,h3{color:#10233f;letter-spacing:-.02em}
[data-testid="stButton"] button,[data-testid="stDownloadButton"] button{border-radius:10px;font-weight:650}
.hero{background:linear-gradient(115deg,#10233f 0%,#1d4ed8 68%,#0891b2 100%);padding:27px 31px;border-radius:20px;color:white;margin:0 0 22px 0;box-shadow:0 12px 28px rgba(30,64,175,.18)}
.hero h1{color:white!important;margin:5px 0 8px;font-size:2rem}.hero p{color:#e2eaff!important;margin:0;font-size:1rem}
.kicker{color:#bfdbfe;text-transform:uppercase;font-size:.74rem;font-weight:800;letter-spacing:.13em}
.insight{border-left:4px solid #0891b2;background:#fff;padding:11px 14px;border-radius:0 10px 10px 0;margin:5px 0 14px;color:#334155}
.small-note{font-size:.88rem;color:#64748b}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown("""
<div class="hero">
  <div class="kicker">Foundations of Business Analytics</div>
  <h1>🏭 SME Discrete Manufacturing</h1>
  <p>Operations intelligence · Production monitoring · Product classification</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------- Data loading -------------------------------
@st.cache_data(show_spinner=False)
def parse_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Parse an uploaded CSV from bytes so Streamlit reruns don't reuse a consumed file handle."""
    return pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig", low_memory=False)


def load_data(uploaded_file, fallback_path: str) -> Optional[pd.DataFrame]:
    try:
        if uploaded_file is not None:
            return parse_csv_bytes(uploaded_file.getvalue())
        if os.path.isfile(fallback_path):
            return pd.read_csv(fallback_path, encoding="utf-8-sig", low_memory=False)
    except Exception as exc:
        st.error(f"Could not read {getattr(uploaded_file, 'name', fallback_path)}: {exc}")
    return None


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    # Parse known operational fields as numeric only when present.
    for col in ["items", "power_avg", "cycle_time", "status_time", "alarm", "status"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "ts" in frame.columns:
        frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce", utc=True)
    return frame

with st.sidebar:
    st.header("📂 Data uploads")
    st.caption("Upload the source data used by this dashboard.")
    full_upload = st.file_uploader("Main manufacturing CSV", type=["csv"], key="main_csv")
    labelled_upload = st.file_uploader("Labelled subset CSV", type=["csv"], key="labelled_csv")
    st.divider()
    st.header("⚙️ Dashboard filters")

df = load_data(full_upload, "company_A (1).csv")
labelled = load_data(labelled_upload, "company_A_labelled_subset.csv")
if df is None:
    st.info("Upload your main manufacturing CSV in the sidebar to start. The file should contain operational fields such as asset, items, status, power_avg, cycle_time, alarm, and/or ts.")
    st.stop()
if df.empty:
    st.error("The main CSV has headers but no data rows. Please choose a non-empty CSV file.")
    st.stop()

df = normalize_columns(df)
if labelled is not None:
    labelled = normalize_columns(labelled)

# ------------------------- Project foundation -------------------------------
st.subheader("Project foundation")
with st.expander("Audience, business problem, and analytical approach", expanded=False):
    st.markdown("**Target audience:** SME production supervisors and operations managers.")
    st.markdown("**Decisions supported:** compare machine activity and production, review operating-state distribution, investigate alarm patterns, and examine how power use relates to cycle time.")
    st.markdown("**Problem statement:** Manufacturing supervisors need a consolidated view of production activity, machine operating states, power use, cycle time, and alarms to identify where further investigation or process review may be warranted. This dashboard summarizes the available operational records and, when labelled product data are provided, tests whether operational features can classify product labels on held-out data. The results support exploratory decision-making and are not a substitute for engineering validation.")
    st.markdown("**Advanced analytics type:** supervised multiclass classification, using `product` as the target and available operational fields as predictors. This choice is conditional on the labelled dataset containing sufficient examples and useful features.")

# ---------------------------- Interactive filters ---------------------------
filtered = df.copy()
with st.sidebar:
    if "asset" in filtered.columns:
        assets = sorted(filtered["asset"].dropna().astype(str).unique().tolist())
        selected_assets = st.multiselect("Machine / asset", options=assets, default=assets)
        filtered = filtered[filtered["asset"].astype(str).isin(selected_assets)]
    if "status" in filtered.columns:
        status_values = sorted(filtered["status"].dropna().unique().tolist())
        status_labels = {0: "0 — Idle", 1: "1 — Manual Production", 2: "2 — Automatic Production", 3: "3 — Alarm / Interrupted"}
        status_options = st.multiselect("Operating status", options=status_values, default=status_values,
                                        format_func=lambda x: status_labels.get(x, str(x)))
        filtered = filtered[filtered["status"].isin(status_options)]
    if "ts" in filtered.columns and filtered["ts"].notna().any():
        valid_ts = filtered["ts"].dropna()
        min_date, max_date = valid_ts.min().date(), valid_ts.max().date()
        selected_dates = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
            start = pd.Timestamp(selected_dates[0], tz="UTC")
            end = pd.Timestamp(selected_dates[1], tz="UTC") + pd.Timedelta(days=1)
            filtered = filtered[filtered["ts"].ge(start) & filtered["ts"].lt(end)]

# ------------------------------ Headline KPIs -------------------------------
def safe_sum(series):
    return series.sum(min_count=1) if series.notna().any() else np.nan

def fmt_num(value, digits=0):
    return "N/A" if pd.isna(value) else f"{value:,.{digits}f}"

st.caption(f"Main dataset: {len(df):,} rows · {len(df.columns)} columns · Filtered view: {len(filtered):,} rows")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Records in view", f"{len(filtered):,}")
k2.metric("Total production items", fmt_num(safe_sum(filtered["items"])) if "items" in filtered else "N/A")
k3.metric("Average power", fmt_num(filtered["power_avg"].mean(), 2) if "power_avg" in filtered and filtered["power_avg"].notna().any() else "N/A")
k4.metric("Records with alarm", f"{int(filtered['alarm'].gt(0).sum()):,}" if "alarm" in filtered and filtered["alarm"].notna().any() else "N/A")

st.subheader("Operations overview")
st.caption("Use these views to compare production, operating states, trends, power/cycle-time relationships, and alarm concentrations.")

# ---------------------------- Five visuals ----------------------------------
chart_count = 0
col1, col2 = st.columns(2)
with col1:
    if "asset" in filtered.columns and "items" in filtered.columns:
        asset_prod = filtered.groupby("asset", dropna=False)["items"].sum(min_count=1).reset_index()
        fig = px.bar(asset_prod, x="asset", y="items", title="1. Which machines contribute the most production?",
                     labels={"asset":"Machine / asset", "items":"Total items"}, color="items", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True); chart_count += 1
        if not asset_prod.empty and asset_prod["items"].notna().any():
            top = asset_prod.loc[asset_prod["items"].idxmax()]
            st.markdown(f'<div class="insight"><b>Insight:</b> {top["asset"]} has the highest recorded total ({top["items"]:,.0f} items) in the current filtered view. Review workload distribution before making capacity changes.</div>', unsafe_allow_html=True)
    elif "asset" in filtered.columns:
        asset_counts = filtered["asset"].astype(str).value_counts().rename_axis("asset").reset_index(name="records")
        fig = px.bar(asset_counts, x="asset", y="records", title="1. Which machines have the most observations?", labels={"asset":"Machine / asset", "records":"Records"})
        st.plotly_chart(fig, use_container_width=True); chart_count += 1
        st.markdown('<div class="insight"><b>Insight:</b> Observation counts show how often each asset appears in the dataset; they do not necessarily indicate production capacity.</div>', unsafe_allow_html=True)
    else:
        st.info("Machine comparison requires an `asset` field.")
with col2:
    if "status" in filtered.columns:
        status_names = {0:"Idle", 1:"Manual Production", 2:"Automatic Production", 3:"Alarm / Interrupted"}
        status_df = filtered["status"].map(lambda x: status_names.get(x, str(x))).value_counts(dropna=False).rename_axis("status").reset_index(name="records")
        fig = px.pie(status_df, names="status", values="records", title="2. How are recorded operating states distributed?", hole=0.42)
        st.plotly_chart(fig, use_container_width=True); chart_count += 1
        if not status_df.empty:
            most_common = status_df.iloc[0]
            pct = 100 * most_common["records"] / max(len(filtered), 1)
            st.markdown(f'<div class="insight"><b>Insight:</b> {most_common["status"]} is the most common recorded state ({pct:.1f}% of the filtered rows). Check whether this pattern matches the expected production schedule.</div>', unsafe_allow_html=True)
    else:
        st.info("Operating-state distribution requires a `status` field.")

col3, col4 = st.columns(2)
with col3:
    if "ts" in filtered.columns and "items" in filtered.columns and filtered["ts"].notna().any():
        time_df = filtered.dropna(subset=["ts"]).set_index("ts").resample("D")["items"].sum(min_count=1).reset_index()
        fig = px.line(time_df, x="ts", y="items", markers=True, title="3. How does production change over time?", labels={"ts":"Date", "items":"Total items"})
        st.plotly_chart(fig, use_container_width=True); chart_count += 1
        nonnull = time_df.dropna(subset=["items"])
        if len(nonnull):
            peak = nonnull.loc[nonnull["items"].idxmax()]
            st.markdown(f'<div class="insight"><b>Insight:</b> The highest daily total in the filtered period is {peak["items"]:,.0f} items on {peak["ts"].date()}. Compare peak and low days with shifts, orders, and downtime records.</div>', unsafe_allow_html=True)
    else:
        st.info("The daily production trend requires valid `ts` timestamps and an `items` field.")
with col4:
    if "power_avg" in filtered.columns and "cycle_time" in filtered.columns:
        scatter = filtered.dropna(subset=["power_avg", "cycle_time"])
        if len(scatter) > 5000:
            scatter = scatter.sample(5000, random_state=42)
        color = "status" if "status" in scatter.columns else None
        fig = px.scatter(scatter, x="cycle_time", y="power_avg", color=color,
                         hover_data=[c for c in ["asset", "items", "alarm"] if c in scatter.columns],
                         title="4. How do power use and cycle time vary together?",
                         labels={"cycle_time":"Cycle time", "power_avg":"Average power", "status":"Operating status"})
        st.plotly_chart(fig, use_container_width=True); chart_count += 1
        valid = filtered[["power_avg", "cycle_time"]].dropna()
        corr = valid["power_avg"].corr(valid["cycle_time"]) if len(valid) >= 2 else np.nan
        corr_text = f"Pearson correlation is {corr:.2f}" if pd.notna(corr) else "There are not enough valid pairs to calculate correlation"
        st.markdown(f'<div class="insight"><b>Insight:</b> {corr_text}. This describes association only; it does not establish that cycle time causes changes in power use.</div>', unsafe_allow_html=True)
    else:
        st.info("The power/cycle-time view requires `power_avg` and `cycle_time` fields.")

if "alarm" in filtered.columns and ("asset" in filtered.columns or "status" in filtered.columns):
    group_col = "asset" if "asset" in filtered.columns else "status"
    alarm_summary = filtered.groupby(group_col, dropna=False)["alarm"].sum(min_count=1).reset_index()
    fig = px.bar(alarm_summary, x=group_col, y="alarm", title=f"5. Where are alarm values concentrated by {group_col}?",
                 labels={group_col: "Machine / asset" if group_col == "asset" else "Operating status", "alarm":"Sum of alarm field"}, color="alarm", color_continuous_scale="OrRd")
    st.plotly_chart(fig, use_container_width=True); chart_count += 1
    if len(alarm_summary) and alarm_summary["alarm"].notna().any():
        highest = alarm_summary.loc[alarm_summary["alarm"].idxmax()]
        st.markdown(f'<div class="insight"><b>Insight:</b> {highest[group_col]} has the largest summed alarm field ({highest["alarm"]:,.2f}) in this view. Confirm what the source alarm values encode before treating the sum as a count of incidents.</div>', unsafe_allow_html=True)
else:
    st.info("The alarm comparison requires an `alarm` field and either `asset` or `status`.")

# ------------------------ Data quality and export ----------------------------
with st.expander("View filtered records and data-quality summary"):
    st.write(f"Filtered records: **{len(filtered):,}** | Columns: **{len(filtered.columns)}**")
    st.dataframe(filtered.head(1000), use_container_width=True)
    quality = pd.DataFrame({"missing_count": filtered.isna().sum(), "missing_percent": (filtered.isna().mean() * 100).round(2)})
    st.dataframe(quality, use_container_width=True)
    st.download_button("Download filtered data as CSV", filtered.to_csv(index=False).encode("utf-8-sig"), "filtered_manufacturing_data.csv", "text/csv")

# -------------------- Findings summary (data-backed) ------------------------
st.subheader("EDA findings summary")
st.caption("Up to 10 observations are calculated from the currently loaded data and active filters. Findings are descriptive, not causal claims.")
findings = []
findings.append(f"The current filtered view contains {len(filtered):,} rows from a source dataset of {len(df):,} rows ({len(filtered) / max(len(df), 1) * 100:.1f}% retained by the active filters).")
if "asset" in filtered.columns:
    asset_count = filtered["asset"].nunique(dropna=True)
    findings.append(f"The filtered view contains {asset_count:,} distinct assets.")
if "items" in filtered.columns and filtered["items"].notna().any():
    findings.append(f"Total recorded production items are {safe_sum(filtered['items']):,.0f}; the mean is {filtered['items'].mean():,.2f} items per row.")
if "asset" in filtered.columns and "items" in filtered.columns:
    by_asset = filtered.groupby("asset")["items"].sum(min_count=1).dropna()
    if len(by_asset):
        findings.append(f"Asset {by_asset.idxmax()} has the highest total recorded output ({by_asset.max():,.0f} items) in this filtered view.")
if "status" in filtered.columns and filtered["status"].notna().any():
    status_counts = filtered["status"].value_counts()
    top_status = status_counts.index[0]
    findings.append(f"Status code {top_status} is most common ({status_counts.iloc[0]:,} rows; {status_counts.iloc[0] / max(len(filtered), 1) * 100:.1f}%). Confirm status-code meanings against the dataset documentation.")
if "alarm" in filtered.columns and filtered["alarm"].notna().any():
    alarm_positive = int(filtered["alarm"].gt(0).sum())
    findings.append(f"{alarm_positive:,} rows ({alarm_positive / max(len(filtered), 1) * 100:.1f}%) have an alarm value greater than zero.")
if "power_avg" in filtered.columns and "cycle_time" in filtered.columns:
    pair = filtered[["power_avg", "cycle_time"]].dropna()
    if len(pair) >= 2:
        corr_value = pair["power_avg"].corr(pair["cycle_time"])
        findings.append(f"Power and cycle time have Pearson correlation {corr_value:.3f} across {len(pair):,} valid pairs; this is association, not causation.")
if "ts" in filtered.columns and filtered["ts"].notna().any():
    findings.append(f"Valid timestamps span {filtered['ts'].min().date()} through {filtered['ts'].max().date()} in the filtered view.")
missing_pct = filtered.isna().mean() * 100
if len(missing_pct):
    missing_col = missing_pct.idxmax()
    findings.append(f"The field with the highest missing-data rate is `{missing_col}` ({missing_pct.max():.2f}% missing).")
if "power_avg" in filtered.columns and filtered["power_avg"].notna().sum() >= 4:
    q1, q3 = filtered["power_avg"].quantile([0.25, 0.75])
    iqr = q3 - q1
    if pd.notna(iqr):
        outlier_count = int(((filtered["power_avg"] < q1 - 1.5 * iqr) | (filtered["power_avg"] > q3 + 1.5 * iqr)).sum())
        findings.append(f"The IQR rule flags {outlier_count:,} power readings ({outlier_count / max(len(filtered), 1) * 100:.2f}%) as potential outliers; investigate before removing any values.")
if "power_avg" in filtered.columns and filtered["power_avg"].notna().any():
    findings.append(f"Power readings have a median of {filtered['power_avg'].median():,.3f} and a maximum of {filtered['power_avg'].max():,.3f}; inspect the distribution for skew or extreme values.")
if labelled is not None and "product" in labelled.columns:
    product_counts = labelled["product"].dropna().value_counts()
    if len(product_counts):
        findings.append(f"The labelled subset contains {len(product_counts):,} product classes; the largest class has {product_counts.iloc[0]:,} records and the smallest has {product_counts.iloc[-1]:,}.")

if not findings:
    st.info("No EDA findings can be calculated from the available columns.")
else:
    for i, finding in enumerate(findings[:10], start=1):
        st.markdown(f"**{i}.** {finding}")
    if len(findings) < 8:
        st.warning(f"Only {len(findings)} findings could be calculated from the available fields. Load the complete dataset to generate more supported findings; do not invent additional findings.")

st.markdown("**Business implication:** Use these findings to prioritize questions for supervisors (asset workload, operating-state patterns, alarms, data quality, and production variation). Validate unusual patterns with maintenance logs, shift schedules, and source definitions before changing operations.")

# --------------------------- Advanced analytics -----------------------------
st.divider()
st.header("Advanced analytics: product classification")
st.write("Target: `product`. Predictors are limited to available operational fields (`items`, `power_avg`, `cycle_time`, `status_time`, `status`, `alarm`, `asset`). The experiment compares a majority-class baseline, Logistic Regression, and Random Forest. It reports holdout metrics, multiclass ROC-AUC when calculable, a confusion matrix, and cross-validation where class counts permit.")
st.markdown("**Why this technique?** If product labels are reliably recorded, classification can test whether operational measurements distinguish product classes. This is exploratory and should be justified in the final paper with specific EDA evidence; it is not a production-control recommendation by itself.")

if labelled is None:
    st.warning("Upload the labelled subset CSV to enable model evaluation. The main dashboard can run without it.")
elif "product" not in labelled.columns:
    st.error("The labelled subset does not contain the required `product` target column. Check the actual target column name in your CSV.")
else:
    model_df = labelled.dropna(subset=["product"]).copy()
    feature_candidates = [c for c in ["items", "power_avg", "cycle_time", "status_time", "status", "alarm", "asset"] if c in model_df.columns]
    # Convert target labels to consistent strings to avoid mixed numeric/string class-label issues.
    y = model_df["product"].astype(str).str.strip()
    valid_y = y.ne("") & y.ne("nan") & y.ne("None")
    model_df = model_df.loc[valid_y].copy()
    y = y.loc[valid_y]
    if len(model_df) < 20:
        st.warning(f"Only {len(model_df)} labelled rows are available. At least 20 are required for this exploratory holdout evaluation.")
    elif y.nunique() < 2:
        st.warning("The `product` target has fewer than two classes after removing missing labels.")
    elif not feature_candidates:
        st.warning("No supported operational predictor columns were found in the labelled subset.")
    else:
        X = model_df[feature_candidates].copy()
        # Keep numeric data numeric; treat asset-like fields as categorical.
        numeric_features = [c for c in feature_candidates if c != "asset" and pd.api.types.is_numeric_dtype(X[c])]
        categorical_features = [c for c in feature_candidates if c not in numeric_features]
        # Some numeric-looking columns can arrive as strings; coerce known operational fields.
        for col in ["items", "power_avg", "cycle_time", "status_time", "status", "alarm"]:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")
        numeric_features = [c for c in feature_candidates if c != "asset" and pd.api.types.is_numeric_dtype(X[c])]
        categorical_features = [c for c in feature_candidates if c not in numeric_features]

        def make_preprocessor():
            transformers = []
            if numeric_features:
                transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_features))
            if categorical_features:
                transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features))
            return ColumnTransformer(transformers, remainder="drop")

        class_counts = y.value_counts()
        n_classes = y.nunique()
        # Use stratification only when each class can appear in both partitions and both sets can contain every class.
        test_n = max(int(math.ceil(len(y) * 0.25)), n_classes)
        train_n = len(y) - test_n
        can_stratify = class_counts.min() >= 2 and test_n >= n_classes and train_n >= n_classes
        test_size = test_n if can_stratify else 0.25
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42,
                stratify=y if can_stratify else None
            )
            if y_train.nunique() < 2:
                st.warning("The training partition contains only one product class. There are not enough examples to fit and compare multiclass classifiers reliably. Add more labelled examples or review the product label distribution.")
                st.stop()
            models = {
                "Majority-class baseline": DummyClassifier(strategy="most_frequent"),
                "Logistic Regression": LogisticRegression(max_iter=2000, class_weight=None),
                "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42, class_weight=None, n_jobs=-1, min_samples_leaf=2)
            }
            result_rows, fitted_models, train_scores, test_predictions = [], {}, {}, {}
            for name, estimator in models.items():
                # Build a fresh preprocessor per model so no fitted state is shared across pipelines.
                model_pipe = Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])
                model_pipe.fit(X_train, y_train)
                train_pred = model_pipe.predict(X_train)
                test_pred = model_pipe.predict(X_test)
                train_scores[name] = accuracy_score(y_train, train_pred)
                test_predictions[name] = test_pred
                row = {
                    "Model": name,
                    "Train accuracy": train_scores[name],
                    "Test accuracy": accuracy_score(y_test, test_pred),
                    "Precision (weighted)": precision_score(y_test, test_pred, average="weighted", zero_division=0),
                    "Recall (weighted)": recall_score(y_test, test_pred, average="weighted", zero_division=0),
                    "F1 (weighted)": f1_score(y_test, test_pred, average="weighted", zero_division=0),
                    "ROC-AUC (macro OVR)": np.nan,
                }
                # Multiclass ROC-AUC requires probability estimates and at least two represented test classes.
                if hasattr(model_pipe.named_steps["model"], "predict_proba") and y_test.nunique() >= 2:
                    try:
                        probabilities = model_pipe.predict_proba(X_test)
                        model_classes = model_pipe.named_steps["model"].classes_
                        if len(model_classes) == 2:
                            row["ROC-AUC (macro OVR)"] = roc_auc_score(y_test, probabilities[:, 1])
                        else:
                            row["ROC-AUC (macro OVR)"] = roc_auc_score(
                                y_test, probabilities, labels=model_classes,
                                multi_class="ovr", average="macro"
                            )
                    except (ValueError, TypeError):
                        # AUC is left unavailable if the holdout split omits classes or scores are undefined.
                        pass
                result_rows.append(row)
                fitted_models[name] = model_pipe

            result_df = pd.DataFrame(result_rows).set_index("Model")
            st.subheader("Holdout evaluation")
            st.dataframe(result_df.style.format("{:.3f}"), use_container_width=True)
            st.caption("ROC-AUC is macro one-vs-rest. It is shown as N/A when the test split or probability outputs do not support a valid multiclass calculation.")

            # Cross-validation is performed on training data only; the holdout test set remains untouched.
            st.subheader("Cross-validation on training data")
            min_train_class = int(y_train.value_counts().min())
            cv_folds = min(3, min_train_class)
            if cv_folds >= 2:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                cv_rows = []
                for cv_name in ["Logistic Regression", "Random Forest"]:
                    cv_pipe = Pipeline([("preprocess", make_preprocessor()), ("model", models[cv_name])])
                    try:
                        cv_scores = cross_val_score(cv_pipe, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=1)
                        cv_rows.append({"Model": cv_name, "Folds": cv_folds, "Mean weighted F1": cv_scores.mean(), "Std. deviation": cv_scores.std()})
                    except Exception as cv_exc:
                        st.warning(f"Cross-validation for {cv_name} could not be completed: {cv_exc}")
                if cv_rows:
                    cv_df = pd.DataFrame(cv_rows).set_index("Model")
                    st.dataframe(cv_df.style.format({"Mean weighted F1": "{:.3f}", "Std. deviation": "{:.3f}"}), use_container_width=True)
                    st.caption("Cross-validation uses only the training partition. The held-out test partition is not used to tune or select model parameters.")
            else:
                st.warning("Cross-validation was skipped because at least one product class has fewer than two training examples. More labelled examples are needed for stratified folds.")

            best_candidates = [n for n in result_df.index if n != "Majority-class baseline"]
            best_name = result_df.loc[best_candidates, "F1 (weighted)"].idxmax()
            baseline_f1 = result_df.loc["Majority-class baseline", "F1 (weighted)"]
            best_f1 = result_df.loc[best_name, "F1 (weighted)"]
            if best_f1 > baseline_f1:
                st.success(f"Best tested classifier by weighted F1: {best_name} ({best_f1:.3f}); majority baseline F1: {baseline_f1:.3f}.")
            else:
                st.warning(f"Neither classifier beat the majority-class baseline by weighted F1 on this split. Best classifier: {best_name} ({best_f1:.3f}); baseline: {baseline_f1:.3f}. Do not claim useful predictive performance from this result.")
            st.caption("A fixed random seed supports reproducibility. Stratification is used only when class counts and split sizes permit it. If classes are rare, the split may not represent every class in both partitions. Results are a single holdout estimate, not proof of production readiness.")

            # Feature importance for the Random Forest, with transformed names where supported.
            rf_pipe = fitted_models["Random Forest"]
            rf_model = rf_pipe.named_steps["model"]
            rf_pre = rf_pipe.named_steps["preprocess"]
            if hasattr(rf_model, "feature_importances_"):
                try:
                    feature_names = rf_pre.get_feature_names_out()
                    importance = pd.DataFrame({"Feature": feature_names, "Importance": rf_model.feature_importances_}).sort_values("Importance", ascending=False).head(12)
                    st.subheader("Random Forest: most important transformed features")
                    fig_imp = px.bar(importance.sort_values("Importance"), x="Importance", y="Feature", orientation="h", title="Which features contributed most to Random Forest splits?", labels={"Importance":"Relative importance", "Feature":"Model feature"})
                    st.plotly_chart(fig_imp, use_container_width=True)
                    st.caption("Feature importance indicates model reliance within this fitted model; it does not establish causality.")
                except Exception as exc:
                    st.info(f"Feature-importance names could not be displayed for this scikit-learn version: {exc}")

            # Train/test gap is a simple diagnostic, not a definitive overfitting test.
            train_acc = result_df.loc[best_name, "Train accuracy"]
            test_acc = result_df.loc[best_name, "Test accuracy"]
            gap = train_acc - test_acc
            st.subheader("Overfitting check and interpretation")
            st.write(f"For {best_name}, train accuracy is {train_acc:.3f}, test accuracy is {test_acc:.3f}, and the train–test gap is {gap:.3f}.")
            if gap > 0.15:
                st.warning("The train–test accuracy gap is relatively large; this may indicate overfitting. Validate with cross-validation and check whether predictors leak information about the product label.")
            else:
                st.info("The train–test accuracy gap is not large under this simple check, but this does not rule out overfitting or data leakage. Validate further before operational use.")
            st.subheader("Classification report for best classifier")
            best_test_pred = test_predictions[best_name]
            st.text(classification_report(y_test, best_test_pred, zero_division=0))

            st.subheader("Confusion matrix")
            matrix_labels = sorted(pd.Index(y_test.astype(str)).union(pd.Index(pd.Series(best_test_pred).astype(str))).tolist())
            cm = confusion_matrix(y_test.astype(str), pd.Series(best_test_pred).astype(str), labels=matrix_labels)
            cm_df = pd.DataFrame(cm, index=matrix_labels, columns=matrix_labels)
            fig_cm = px.imshow(
                cm_df, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                labels={"x": "Predicted product class", "y": "Actual product class", "color": "Records"},
                title=f"Confusion matrix — {best_name}"
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            st.caption("Diagonal cells are correct predictions; off-diagonal cells show misclassifications. Classes absent from the test set cannot be assessed in this matrix.")

            st.download_button("Download model comparison as CSV", result_df.reset_index().to_csv(index=False).encode("utf-8-sig"), "classification_model_comparison.csv", "text/csv")
        except Exception as exc:
            st.error(f"Model evaluation could not be completed: {type(exc).__name__}: {exc}")
            st.info("Check that the labelled CSV has a `product` target and supported operational feature columns. If this is the same class-weight error as before, make sure you started this exact app.py file, not an older app_designed.py copy.")

# ------------------------ Limitations / next actions -------------------------
st.divider()
st.subheader("Limitations and recommended next steps")
st.markdown("""
- **Data definitions:** Confirm the meaning and units of `items`, `power_avg`, `cycle_time`, `status_time`, `alarm`, and `status` against the dataset documentation.
- **Descriptive results:** The dashboard summarizes recorded rows; missing data, uneven logging, or sampling can affect interpretation.
- **Classification:** Results depend on the labelled subset, class balance, and selected predictors. The app reports a holdout evaluation and training-only stratified cross-validation where class counts permit. If records are time-dependent or repeated by machine, a time- or group-aware validation design may be more appropriate than a random split. Check for target leakage before interpreting scores.
- **Business action:** Use high alarm totals, workload differences, or unusual power/cycle-time patterns as investigation prompts. Confirm with maintenance logs and production staff before changing schedules or machine settings.
- **Submission items outside the app:** The project brief also requires the project plan/findings summary, Google Colab notebook, CSV dataset, 6–8 page IEEE-format paper, and a 5–10 minute recorded presentation. The dashboard does not replace these deliverables.
""")
st.caption("SME Discrete Manufacturing Analytics · Exploratory decision support, not an automated production-control system")
