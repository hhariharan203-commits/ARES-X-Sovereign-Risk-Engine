from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ✅ FIXED IMPORT
from app.utils import (
    align_features,
    load_data,
    load_model,
    risk_label,
    load_explainer,
    apply_dark_theme,
    humanize_feature,
)


# =========================
# ADD PROBABILITIES
# =========================
def add_probabilities(df: pd.DataFrame, model) -> pd.DataFrame:
    aligned = align_features(df).fillna(0)

    if hasattr(model, "feature_names_in_"):
        aligned = aligned.reindex(columns=model.feature_names_in_, fill_value=0)

    aligned = aligned.astype(float)

    df = df.copy()
    df["crisis_prob"] = model.predict_proba(aligned)[:, 1]
    df["risk_level"] = df["crisis_prob"].apply(risk_label)

    return df


# =========================
# SHAP DRIVERS (FIXED)
# =========================
def top_shap_drivers(row: pd.DataFrame, explainer, top_n: int = 2):
    aligned = align_features(row).iloc[[0]]

    shap_values = explainer.shap_values(aligned)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.array(shap_values).flatten()
    cols = list(aligned.columns)

    top_idx = np.argsort(np.abs(shap_values))[::-1][:top_n]

    drivers = []
    for i in top_idx:
        feat = cols[int(i)]
        direction = "↑" if shap_values[int(i)] > 0 else "↓"

        # ✅ HUMAN READABLE FIX
        drivers.append(f"{humanize_feature(feat)} {direction}")

    return drivers


# =========================
# MAIN
# =========================
def main():
    st.title("Executive Dashboard")

    df = load_data()
    model = load_model()
    explainer = load_explainer()

    df = add_probabilities(df, model)

    latest = (
        df.sort_values("month")
        .groupby("country")
        .tail(1)
        .sort_values("crisis_prob", ascending=False)
    )

    # =========================
    # KPI SECTION
    # =========================
    total_countries = latest["country"].nunique()
    high_risk = (latest["risk_level"] == "HIGH").sum()
    avg_prob = latest["crisis_prob"].mean()
    top_risk = latest.iloc[0] if not latest.empty else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Countries", total_countries)
    col2.metric("High Risk Countries", high_risk)
    col3.metric("Avg Crisis Probability", f"{avg_prob:.2%}")

    if top_risk is not None:
        col4.metric(
            "Top Risk Country",
            top_risk["country"],
            f"{top_risk['crisis_prob']:.1%}"
        )
    else:
        col4.metric("Top Risk Country", "N/A")

    # =========================
    # RISK DISTRIBUTION
    # =========================
    st.subheader("Risk Distribution")

    dist = latest["risk_level"].value_counts().reset_index()
    dist.columns = ["risk_level", "count"]

    fig_dist = px.bar(dist, x="risk_level", y="count", title="Risk Levels")
    fig_dist = apply_dark_theme(fig_dist)

    st.plotly_chart(fig_dist, use_container_width=True)

    # =========================
    # GLOBAL TREND
    # =========================
    st.subheader("Global Risk Trend")

    trend = df.groupby("month")["crisis_prob"].mean().reset_index()

    fig_trend = px.line(trend, x="month", y="crisis_prob")
    fig_trend = apply_dark_theme(fig_trend)

    fig_trend.update_traces(line=dict(width=3))
    fig_trend.update_layout(yaxis_tickformat=".0%")

    st.plotly_chart(fig_trend, use_container_width=True)

    # =========================
    # TOP COUNTRIES
    # =========================
    st.subheader("Top 3 Risk Countries with Drivers")

    top3 = latest.head(3)

    for _, row in top3.iterrows():
        drivers = top_shap_drivers(
            df[df["country"] == row["country"]].tail(1),
            explainer
        )

        st.write(
            f"- **{row['country']}** — {row['crisis_prob']:.1%} risk | Drivers: {', '.join(drivers)}"
        )

    # =========================
    # BENCHMARK
    # =========================
    st.subheader("Country vs Global Benchmark")

    selected = st.selectbox("Select country", sorted(latest["country"].unique()))

    sel_prob = float(latest[latest["country"] == selected]["crisis_prob"])
    diff = sel_prob - avg_prob

    if diff > 0.002:
        label = "Above global risk"
    elif diff < -0.002:
        label = "Below global risk"
    else:
        label = "In line with global average"

    st.write(f"{selected}: {label}")


if __name__ == "__main__":
    main()
