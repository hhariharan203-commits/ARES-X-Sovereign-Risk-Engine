from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (
    align_features,
    load_data,
    load_model,
    risk_label,
    apply_dark_theme,
)


# =========================
# LATEST PREDICTIONS
# =========================
def latest_predictions(df: pd.DataFrame, model) -> pd.DataFrame:
    latest = df.sort_values("month").groupby("country").tail(1).copy()

    aligned = align_features(latest).fillna(0)

    if hasattr(model, "feature_names_in_"):
        aligned = aligned.reindex(columns=model.feature_names_in_, fill_value=0)

    aligned = aligned.astype(float)

    probs = model.predict_proba(aligned)[:, 1]

    latest["crisis_prob"] = probs
    latest["risk_level"] = latest["crisis_prob"].apply(risk_label)

    return latest.sort_values("crisis_prob", ascending=False)


# =========================
# ALERT SYSTEM (IMPROVED)
# =========================
def compute_alerts(df: pd.DataFrame, model):
    records = []

    for country, g in df.sort_values("month").groupby("country"):
        if len(g) < 2:
            continue

        tail = g.tail(2)

        aligned = align_features(tail).fillna(0)

        if hasattr(model, "feature_names_in_"):
            aligned = aligned.reindex(columns=model.feature_names_in_, fill_value=0)

        aligned = aligned.astype(float)

        probs = model.predict_proba(aligned)[:, 1]

        prev, latest = probs[-2], probs[-1]
        change = latest - prev

        records.append({
            "country": country,
            "latest_prob": latest,
            "change": change,
        })

    alert_df = pd.DataFrame(records)

    if alert_df.empty:
        return [], []

    rising = alert_df.sort_values("change", ascending=False).head(3)
    easing = alert_df.sort_values("change", ascending=True).head(3)

    return rising.to_dict("records"), easing.to_dict("records")


# =========================
# MAIN
# =========================
def main():
    st.title("Global Risk")

    df = load_data()
    model = load_model()

    pred_df = latest_predictions(df, model)
    rising, easing = compute_alerts(df, model)

    # =========================
    # KPI SUMMARY
    # =========================
    total = pred_df["country"].nunique()
    high = (pred_df["risk_level"] == "HIGH").sum()
    avg = pred_df["crisis_prob"].mean()
    top = pred_df.iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Countries Covered", total)
    col2.metric("High Risk Countries", high)
    col3.metric("Avg Risk", f"{avg:.2%}")
    col4.metric("Top Risk", top["country"], f"{top['crisis_prob']:.1%}")

    # =========================
    # STRATEGIC ALERTS
    # =========================
    st.subheader("Strategic Risk Signals")

    st.markdown("**Rising Risk (Top 3):**")
    for r in rising:
        st.write(f"- {r['country']} ↑ {r['change']:.2%}")

    st.markdown("**Easing Risk (Top 3):**")
    for r in easing:
        st.write(f"- {r['country']} ↓ {abs(r['change']):.2%}")

    # =========================
    # DISTRIBUTION
    # =========================
    st.subheader("Risk Distribution")

    dist = pred_df["risk_level"].value_counts().reset_index()
    dist.columns = ["risk_level", "count"]

    fig_dist = px.bar(dist, x="risk_level", y="count")
    fig_dist = apply_dark_theme(fig_dist)

    st.plotly_chart(fig_dist, use_container_width=True)

    # =========================
    # TOP 5 COUNTRIES
    # =========================
    st.subheader("Top 5 Risk Countries")

    top5 = pred_df.head(5)

    fig = px.bar(
        top5,
        x="crisis_prob",
        y="country",
        orientation="h",
        text=top5["crisis_prob"].map(lambda x: f"{x:.1%}"),
        color="crisis_prob",
        color_continuous_scale=["green", "orange", "red"],
    )

    fig = apply_dark_theme(fig)
    fig.update_layout(xaxis_tickformat=".0%")

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # GLOBAL HEATMAP
    # =========================
    st.subheader("Global Risk Heatmap")

    fig_map = px.choropleth(
        pred_df,
        locations="country",
        locationmode="ISO-3",
        color="crisis_prob",
        color_continuous_scale=[[0, "green"], [0.5, "yellow"], [1, "red"]],
    )

    fig_map = apply_dark_theme(fig_map)
    fig_map.update_layout(geo=dict(bgcolor="#0e1117"))

    st.plotly_chart(fig_map, use_container_width=True)

    # =========================
    # EXECUTIVE SUMMARY (NEW 🔥)
    # =========================
    st.subheader("Executive Summary")

    summary = (
        f"Global risk average stands at {avg:.1%}. "
        f"{high} countries are currently in HIGH risk zone. "
        f"Highest exposure observed in {top['country']} ({top['crisis_prob']:.1%})."
    )

    st.write(summary)

    # =========================
    # FULL TABLE
    # =========================
    st.subheader("Full Country Table")

    st.dataframe(
        pred_df[["country", "crisis_prob", "risk_level"]]
        .sort_values("crisis_prob", ascending=False)
        .style.format({"crisis_prob": "{:.2%}"}),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
