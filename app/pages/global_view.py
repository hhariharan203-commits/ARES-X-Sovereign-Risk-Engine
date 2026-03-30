from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

# ✅ FIXED IMPORT
from app.utils import (
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
    latest["risk_level"] = [risk_label(p) for p in probs]

    return latest.sort_values("crisis_prob", ascending=False)


# =========================
# ALERTS
# =========================
def compute_alerts(df: pd.DataFrame, model):
    records = []

    for country, g in df.sort_values("month").groupby("country"):
        tail = g.tail(2)
        if len(tail) < 1:
            continue

        aligned = align_features(tail).fillna(0)

        if hasattr(model, "feature_names_in_"):
            aligned = aligned.reindex(columns=model.feature_names_in_, fill_value=0)

        aligned = aligned.astype(float)

        probs = model.predict_proba(aligned)[:, 1]

        if len(probs) == 1:
            latest_prob, prev_prob = probs[0], None
        else:
            prev_prob, latest_prob = probs[-2], probs[-1]

        change = None if prev_prob is None else latest_prob - prev_prob

        records.append({
            "country": country,
            "latest_prob": latest_prob,
            "prev_prob": prev_prob,
            "change": change,
        })

    alert_df = pd.DataFrame(records)

    if alert_df.empty:
        return None, None, None

    top_risk = alert_df.sort_values("latest_prob", ascending=False).iloc[0]

    inc_df = alert_df.dropna(subset=["change"]).sort_values("change", ascending=False)
    top_increase = inc_df.iloc[0] if not inc_df.empty else None

    return alert_df, top_risk, top_increase


# =========================
# MAIN
# =========================
def main():
    st.title("Global Risk")

    df = load_data()
    model = load_model()

    pred_df = latest_predictions(df, model)
    alert_df, top_risk, top_increase = compute_alerts(df, model)

    # =========================
    # ALERTS
    # =========================
    if top_risk is not None:
        st.warning(
            f"Top Risk: **{top_risk['country']}** at {top_risk['latest_prob']:.1%}",
            icon="⚠️",
        )

    if top_increase is not None and top_increase["change"] > 0:
        st.error(
            f"Rising Risk: **{top_increase['country']}** increased by {top_increase['change']:.1%}",
            icon="⬆️",
        )

    # =========================
    # TABLE
    # =========================
    st.subheader("Latest Risk by Country")

    st.dataframe(
        pred_df[["country", "crisis_prob", "risk_level"]]
        .style.format({"crisis_prob": "{:.2%}"}),
        use_container_width=True,
    )

    # =========================
    # TOP 5 BAR
    # =========================
    st.subheader("Top 5 Most Risky Countries")

    top5 = pred_df.head(5)

    fig = px.bar(
        top5,
        x="crisis_prob",
        y="country",
        orientation="h",
        text=top5["crisis_prob"].map(lambda x: f"{x:.1%}"),
        color="crisis_prob",
        color_continuous_scale=["green", "orange", "red"],
        title="Top 5 Crisis Probabilities",
    )

    fig = apply_dark_theme(fig)

    fig.update_layout(
        xaxis_tickformat=".0%",
        coloraxis_showscale=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # HEATMAP
    # =========================
    st.subheader("Global Risk Heatmap")

    fig_map = px.choropleth(
        pred_df,
        locations="country",
        locationmode="ISO-3",
        color="crisis_prob",
        color_continuous_scale=[[0, "green"], [0.5, "yellow"], [1, "red"]],
        title="Global Crisis Probability",
    )

    fig_map = apply_dark_theme(fig_map)

    fig_map.update_layout(
        geo=dict(bgcolor="#0e1117"),
    )

    st.plotly_chart(fig_map, use_container_width=True)

    # =========================
    # FULL TABLE
    # =========================
    st.subheader("Full Country Table")

    st.dataframe(
        pred_df.sort_values("crisis_prob", ascending=False)
        .style.format({"crisis_prob": "{:.2%}"}),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
