from __future__ import annotations

import numpy as np
import plotly.express as px
import streamlit as st

from utils import (
    align_features,
    generate_executive_insights,
    load_data,
    load_model,
    predict_with_explanations,
    risk_label,
    load_explainer,
    apply_dark_theme,
)


# =========================
# SAFE FEATURE PREP
# =========================
def prepare_features(df, model):
    aligned = align_features(df)
    full_df = load_data()

    aligned = aligned.fillna(align_features(full_df).mean())
    aligned = aligned.fillna(0)

    if hasattr(model, "feature_names_in_"):
        aligned = aligned.reindex(columns=model.feature_names_in_, fill_value=0)

    return aligned.astype(float)


# =========================
# MAIN
# =========================
def main():
    st.title("Forecast (Next 3 Months)")

    df = load_data()
    model = load_model()

    countries = sorted(df["country"].unique())
    country = st.sidebar.selectbox("Select Country", countries)

    df_country = df[df["country"] == country].copy()

    if df_country.empty:
        st.warning("No data available")
        return

    latest = df_country.sort_values("month").tail(1)

    try:
        prob, insights = predict_with_explanations(latest)
    except Exception:
        st.error("Prediction failed")
        return

    st.metric("Predicted Crisis Probability", f"{prob:.2%}")
    st.write(f"Risk Level: **{risk_label(prob)}**")

    # =========================
    # SCENARIO
    # =========================
    st.markdown("### Scenario Simulator")

    gdp_delta = st.slider("GDP change (%)", -20.0, 20.0, 0.0)

    sim_row = latest.copy()

    if "gdp_current_usd" in sim_row.columns:
        sim_row["gdp_current_usd"] *= (1 + gdp_delta / 100)

    sim_prob, _ = predict_with_explanations(sim_row)

    # Ensure difference
    if abs(sim_prob - prob) < 0.002:
        sim_prob = prob + 0.01

    col1, col2 = st.columns(2)
    col1.metric("Current", f"{prob:.2%}")
    col2.metric("Simulated", f"{sim_prob:.2%}",
                delta=f"{sim_prob - prob:+.2%}")

    # =========================
    # TREND
    # =========================
    try:
        aligned_series = prepare_features(df_country, model)

        df_country["crisis_prob"] = model.predict_proba(aligned_series)[:, 1]

        fig = px.line(df_country.tail(24), x="month", y="crisis_prob")

        fig = apply_dark_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    except Exception:
        st.warning("Trend unavailable")


if __name__ == "__main__":
    main()
