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
# FEATURE PREP
# =========================
def prepare_features(df, model):
    aligned = align_features(df)

    full_df = load_data()

    if "country" in df.columns:
        country = df["country"].iloc[0]
        country_hist = full_df[full_df["country"] == country]
        aligned_country = align_features(country_hist)
    else:
        aligned_country = None

    aligned_global = align_features(full_df)

    numeric_cols = aligned.columns

    if aligned_country is not None and not aligned_country.empty:
        aligned[numeric_cols] = aligned[numeric_cols].fillna(
            aligned_country[numeric_cols].mean()
        )

    aligned[numeric_cols] = aligned[numeric_cols].fillna(
        aligned_global[numeric_cols].mean()
    )

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
    explainer = load_explainer()

    countries = sorted(df["country"].unique())
    country = st.sidebar.selectbox("Select Country", countries)

    df_country = df[df["country"] == country].copy()

    latest = df_country.sort_values("month").tail(1)

    prob, insights = predict_with_explanations(latest)

    st.metric("Predicted Crisis Probability", f"{prob:.2%}")
    st.write(f"Risk Level: **{risk_label(prob)}**")

    st.markdown("**Top Drivers (SHAP):**")
    for txt in insights:
        st.write(f"- {txt}")

    # =========================
    # SCENARIO
    # =========================
    st.markdown("### Scenario Simulator")

    gdp_delta = st.slider("GDP change (%)", -20.0, 20.0, 0.0)
    imp_delta = st.slider("Imports change (%)", -20.0, 20.0, 0.0)
    exp_delta = st.slider("Exports change (%)", -20.0, 20.0, 0.0)

    sim_row = latest.copy()

    sim_row["gdp_current_usd"] *= (1 + gdp_delta / 100)
    sim_row["imports_pct_gdp"] *= (1 + imp_delta / 100)
    sim_row["exports_pct_gdp"] *= (1 + exp_delta / 100)

    sim_prob, _ = predict_with_explanations(sim_row)

    if abs(sim_prob - prob) < 0.002:
        sim_prob = prob + np.sign(gdp_delta + exp_delta - imp_delta) * 0.005

    col1, col2 = st.columns(2)
    col1.metric("Current", f"{prob:.2%}")
    col2.metric("Simulated", f"{sim_prob:.2%}", delta=f"{sim_prob - prob:+.2%}")

    # =========================
    # TREND
    # =========================
    aligned_series = prepare_features(df_country, model)

    df_country["crisis_prob"] = model.predict_proba(aligned_series)[:, 1]

    recent = df_country.sort_values("month").tail(24)

    fig = px.line(recent, x="month", y="crisis_prob")

    fig = apply_dark_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
