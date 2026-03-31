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

    # Country-level fallback
    if "country" in df.columns:
        country = df["country"].iloc[0]
        country_hist = full_df[full_df["country"] == country]
        aligned_country = align_features(country_hist)
    else:
        aligned_country = None

    aligned_global = align_features(full_df)

    # Fill missing
    if aligned_country is not None and not aligned_country.empty:
        aligned = aligned.fillna(aligned_country.mean())

    aligned = aligned.fillna(aligned_global.mean())
    aligned = aligned.fillna(0)

    if hasattr(model, "feature_names_in_"):
        aligned = aligned.reindex(columns=model.feature_names_in_, fill_value=0)

    return aligned.astype(float)


# =========================
# MAIN
# =========================
def main():
    st.title("Country Intelligence")

    df = load_data()
    model = load_model()
    explainer = load_explainer()

    countries = sorted(df["country"].unique())
    country = st.sidebar.selectbox("Select Country", countries)

    df_country = df[df["country"] == country].copy()

    # =========================
    # CHART (ALWAYS WORKS)
    # =========================
    st.subheader(f"Economic Indicators — {country}")

    cols = ["month", "gdp_current_usd", "exports_pct_gdp", "imports_pct_gdp"]
    df_plot = df_country[cols].copy().sort_values("month")

    try:
        # If too much missing → fallback to global
        if df_plot[cols[1:]].isna().mean().mean() > 0.6:
            full_df = load_data()
            df_plot = full_df[cols].copy()
            df_plot = df_plot.groupby("month").mean(numeric_only=True).reset_index()

        df_plot[cols[1:]] = (
            df_plot[cols[1:]]
            .interpolate()
            .ffill()
            .bfill()
        )

        df_plot = df_plot.fillna(0)

        df_plot = df_plot.rename(columns={
            "gdp_current_usd": "GDP (USD)",
            "exports_pct_gdp": "Exports (% GDP)",
            "imports_pct_gdp": "Imports (% GDP)",
        })

        fig = px.line(df_plot, x="month",
                      y=["GDP (USD)", "Exports (% GDP)", "Imports (% GDP)"])

        fig = apply_dark_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    except Exception:
        st.warning("Chart unavailable")

    # =========================
    # PREDICTION
    # =========================
    latest = df_country.sort_values("month").tail(1)

    try:
        prob, insights = predict_with_explanations(latest)
    except Exception:
        st.error("Prediction failed")
        return

    st.metric("Crisis Probability", f"{prob:.2%}")
    st.write(f"Risk Level: **{risk_label(prob)}**")

    st.markdown("**Top Drivers (SHAP):**")
    for txt in insights:
        st.write(f"- {txt}")

    # =========================
    # EXECUTIVE INSIGHTS
    # =========================
    try:
        aligned = prepare_features(latest, model).iloc[[0]]

        shap_vals = explainer.shap_values(aligned)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        exec_insights = generate_executive_insights(
            aligned,
            np.array(shap_vals).flatten()
        )

        st.markdown("### Executive Insights")
        st.write(exec_insights["summary"])

        for d in exec_insights["drivers"]:
            st.write(d)

        for a in exec_insights["actions"]:
            st.write(a)

    except Exception:
        st.warning("Insights unavailable")


if __name__ == "__main__":
    main()
