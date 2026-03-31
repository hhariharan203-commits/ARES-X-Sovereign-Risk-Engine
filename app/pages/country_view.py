from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (
    load_data,
    load_model,
    load_explainer,
    add_probabilities,
    predict_with_explanations,
    generate_executive_insights,
    align_features,
    apply_dark_theme,
    risk_label,
)

# =========================
# PREP FEATURES
# =========================
def prepare_features(df, model):
    aligned = align_features(df)
    aligned = aligned.fillna(0)

    if hasattr(model, "feature_names_in_"):
        aligned = aligned.reindex(columns=model.feature_names_in_, fill_value=0)

    return aligned.astype(float)


# =========================
# MAIN
# =========================
def main():
    st.title("Country Intelligence")

    try:
        df = load_data()

        if df.empty:
            st.warning("No data available")
            return

        df["month"] = pd.to_datetime(df["month"], errors="coerce")

        model = load_model()
        explainer = load_explainer()

        # ✅ SINGLE SOURCE OF TRUTH
        df = add_probabilities(df)

        countries = sorted(df["country"].dropna().unique())
        country = st.sidebar.selectbox("Select Country", countries)

        df_country = df[df["country"] == country]

        if df_country.empty:
            st.warning("No data for selected country")
            return

        latest = df_country.sort_values("month").tail(1)

        # =========================
        # CURRENT PREDICTION
        # =========================
        prob, insights = predict_with_explanations(latest)

        st.metric("Crisis Probability", f"{prob:.2%}")
        st.write(f"Risk Level: **{risk_label(prob)}**")

        st.markdown("**Top Drivers:**")
        for txt in insights:
            st.write(f"- {txt}")

        # =========================
        # EXECUTIVE INSIGHTS
        # =========================
        aligned = prepare_features(latest, model).iloc[[0]]

        shap_vals = explainer.shap_values(aligned)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        shap_vals = np.array(shap_vals)[0]

        exec_insights = generate_executive_insights(aligned, shap_vals)

        st.markdown("### Executive Insights")
        st.write(exec_insights["summary"])

        for d in exec_insights["drivers"]:
            st.write(d)

        # =========================
        # SIMULATOR
        # =========================
        st.markdown("---")
        st.subheader("Policy Simulator")

        col1, col2, col3 = st.columns(3)

        delta_rate = col1.slider("Interest Rate Change", -5.0, 5.0, 0.0)
        delta_infl = col2.slider("Inflation Change", -5.0, 5.0, 0.0)
        delta_unemp = col3.slider("Unemployment Change", -5.0, 5.0, 0.0)

        sim_row = latest.copy()

        if "interest_rate" in sim_row.columns:
            sim_row["interest_rate"] += delta_rate

        if "inflation" in sim_row.columns:
            sim_row["inflation"] += delta_infl

        if "unemployment" in sim_row.columns:
            sim_row["unemployment"] += delta_unemp

        sim_aligned = prepare_features(sim_row, model).iloc[[0]]

        sim_prob = float(model.predict_proba(sim_aligned)[0, 1])

        colA, colB = st.columns(2)

        colA.metric("Current", f"{prob:.2%}")
        colB.metric(
            "Simulated",
            f"{sim_prob:.2%}",
            delta=f"{sim_prob - prob:+.2%}",
        )

        st.write(f"Simulated Risk: **{risk_label(sim_prob)}**")

        # =========================
        # TREND CHART
        # =========================
        st.markdown("---")
        st.subheader("Economic Trend")

        plot_cols = [
            "gdp_growth",
            "inflation",
            "interest_rate",
            "unemployment",
        ]

        available = [c for c in plot_cols if c in df_country.columns]

        if "month" in df_country.columns and available:
            df_plot = df_country.sort_values("month")

            fig = px.line(
                df_plot,
                x="month",
                y=available,
                title=f"{country} Economic Indicators",
            )

            fig = apply_dark_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
