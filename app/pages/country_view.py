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
    load_feature_cols,
    apply_dark_theme,
)


def main():
    st.title("Country Intelligence")
    df = load_data()
    model = load_model()

    countries = sorted(df["country"].unique())
    country = st.sidebar.selectbox("Select Country", countries)

    df_country = df[df["country"] == country].copy()

    st.subheader(f"Economic Indicators — {country}")
    fig = px.line(
        df_country,
        x="month",
        y=["gdp_current_usd", "exports_pct_gdp", "imports_pct_gdp"],
        labels={"value": "Value", "variable": "Series"},
        title=f"Economic Indicators — {country}",
    )
    fig = apply_dark_theme(fig)
    fig.update_traces(line=dict(width=3))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latest Crisis Prediction")
    if not df_country.empty:
        latest = df_country.sort_values("month").tail(1)
        prob, insights = predict_with_explanations(latest)
        st.metric("Crisis Probability", f"{prob:.2%}", help="Predicted risk 3 months ahead")
        st.write(f"Risk Level: **{risk_label(prob)}**")

        st.markdown("**Top Drivers (SHAP):**")
        for txt in insights:
            st.write(f"- {txt}")

        # Executive Insights
        st.markdown("### Executive Insights")
        explainer = load_explainer()
        aligned = align_features(latest).iloc[[0]]
        shap_vals = explainer.shap_values(aligned)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_vals = np.array(shap_vals)
        if len(shap_vals.shape) > 1:
            shap_vals = shap_vals[0]
        shap_vals = shap_vals.flatten()
        exec_insights = generate_executive_insights(aligned, shap_vals, 0.0)
        st.write(exec_insights["summary"])
        st.markdown("**Key Drivers:**")
        for d in exec_insights["drivers"]:
            st.write(d)
        st.markdown("**Suggested Actions:**")
        for a in exec_insights["actions"]:
            st.write(a)

        st.markdown("### Scenario Simulator")
        gdp_delta = st.slider("GDP change (%)", -20.0, 20.0, 0.0, 0.5)
        imp_delta = st.slider("Imports change (%)", -20.0, 20.0, 0.0, 0.5)
        exp_delta = st.slider("Exports change (%)", -20.0, 20.0, 0.0, 0.5)
        rate_delta = st.slider("Interest rate change (bps)", -300.0, 300.0, 0.0, 25.0)
        infl_delta = st.slider("Inflation change (pp)", -5.0, 5.0, 0.0, 0.25)

        sim_row = latest.copy()
        sim_row["gdp_current_usd"] *= (1 + gdp_delta / 100)
        sim_row["imports_pct_gdp"] *= (1 + imp_delta / 100)
        sim_row["exports_pct_gdp"] *= (1 + exp_delta / 100)
        if "interest_rate_pct" in sim_row.columns:
            sim_row["interest_rate_pct"] += rate_delta / 100
        if "inflation_cpi_pct" in sim_row.columns:
            sim_row["inflation_cpi_pct"] += infl_delta

        sim_prob, sim_insights = predict_with_explanations(sim_row)
        col_a, col_b = st.columns(2)
        col_a.metric("Current Probability", f"{prob:.2%}")
        col_b.metric("Simulated Probability", f"{sim_prob:.2%}", delta=f"{sim_prob - prob:+.2%}")
        st.write(f"Simulated Risk Level: **{risk_label(sim_prob)}**")
        st.markdown("**Simulated Top Drivers:**")
        for txt in sim_insights:
            st.write(f"- {txt}")

        # Executive insights for simulation
        sim_aligned = align_features(sim_row).iloc[[0]]
        sim_shap_vals = explainer.shap_values(sim_aligned)
        if isinstance(sim_shap_vals, list):
            sim_shap_vals = sim_shap_vals[1]
        sim_shap_vals = np.array(sim_shap_vals)
        if len(sim_shap_vals.shape) > 1:
            sim_shap_vals = sim_shap_vals[0]
        sim_shap_vals = sim_shap_vals.flatten()
        sim_exec = generate_executive_insights(sim_aligned, sim_shap_vals, 0.0)
        st.markdown("**Simulated Executive Insights:**")
        st.write(sim_exec["summary"])
        st.markdown("**Simulated Suggested Actions:**")
        for a in sim_exec["actions"]:
            st.write(a)
    else:
        st.info("No data available for this country.")


if __name__ == "__main__":
    main()
