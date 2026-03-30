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
    st.title("Forecast (Next 3 Months)")
    df = load_data()
    model = load_model()

    countries = sorted(df["country"].unique())
    country = st.sidebar.selectbox("Select Country", countries, key="forecast_country")

    df_country = df[df["country"] == country].copy()

    if df_country.empty:
        st.info("No data for this country.")
        return

    latest = df_country.sort_values("month").tail(1)
    prob, insights = predict_with_explanations(latest)
    expected = float(prob)
    st.metric("Predicted Crisis Probability (3 months ahead)", f"{prob:.2%}")
    st.write(f"Risk Level: **{risk_label(prob)}**")
    st.markdown("**Top Drivers (SHAP):**")
    for txt in insights:
        st.write(f"- {txt}")

    # Executive Insights
    explainer = load_explainer()
    aligned = align_features(latest).iloc[[0]]
    shap_vals = explainer.shap_values(aligned)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_vals = np.array(shap_vals)
    if len(shap_vals.shape) > 1:
        shap_vals = shap_vals[0]
    shap_vals = shap_vals.flatten()
    feature_cols = list(load_feature_cols())
    exec_insights = generate_executive_insights(aligned, shap_vals, expected)
    st.markdown("### Executive Insights")
    st.write(exec_insights["summary"])
    st.markdown("**Key Drivers:**")
    for d in exec_insights["drivers"]:
        st.write(d)
    st.markdown("**Suggested Actions:**")
    for a in exec_insights["actions"]:
        st.write(a)

    st.markdown("### Scenario Simulator")
    gdp_delta = st.slider("GDP change (%)", -20.0, 20.0, 0.0, 0.5, key="sim_gdp_forecast")
    imp_delta = st.slider("Imports change (%)", -20.0, 20.0, 0.0, 0.5, key="sim_imp_forecast")
    exp_delta = st.slider("Exports change (%)", -20.0, 20.0, 0.0, 0.5, key="sim_exp_forecast")
    rate_delta = st.slider("Interest rate change (bps)", -300.0, 300.0, 0.0, 25.0, key="sim_rate_forecast")
    infl_delta = st.slider("Inflation change (pp)", -5.0, 5.0, 0.0, 0.25, key="sim_infl_forecast")

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

    # Trend with simple confidence band
    aligned_series = align_features(df_country)
    df_country = df_country.copy()
    df_country["crisis_prob"] = load_model().predict_proba(aligned_series)[:, 1]
    recent = df_country.sort_values("month").tail(24)
    if not recent.empty:
        recent["upper"] = (recent["crisis_prob"] + 0.05).clip(0, 1)
        recent["lower"] = (recent["crisis_prob"] - 0.05).clip(0, 1)
        line_fig = px.line(
            recent,
            x="month",
            y="crisis_prob",
            title="Predicted Trend (with confidence band)",
            labels={"crisis_prob": "Crisis Probability"},
        )
        line_fig.add_scatter(x=recent["month"], y=recent["upper"], mode="lines", line=dict(width=0), showlegend=False)
        line_fig.add_scatter(
            x=recent["month"],
            y=recent["lower"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(255,0,0,0.15)",
            showlegend=False,
        )
        line_fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Crisis Probability",
            yaxis_tickformat=".0%",
        )
        line_fig = apply_dark_theme(line_fig)
        line_fig.update_traces(line=dict(width=3))
        st.plotly_chart(line_fig, use_container_width=True)

    # Executive insights for simulation
    sim_aligned = align_features(sim_row).iloc[[0]]
    sim_shap_vals = explainer.shap_values(sim_aligned)
    if isinstance(sim_shap_vals, list):
        sim_shap_vals = sim_shap_vals[1]
    sim_shap_vals = np.array(sim_shap_vals)
    if len(sim_shap_vals.shape) > 1:
        sim_shap_vals = sim_shap_vals[0]
    sim_shap_vals = sim_shap_vals.flatten()
    sim_exec = generate_executive_insights(sim_aligned, sim_shap_vals, expected)
    st.markdown("**Simulated Executive Insights:**")
    st.write(sim_exec["summary"])
    st.markdown("**Simulated Suggested Actions:**")
    for a in sim_exec["actions"]:
        st.write(a)


if __name__ == "__main__":
    main()
