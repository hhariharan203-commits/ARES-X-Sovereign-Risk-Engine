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

    aligned_global = align_features(full_df)

    # ✅ numeric-only mean fill (fixes TypeError)
    num_cols = aligned.select_dtypes(include=[np.number]).columns
    aligned[num_cols] = aligned[num_cols].fillna(
        aligned_global[num_cols].mean()
    )
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
    # ECONOMIC INDICATORS CHART
    # =========================
    st.subheader(f"Economic Indicators — {country}")

    cols = ["month", "gdp_current_usd", "exports_pct_gdp", "imports_pct_gdp"]
    df_plot = df_country[cols].copy().sort_values("month")

    try:
        missing_ratio = df_plot[cols[1:]].isna().mean().mean()

        if missing_ratio > 0.6:
            full_df = load_data()
            df_plot = (
                full_df[cols]
                .groupby("month")
                .mean(numeric_only=True)
                .reset_index()
            )

        df_plot[cols[1:]] = (
            df_plot[cols[1:]]
            .interpolate()
            .ffill()
            .bfill()
            .fillna(0)
        )

        df_plot = df_plot.rename(columns={
            "gdp_current_usd": "GDP (USD)",
            "exports_pct_gdp": "Exports (% GDP)",
            "imports_pct_gdp": "Imports (% GDP)",
        })

        if df_plot[["GDP (USD)", "Exports (% GDP)", "Imports (% GDP)"]].sum().sum() == 0:
            st.warning("No sufficient data for chart")
        else:
            fig = px.line(
                df_plot,
                x="month",
                y=["GDP (USD)", "Exports (% GDP)", "Imports (% GDP)"],
            )
            fig = apply_dark_theme(fig)
            fig.update_traces(line=dict(width=2))
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Chart unavailable: {e}")

    # =========================
    # LATEST PREDICTION
    # =========================
    st.subheader("Latest Crisis Prediction")

    if df_country.empty:
        st.info("No data available for this country.")
        return

    latest = df_country.sort_values("month").tail(1).copy()

    try:
        prob, insights = predict_with_explanations(latest)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
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

        shap_vals = np.array(shap_vals).flatten()

        exec_insights = generate_executive_insights(aligned, shap_vals)

        st.markdown("### Executive Insights")
        st.write(exec_insights["summary"])

        st.markdown("**Key Drivers:**")
        for d in exec_insights["drivers"]:
            st.write(d)

        st.markdown("**Suggested Actions:**")
        for a in exec_insights["actions"]:
            st.write(a)

    except Exception as e:
        st.warning(f"Executive insights unavailable: {e}")

    # =========================
    # SCENARIO SIMULATOR
    # =========================
    st.markdown("### Scenario Simulator")

    gdp_delta  = st.slider("GDP change (%)",           -20.0, 20.0,   0.0, 0.5)
    imp_delta  = st.slider("Imports change (%)",        -20.0, 20.0,   0.0, 0.5)
    exp_delta  = st.slider("Exports change (%)",        -20.0, 20.0,   0.0, 0.5)
    rate_delta = st.slider("Interest rate change (bps)",-300.0,300.0,  0.0, 25.0)
    infl_delta = st.slider("Inflation change (pp)",      -5.0,  5.0,   0.0, 0.25)

    sim_row = latest.copy()

    # ✅ numeric-only fill on sim_row
    num_cols = sim_row.select_dtypes(include=[np.number]).columns
    sim_row[num_cols] = sim_row[num_cols].fillna(sim_row[num_cols].mean())
    sim_row[num_cols] = sim_row[num_cols].fillna(0)

    if "gdp_current_usd"   in sim_row.columns: sim_row["gdp_current_usd"]   *= (1 + gdp_delta  / 100)
    if "imports_pct_gdp"   in sim_row.columns: sim_row["imports_pct_gdp"]   *= (1 + imp_delta  / 100)
    if "exports_pct_gdp"   in sim_row.columns: sim_row["exports_pct_gdp"]   *= (1 + exp_delta  / 100)
    if "interest_rate_pct" in sim_row.columns: sim_row["interest_rate_pct"] += rate_delta / 100
    if "inflation_cpi_pct" in sim_row.columns: sim_row["inflation_cpi_pct"] += infl_delta

    try:
        sim_prob, sim_insights = predict_with_explanations(sim_row)
    except Exception as e:
        st.error(f"Simulation failed: {e}")
        return

    col1, col2 = st.columns(2)
    col1.metric("Current Probability",   f"{prob:.2%}")
    col2.metric("Simulated Probability", f"{sim_prob:.2%}",
                delta=f"{sim_prob - prob:+.2%}")

    st.write(f"Simulated Risk Level: **{risk_label(sim_prob)}**")

    change = sim_prob - prob
    if change > 0.05:
        st.error("Risk significantly increasing")
    elif change < -0.05:
        st.success("Risk significantly decreasing")
    else:
        st.info("Limited impact scenario")

    st.markdown("**Simulated Top Drivers:**")
    for txt in sim_insights:
        st.write(f"- {txt}")

    # =========================
    # SIMULATED EXECUTIVE INSIGHTS
    # =========================
    try:
        sim_aligned = prepare_features(sim_row, model).iloc[[0]]

        sim_shap_vals = explainer.shap_values(sim_aligned)
        if isinstance(sim_shap_vals, list):
            sim_shap_vals = sim_shap_vals[1]

        sim_shap_vals = np.array(sim_shap_vals).flatten()

        sim_exec = generate_executive_insights(sim_aligned, sim_shap_vals)

        st.markdown("**Simulated Executive Insights:**")
        st.write(sim_exec["summary"])

        st.markdown("**Simulated Suggested Actions:**")
        for a in sim_exec["actions"]:
            st.write(a)

    except Exception as e:
        st.warning(f"Simulated insights unavailable: {e}")


if __name__ == "__main__":
    main()
