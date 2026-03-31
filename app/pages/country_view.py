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
# SAFE FEATURE PREP (FINAL)
# =========================
def prepare_features(df, model):
    aligned = align_features(df)

    # ✅ ONLY NUMERIC MEAN
    numeric_cols = aligned.select_dtypes(include=[np.number]).columns
    aligned[numeric_cols] = aligned[numeric_cols].fillna(
        aligned[numeric_cols].mean()
    )

    # ✅ FALLBACK
    aligned = aligned.fillna(0)

    if hasattr(model, "feature_names_in_"):
        aligned = aligned.reindex(columns=model.feature_names_in_, fill_value=0)

    return aligned.astype(float)


# =========================
# MAIN APP
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
    # 📊 ECONOMIC INDICATORS
    # =========================
    st.subheader(f"Economic Indicators — {country}")

    cols = ["month", "gdp_current_usd", "exports_pct_gdp", "imports_pct_gdp"]
    df_plot = df_country[cols].copy()

    df_plot = df_plot.dropna(
        subset=["gdp_current_usd", "exports_pct_gdp", "imports_pct_gdp"]
    )

    if df_plot.empty or len(df_plot) < 3:
        st.warning("No sufficient data for chart")
    else:
        df_plot = df_plot.rename(columns={
            "gdp_current_usd": "GDP (USD)",
            "exports_pct_gdp": "Exports (% GDP)",
            "imports_pct_gdp": "Imports (% GDP)",
        })

        fig = px.line(
            df_plot,
            x="month",
            y=["GDP (USD)", "Exports (% GDP)", "Imports (% GDP)"],
        )

        fig = apply_dark_theme(fig)
        fig.update_traces(line=dict(width=3))
        st.plotly_chart(fig, use_container_width=True)

    # =========================
    # 🔮 LATEST PREDICTION
    # =========================
    st.subheader("Latest Crisis Prediction")

    if df_country.empty:
        st.info("No data available for this country.")
        return

    latest = df_country.sort_values("month").tail(1)

    # ✅ NUMERIC SAFE CLEAN
    numeric_cols = latest.select_dtypes(include=[np.number]).columns
    latest[numeric_cols] = latest[numeric_cols].fillna(
        latest[numeric_cols].mean()
    )

    latest = latest.fillna(0)

    try:
        prob, insights = predict_with_explanations(latest)
    except Exception:
        st.error("Prediction failed due to data issues.")
        return

    st.metric("Crisis Probability", f"{prob:.2%}")
    st.write(f"Risk Level: **{risk_label(prob)}**")

    st.markdown("**Top Drivers (SHAP):**")
    for txt in insights:
        st.write(f"- {txt}")

    # =========================
    # 🧠 EXECUTIVE INSIGHTS
    # =========================
    st.markdown("### Executive Insights")

    aligned = prepare_features(latest, model).iloc[[0]]

    try:
        shap_vals = explainer.shap_values(aligned)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        shap_vals = np.array(shap_vals).flatten()

        exec_insights = generate_executive_insights(aligned, shap_vals)

        st.write(exec_insights["summary"])

        st.markdown("**Key Drivers:**")
        for d in exec_insights["drivers"]:
            st.write(d)

        st.markdown("**Suggested Actions:**")
        for a in exec_insights["actions"]:
            st.write(a)

    except Exception:
        st.warning("Executive insights unavailable.")

    # =========================
    # 🎯 SCENARIO SIMULATOR
    # =========================
    st.markdown("### Scenario Simulator")

    gdp_delta = st.slider("GDP change (%)", -20.0, 20.0, 0.0, 0.5)
    imp_delta = st.slider("Imports change (%)", -20.0, 20.0, 0.0, 0.5)
    exp_delta = st.slider("Exports change (%)", -20.0, 20.0, 0.0, 0.5)
    rate_delta = st.slider("Interest rate change (bps)", -300.0, 300.0, 0.0, 25.0)
    infl_delta = st.slider("Inflation change (pp)", -5.0, 5.0, 0.0, 0.25)

    sim_row = latest.copy()

    # ✅ NUMERIC SAFE CLEAN
    numeric_cols = sim_row.select_dtypes(include=[np.number]).columns
    sim_row[numeric_cols] = sim_row[numeric_cols].fillna(
        sim_row[numeric_cols].mean()
    )

    sim_row = sim_row.fillna(0)

    # Safe transformations
    if "gdp_current_usd" in sim_row.columns:
        sim_row["gdp_current_usd"] *= (1 + gdp_delta / 100)

    if "imports_pct_gdp" in sim_row.columns:
        sim_row["imports_pct_gdp"] *= (1 + imp_delta / 100)

    if "exports_pct_gdp" in sim_row.columns:
        sim_row["exports_pct_gdp"] *= (1 + exp_delta / 100)

    if "interest_rate_pct" in sim_row.columns:
        sim_row["interest_rate_pct"] += rate_delta / 100

    if "inflation_cpi_pct" in sim_row.columns:
        sim_row["inflation_cpi_pct"] += infl_delta

    try:
        sim_prob, sim_insights = predict_with_explanations(sim_row)
    except Exception:
        st.error("Simulation failed.")
        return

    col1, col2 = st.columns(2)
    col1.metric("Current Probability", f"{prob:.2%}")
    col2.metric(
        "Simulated Probability",
        f"{sim_prob:.2%}",
        delta=f"{sim_prob - prob:+.2%}",
    )

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
    # 🧠 SIMULATED INSIGHTS
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

    except Exception:
        st.warning("Simulated insights unavailable.")


if __name__ == "__main__":
    main()
