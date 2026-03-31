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
# PREPARE FEATURES
# =========================
def prepare_features(df, model):
    aligned = align_features(df)

    # ✅ safety
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

    df_plot = df_country.rename(columns={
        "gdp_current_usd": "GDP (USD)",
        "exports_pct_gdp": "Exports (% GDP)",
        "imports_pct_gdp": "Imports (% GDP)",
    })

    # ✅ FIX: remove empty rows
    df_plot = df_plot.dropna()

    if not df_plot.empty:
        fig = px.line(
            df_plot,
            x="month",
            y=["GDP (USD)", "Exports (% GDP)", "Imports (% GDP)"],
            title=f"Economic Indicators — {country}",
        )

        fig = apply_dark_theme(fig)
        fig.update_traces(line=dict(width=3))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No sufficient data for chart")

    # =========================
    # 🔮 LATEST PREDICTION
    # =========================
    st.subheader("Latest Crisis Prediction")

    if not df_country.empty:
        latest = df_country.sort_values("month").tail(1)

        # 🚨 CRITICAL FIX
        if latest.isnull().all(axis=1).values[0]:
            st.error("❌ No valid data available for prediction")
            return

        prob, insights = predict_with_explanations(latest)

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

        # =========================
        # 🎯 SCENARIO SIMULATOR
        # =========================
        st.markdown("### Scenario Simulator")

        # 🚨 warning if bad data
        if latest.isnull().sum().sum() > 0:
            st.warning("⚠️ Simulation may be less accurate due to missing data")

        st.markdown("#### Economic Growth")
        gdp_delta = st.slider("GDP change (%)", -20.0, 20.0, 0.0, 0.5)

        st.markdown("#### Trade Dynamics")
        imp_delta = st.slider("Imports change (%)", -20.0, 20.0, 0.0, 0.5)
        exp_delta = st.slider("Exports change (%)", -20.0, 20.0, 0.0, 0.5)

        st.markdown("#### Monetary Conditions")
        rate_delta = st.slider("Interest rate change (bps)", -300.0, 300.0, 0.0, 25.0)
        infl_delta = st.slider("Inflation change (pp)", -5.0, 5.0, 0.0, 0.25)

        sim_row = latest.copy()

        # ✅ safe transformations
        sim_row = sim_row.fillna(0)

        sim_row["gdp_current_usd"] *= (1 + gdp_delta / 100)
        sim_row["imports_pct_gdp"] *= (1 + imp_delta / 100)
        sim_row["exports_pct_gdp"] *= (1 + exp_delta / 100)

        if "interest_rate_pct" in sim_row.columns:
            sim_row["interest_rate_pct"] += rate_delta / 100

        if "inflation_cpi_pct" in sim_row.columns:
            sim_row["inflation_cpi_pct"] += infl_delta

        sim_prob, sim_insights = predict_with_explanations(sim_row)

        col1, col2 = st.columns(2)
        col1.metric("Current Probability", f"{prob:.2%}")
        col2.metric(
            "Simulated Probability",
            f"{sim_prob:.2%}",
            delta=f"{sim_prob - prob:+.2%}",
        )

        st.write(f"Simulated Risk Level: **{risk_label(sim_prob)}**")

        change = sim_prob - prob

        # ✅ better threshold
        if change > 0.05:
            st.error("Risk is significantly increasing under this scenario")
        elif change < -0.05:
            st.success("Risk is significantly decreasing under this scenario")
        else:
            st.info("Scenario has limited impact on risk")

        st.markdown("**Simulated Top Drivers:**")
        for txt in sim_insights:
            st.write(f"- {txt}")

        # =========================
        # 🧠 SIMULATED INSIGHTS
        # =========================
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

    else:
        st.info("No data available for this country.")


if __name__ == "__main__":
    main()
