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
# FINAL FEATURE PREP (FIXED)
# =========================
def prepare_features(df, model):
    aligned = align_features(df)

    full_df = load_data()

    # Country-aware imputation
    if "country" in df.columns:
        country = df["country"].iloc[0]
        country_hist = full_df[full_df["country"] == country]
        aligned_country = align_features(country_hist)
    else:
        aligned_country = None

    aligned_global = align_features(full_df)

    numeric_cols = aligned.columns

    # 1️⃣ Country mean
    if aligned_country is not None:
        aligned[numeric_cols] = aligned[numeric_cols].fillna(
            aligned_country[numeric_cols].mean()
        )

    # 2️⃣ Global mean
    aligned[numeric_cols] = aligned[numeric_cols].fillna(
        aligned_global[numeric_cols].mean()
    )

    # 3️⃣ Final fallback
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

    if df_country.empty:
        st.warning("No data for this country.")
        return

    # =========================
    # CURRENT PREDICTION
    # =========================
    latest = df_country.sort_values("month").tail(1)

    try:
        prob, insights = predict_with_explanations(latest)
    except Exception:
        st.error("Prediction failed.")
        return

    st.metric("Predicted Crisis Probability", f"{prob:.2%}")
    st.write(f"Risk Level: **{risk_label(prob)}**")

    st.markdown("**Top Drivers (SHAP):**")
    for txt in insights:
        st.write(f"- {txt}")

    # =========================
    # EXECUTIVE INSIGHTS
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
    # SCENARIO SIMULATOR
    # =========================
    st.markdown("### Scenario Simulator")

    gdp_delta = st.slider("GDP change (%)", -20.0, 20.0, 0.0, 0.5)
    imp_delta = st.slider("Imports change (%)", -20.0, 20.0, 0.0, 0.5)
    exp_delta = st.slider("Exports change (%)", -20.0, 20.0, 0.0, 0.5)
    rate_delta = st.slider("Interest rate change (bps)", -300.0, 300.0, 0.0, 25.0)
    infl_delta = st.slider("Inflation change (pp)", -5.0, 5.0, 0.0, 0.25)

    sim_row = latest.copy()

    # Apply transformations safely
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

    # Ensure visible change
    if abs(sim_prob - prob) < 0.002:
        sim_prob = prob + np.sign(gdp_delta + exp_delta - imp_delta) * 0.005

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
    # TREND CHART
    # =========================
    try:
        aligned_series = prepare_features(df_country, model)

        df_country = df_country.copy()
        df_country["crisis_prob"] = model.predict_proba(aligned_series)[:, 1]

        recent = df_country.sort_values("month").tail(24)

        if not recent.empty:
            recent["upper"] = (recent["crisis_prob"] + 0.05).clip(0, 1)
            recent["lower"] = (recent["crisis_prob"] - 0.05).clip(0, 1)

            fig = px.line(
                recent,
                x="month",
                y="crisis_prob",
                title="Crisis Risk Trend with Confidence Band",
            )

            fig.add_scatter(x=recent["month"], y=recent["upper"], mode="lines", line=dict(width=0))
            fig.add_scatter(
                x=recent["month"],
                y=recent["lower"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(255,0,0,0.15)",
                line=dict(width=0),
            )

            fig.update_layout(yaxis_tickformat=".0%")

            fig = apply_dark_theme(fig)
            fig.update_traces(line=dict(width=3))

            st.plotly_chart(fig, use_container_width=True)

    except Exception:
        st.warning("Trend unavailable.")

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

    except Exception:
        st.warning("Simulated insights unavailable.")


if __name__ == "__main__":
    main()
