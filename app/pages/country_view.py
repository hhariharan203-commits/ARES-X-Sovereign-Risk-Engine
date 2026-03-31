from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (
    add_probabilities,
    align_features,
    generate_executive_insights,
    load_data,
    load_model,
    predict_with_explanations,
    risk_label,
    load_explainer,
    apply_dark_theme,
)


def safe_load_dataframe():
    try:
        df = load_data()
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower()
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()


def prepare_features(df, model):
    aligned = align_features(df)
    full_df = load_data()
    aligned_global = align_features(full_df)
    num_cols = aligned.select_dtypes(include=[np.number]).columns
    aligned[num_cols] = aligned[num_cols].fillna(aligned_global[num_cols].mean())
    aligned = aligned.fillna(0)
    if hasattr(model, "feature_names_in_"):
        aligned = aligned.reindex(columns=model.feature_names_in_, fill_value=0)
    return aligned.astype(float)


def main():
    try:
        st.title("Country View")

        df = safe_load_dataframe()
        model = load_model()
        explainer = load_explainer()

        if "crisis_prob" not in df.columns:
            df = add_probabilities(df, model)

        if "country" not in df.columns:
            st.warning("Country column not available.")
            st.stop()

        countries = sorted(df["country"].dropna().unique())
        if not countries:
            st.warning("No countries available.")
            st.stop()

        country = st.sidebar.selectbox("Select Country", countries)
        df_country = df[df["country"] == country] if "country" in df.columns else pd.DataFrame()

        if df_country.empty:
            st.warning("No data for selected country")
            st.stop()

        # Current prediction
        latest = df_country.sort_values("month").tail(1).copy() if "month" in df_country.columns else df_country.tail(1).copy()
        try:
            prob, insights = predict_with_explanations(latest)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        st.metric("Predicted Crisis Probability", f"{prob:.2%}")
        st.write(f"Risk Level: **{risk_label(prob)}**")

        st.markdown("**Top Drivers (SHAP):**")
        for txt in insights:
            st.write(f"- {txt}")

    try:
        aligned = prepare_features(latest, model).iloc[[0]]
        shap_vals = explainer.shap_values(aligned)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_vals = np.array(shap_vals)
        if shap_vals.ndim > 1:
            shap_vals = shap_vals[0]
        shap_vals = shap_vals.flatten()
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

    # Policy Simulator
    st.markdown("---")
    st.subheader("Policy Simulator")
    colA, colB, colC = st.columns(3)
    delta_rate = colA.slider("Interest rate change (%)", -5.0, 5.0, 0.0, 0.25)
    delta_infl = colB.slider("Inflation change (%)", -5.0, 5.0, 0.0, 0.25)
    delta_unemp = colC.slider("Unemployment change (%)", -5.0, 5.0, 0.0, 0.25)

    sim_row = latest.copy()
    for col, delta in [
        ("interest_rate", delta_rate),
        ("inflation", delta_infl),
        ("unemployment", delta_unemp),
        ("unemployment_rate", delta_unemp),
    ]:
        if col in sim_row.columns:
            sim_row[col] = sim_row[col] + delta

        try:
            sim_aligned = prepare_features(sim_row, model).iloc[[0]]
            sim_prob = float(model.predict_proba(sim_aligned)[0, 1])
            st.metric("Simulated Probability", f"{sim_prob:.2%}")
            st.write(f"Simulated Risk Level: **{risk_label(sim_prob)}**")
        except Exception:
            st.warning("Simulator unavailable with current model.")

    required_cols = [
        "month",
        "gdp_growth",
        "inflation",
        "interest_rate",
        "unemployment_rate",
        "unemployment",
    ]
    cols = [c for c in required_cols if c in df_country.columns]
    if len(cols) <= 1:
        st.warning("Not enough economic indicators available")
        st.stop()

    df_plot = df_country[cols].copy()
    if "month" in df_plot.columns:
        df_plot = df_plot.sort_values("month")
    if df_plot.empty:
        st.warning("No data available for visualization")
        st.stop()

        for col in cols:
            if col == "month":
                continue
            fig = px.line(df_plot, x="month", y=col, title=f"{country} - {col}")
            fig = apply_dark_theme(fig)
            fig.update_traces(line=dict(width=2))
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Safe fallback: {e}")


if __name__ == "__main__":
    main()
