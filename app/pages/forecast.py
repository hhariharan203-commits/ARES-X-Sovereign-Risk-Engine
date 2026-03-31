from __future__ import annotations

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (
    load_data,
    load_model,
    align_features,
    apply_dark_theme,
    risk_label,
)


# =========================
# FEATURE PREP
# =========================
def prepare_features(df, model):
    aligned = align_features(df).fillna(0)

    if hasattr(model, "feature_names_in_"):
        aligned = aligned.reindex(columns=model.feature_names_in_, fill_value=0)

    return aligned.astype(float)


# =========================
# IMPROVED FORECAST ENGINE
# =========================
def generate_forecast(df_country, model, steps=3):
    df_country = df_country.sort_values("month").copy()

    # Use last 3 periods for trend
    recent = df_country.tail(3).copy()

    latest = recent.tail(1).copy()
    forecasts = []

    current = latest.copy()

    for i in range(steps):
        # 🔥 TREND-BASED CHANGE (NOT RANDOM)
        for col in current.columns:
            if col not in ["month", "country"] and np.issubdtype(current[col].dtype, np.number):

                if col in recent.columns:
                    trend = recent[col].pct_change().mean()
                    trend = 0 if pd.isna(trend) else trend

                    # dampen to avoid explosion
                    trend = np.clip(trend, -0.05, 0.05)

                    current[col] = current[col] * (1 + trend)

        aligned = prepare_features(current, model)
        prob = float(model.predict_proba(aligned)[0, 1])

        forecasts.append({
            "month": latest["month"].values[0] + pd.DateOffset(months=i + 1),
            "prediction": prob
        })

    return pd.DataFrame(forecasts)


# =========================
# MAIN
# =========================
def main():
    st.title("Forecast — Next 3 Months")

    try:
        df = load_data()

        if df.empty:
            st.warning("No data available")
            return

        df["month"] = pd.to_datetime(df["month"], errors="coerce")

        model = load_model()

        if "country" not in df.columns:
            st.error("Country column missing")
            return

        countries = sorted(df["country"].dropna().unique())
        country = st.selectbox("Select Country", countries)

        df_country = df[df["country"] == country]

        if df_country.empty:
            st.warning("No data for selected country")
            return

        # =========================
        # CURRENT
        # =========================
        latest = df_country.sort_values("month").tail(1)

        aligned = prepare_features(latest, model)
        current_prob = float(model.predict_proba(aligned)[0, 1])

        col1, col2 = st.columns(2)
        col1.metric("Current Risk", f"{current_prob:.2%}")
        col2.metric("Risk Level", risk_label(current_prob))

        # =========================
        # HISTORICAL
        # =========================
        hist = df_country.sort_values("month").copy()
        hist["prediction"] = model.predict_proba(
            prepare_features(hist, model)
        )[:, 1]

        # =========================
        # FORECAST
        # =========================
        forecast_df = generate_forecast(df_country, model)

        # =========================
        # COMBINE
        # =========================
        hist_recent = hist.tail(12)

        combined = pd.concat([
            hist_recent.assign(type="Historical"),
            forecast_df.assign(type="Forecast")
        ])

        # =========================
        # CONFIDENCE BAND
        # =========================
        combined["upper"] = (combined["prediction"] + 0.05).clip(0, 1)
        combined["lower"] = (combined["prediction"] - 0.05).clip(0, 1)

        # =========================
        # CHART
        # =========================
        fig = px.line(
            combined,
            x="month",
            y="prediction",
            color="type",
            title=f"{country} Risk Forecast",
        )

        # confidence band
        fig.add_scatter(
            x=combined["month"],
            y=combined["upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        )

        fig.add_scatter(
            x=combined["month"],
            y=combined["lower"],
            mode="lines",
            fill="tonexty",
            line=dict(width=0),
            fillcolor="rgba(255,0,0,0.1)",
            showlegend=False,
        )

        fig.update_layout(yaxis_tickformat=".0%")
        fig = apply_dark_theme(fig)

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # FORECAST TABLE
        # =========================
        st.subheader("Forecast Data")
        st.dataframe(forecast_df, use_container_width=True)

        # =========================
        # EXECUTIVE INTERPRETATION
        # =========================
        change = forecast_df["prediction"].iloc[-1] - current_prob

        st.subheader("Outlook")

        if change > 0.03:
            st.error("Risk expected to rise significantly — potential instability ahead")
        elif change > 0.01:
            st.warning("Risk trending upward — monitor closely")
        elif change < -0.03:
            st.success("Risk expected to decline — improving conditions")
        elif change < -0.01:
            st.info("Moderate improvement expected")
        else:
            st.info("Risk expected to remain stable")

    except Exception as e:
        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
