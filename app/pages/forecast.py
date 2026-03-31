from __future__ import annotations

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
# SIMPLE FORECAST ENGINE
# =========================
def generate_forecast(df_country, model, steps=3):
    df_country = df_country.sort_values("month").copy()

    latest = df_country.tail(1).copy()
    forecasts = []

    current = latest.copy()

    for i in range(steps):
        # simple forward simulation (trend continuation)
        for col in current.columns:
            if current[col].dtype != "object" and col not in ["month"]:
                current[col] = current[col] * (1 + np.random.uniform(-0.02, 0.02))

        aligned = prepare_features(current, model)
        prob = float(model.predict_proba(aligned)[0, 1])

        forecasts.append({
            "month": latest["month"].values[0] + pd.DateOffset(months=i+1),
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

        countries = sorted(df["country"].dropna().unique())
        country = st.selectbox("Select Country", countries)

        df_country = df[df["country"] == country]

        if df_country.empty:
            st.warning("No data for selected country")
            return

        # =========================
        # CURRENT PROBABILITY
        # =========================
        latest = df_country.sort_values("month").tail(1)

        aligned = prepare_features(latest, model)
        current_prob = float(model.predict_proba(aligned)[0, 1])

        st.metric("Current Risk", f"{current_prob:.2%}")
        st.write(f"Risk Level: **{risk_label(current_prob)}**")

        # =========================
        # FORECAST
        # =========================
        forecast_df = generate_forecast(df_country, model)

        # combine
        hist = df_country.sort_values("month")[["month"]].copy()
        hist["prediction"] = model.predict_proba(
            prepare_features(df_country, model)
        )[:, 1]

        combined = pd.concat([hist.tail(12), forecast_df])

        # =========================
        # CHART
        # =========================
        fig = px.line(
            combined,
            x="month",
            y="prediction",
            title=f"{country} Risk Forecast",
        )

        fig = apply_dark_theme(fig)
        fig.update_layout(yaxis_tickformat=".0%")

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # TABLE
        # =========================
        st.subheader("Forecast Data")
        st.dataframe(forecast_df, use_container_width=True)

        # =========================
        # INTERPRETATION
        # =========================
        change = forecast_df["prediction"].iloc[-1] - current_prob

        if change > 0.02:
            st.error("Risk expected to increase in coming months")
        elif change < -0.02:
            st.success("Risk expected to decrease")
        else:
            st.info("Risk expected to remain stable")

    except Exception as e:
        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
