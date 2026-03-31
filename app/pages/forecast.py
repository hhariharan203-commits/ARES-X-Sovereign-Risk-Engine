from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import apply_dark_theme, assign_risk_levels


def load_forecast(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.warning("Forecast file not found. Please run the training pipeline.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to load forecast: {e}")
        return pd.DataFrame()
    df.columns = df.columns.str.strip().str.lower()
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
    else:
        df["month"] = pd.to_datetime(df.index, errors="coerce")
    return df


def main():
    try:
        st.title("Forecast")

        forecast_path = "data/forecast.csv"
        df = load_forecast(forecast_path)

        if df.empty:
            st.info("No forecast data available.")
            return

        required_cols = ["month", "prediction"]
        cols_present = [c for c in required_cols if c in df.columns]
        if len(cols_present) < 2 or "country" not in df.columns:
            st.warning("Forecast data incomplete")
            st.stop()

        countries = sorted(df["country"].dropna().unique())
        country = st.selectbox("Select Country", countries)

        df_country = df[df["country"] == country].sort_values("month")
        if df_country.empty:
            st.warning("No forecast data for this country.")
            return

        df_country["risk_level"] = assign_risk_levels(df_country["prediction"])

        fig = px.line(
            df_country,
            x="month",
            y="prediction",
            title=f"{country} - Forecast",
        )
        fig = apply_dark_theme(fig)
        fig.update_traces(line=dict(width=3))
        fig.update_layout(yaxis_title="Prediction")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df_country[["month", "prediction"]].reset_index(drop=True))
    except Exception as e:
        st.warning(f"Safe fallback: {e}")


if __name__ == "__main__":
    main()
