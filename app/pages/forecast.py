from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import apply_dark_theme


def load_forecast(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.warning("Forecast file not found. Please run the training pipeline.")
        return pd.DataFrame()
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"])
    return df


def main():
    st.title("Forecast")

    forecast_path = "data/forecast.csv"
    df = load_forecast(forecast_path)

    if df.empty:
        st.info("No forecast data available.")
        return

    if "country" not in df.columns or "month" not in df.columns or "prediction" not in df.columns:
        st.error("Forecast data missing required columns (country, month, prediction).")
        return

    countries = sorted(df["country"].dropna().unique())
    country = st.selectbox("Select Country", countries)

    df_country = df[df["country"] == country].sort_values("month")
    if df_country.empty:
        st.warning("No forecast data for this country.")
        return

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


if __name__ == "__main__":
    main()
