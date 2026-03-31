import pandas as pd
import plotly.express as px
import streamlit as st

from utils import apply_dark_theme, load_data, add_probabilities, assign_risk_levels


def safe_load():
    try:
        df = load_data()
        df.columns = df.columns.str.strip().str.lower()
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()


def main():
    try:
        st.title("Global Risk Overview")

        df = safe_load()
        if df.empty:
            st.warning("No data available")
            return

        if "month" in df.columns:
            df["month"] = pd.to_datetime(df["month"], errors="coerce")
        else:
            st.warning("Month column missing")
            return

        if "crisis_prob" not in df.columns:
            df = add_probabilities(df)

        latest_month = df["month"].max()
        latest_df = df[df["month"] == latest_month]

        if latest_df.empty:
            st.warning("No recent data available")
            return

        if "risk_level" not in df.columns and "crisis_prob" in df.columns:
            df["risk_level"] = assign_risk_levels(df["crisis_prob"])
        if "risk_level" not in latest_df.columns and "crisis_prob" in latest_df.columns:
            latest_df["risk_level"] = assign_risk_levels(latest_df["crisis_prob"])

        if {"country", "crisis_prob"} <= set(latest_df.columns):
            table = latest_df[["country", "crisis_prob", "risk_level"]].sort_values("crisis_prob", ascending=False)
            st.subheader("Latest Risk by Country")
            st.dataframe(table)
        else:
            st.warning("Risk data not available")

        if "risk_level" in df.columns:
            risk_counts = df["risk_level"].value_counts()
            if not risk_counts.empty:
                fig = px.pie(
                    names=risk_counts.index,
                    values=risk_counts.values,
                    title="Risk Distribution (All Time)"
                )
                fig = apply_dark_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Risk level data not available")
        else:
            st.warning("Risk level column missing")

        if "crisis_prob" in latest_df.columns and "country" in latest_df.columns:
            top5 = latest_df.sort_values("crisis_prob", ascending=False).head(5)
            st.subheader("Top 5 Highest Risk Countries")
            st.table(top5[["country", "crisis_prob"]])

        if {"country", "crisis_prob"} <= set(latest_df.columns):
            fig = px.choropleth(
                latest_df,
                locations="country",
                locationmode="ISO-3",
                color="crisis_prob",
                hover_name="country",
                color_continuous_scale=[
                    [0, "green"],
                    [0.5, "yellow"],
                    [1, "red"]
                ],
                title="Global Crisis Probability (Latest Month)"
            )
            fig.update_layout(
                geo=dict(
                    bgcolor="#0e1117",
                    lakecolor="#0e1117",
                    landcolor="#1a1a1a"
                )
            )
            fig = apply_dark_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient data for map view")
    except Exception as e:
        st.warning(f"Safe fallback: {e}")


if __name__ == "__main__":
    main()
