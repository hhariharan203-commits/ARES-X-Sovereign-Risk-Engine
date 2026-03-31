import pandas as pd
import plotly.express as px
import streamlit as st

from utils import apply_dark_theme, load_data, add_probabilities


def main():
    st.title("Global Risk Overview")

    try:
        df = load_data()

        if df.empty:
            st.warning("No data available")
            return

        # Ensure datetime
        df["month"] = pd.to_datetime(df["month"], errors="coerce")

        # ALWAYS recompute predictions (single source of truth)
        df = add_probabilities(df)

        # Latest snapshot
        latest_month = df["month"].max()
        latest_df = df[df["month"] == latest_month].copy()

        if latest_df.empty:
            st.warning("No latest data available")
            return

        # =========================
        # TABLE
        # =========================
        st.subheader("Latest Risk by Country")

        table = latest_df[
            ["country", "crisis_prob", "risk_level"]
        ].sort_values("crisis_prob", ascending=False)

        st.dataframe(
            table.style.format({"crisis_prob": "{:.2%}"}),
            use_container_width=True,
        )

        # =========================
        # RISK DISTRIBUTION
        # =========================
        st.subheader("Risk Distribution")

        risk_counts = latest_df["risk_level"].value_counts()

        fig_pie = px.pie(
            names=risk_counts.index,
            values=risk_counts.values,
            title="Risk Distribution (Latest Month)",
        )

        fig_pie = apply_dark_theme(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)

        # =========================
        # TOP 5 COUNTRIES
        # =========================
        st.subheader("Top 5 Highest Risk Countries")

        top5 = latest_df.sort_values("crisis_prob", ascending=False).head(5)

        fig_bar = px.bar(
            top5,
            x="crisis_prob",
            y="country",
            orientation="h",
            text=top5["crisis_prob"].map(lambda x: f"{x:.1%}"),
            title="Top Risk Countries",
        )

        fig_bar = apply_dark_theme(fig_bar)
        fig_bar.update_layout(xaxis_tickformat=".0%")

        st.plotly_chart(fig_bar, use_container_width=True)

        # =========================
        # GLOBAL MAP
        # =========================
        st.subheader("Global Risk Map")

        fig_map = px.choropleth(
            latest_df,
            locations="country",
            locationmode="ISO-3",
            color="crisis_prob",
            hover_name="country",
            color_continuous_scale=[
                [0, "green"],
                [0.5, "yellow"],
                [1, "red"],
            ],
            title="Global Crisis Probability",
        )

        fig_map.update_layout(
            geo=dict(bgcolor="#0e1117"),
            coloraxis_showscale=False,
        )

        fig_map = apply_dark_theme(fig_map)

        st.plotly_chart(fig_map, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
