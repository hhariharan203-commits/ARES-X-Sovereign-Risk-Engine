import sys
import os

CURRENT_DIR = os.path.dirname(__file__)
APP_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import shap

from utils import (
    load_data,
    load_model,
    add_probabilities,
    apply_dark_theme,
    humanize_feature,
    align_features,
)


# =========================
# SAFE SHAP DRIVERS
# =========================
def top_shap_drivers(row, model, top_n=2):
    try:
        aligned = align_features(row).iloc[[0]].fillna(0)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(aligned)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_values = np.array(shap_values)[0]

        cols = list(aligned.columns)
        top_idx = np.argsort(np.abs(shap_values))[::-1][:top_n]

        drivers = []
        for i in top_idx:
            feat = cols[i]
            direction = "↑" if shap_values[i] > 0 else "↓"
            drivers.append(f"{humanize_feature(feat)} {direction}")

        return drivers if drivers else ["Drivers unavailable"]

    except Exception:
        return ["Drivers unavailable"]


# =========================
# MAIN
# =========================
def main():
    st.title("Executive Dashboard")

    try:
        df = load_data()

        if df.empty:
            st.warning("No data available")
            return

        df["month"] = pd.to_datetime(df["month"], errors="coerce")

        model = load_model()

        # =========================
        # PREDICTIONS
        # =========================
        df = add_probabilities(df)

        latest = (
            df.sort_values("month")
            .groupby("country")
            .tail(1)
            .sort_values("crisis_prob", ascending=False)
        )

        if latest.empty:
            st.warning("No latest data")
            return

        # =========================
        # KPI SECTION
        # =========================
        total = latest["country"].nunique()
        high = (latest["risk_level"] == "HIGH").sum()
        avg = latest["crisis_prob"].mean()
        top = latest.iloc[0]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Countries Covered", total)
        c2.metric("High Risk Economies", high)
        c3.metric("Average Risk", f"{avg:.2%}")
        c4.metric("Highest Risk", top["country"], f"{top['crisis_prob']:.1%}")

        st.markdown("---")

        # =========================
        # DISTRIBUTION
        # =========================
        st.subheader("Risk Distribution")

        dist = (
            latest["risk_level"]
            .value_counts()
            .reindex(["LOW", "MEDIUM", "HIGH"], fill_value=0)
            .reset_index()
        )
        dist.columns = ["risk", "count"]

        fig = px.bar(dist, x="risk", y="count", title="Risk Level Distribution")
        fig = apply_dark_theme(fig)

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # GLOBAL TREND
        # =========================
        st.subheader("Global Risk Trend")

        trend = df.groupby("month")["crisis_prob"].mean().reset_index()

        fig2 = px.line(
            trend,
            x="month",
            y="crisis_prob",
            title="Average Global Crisis Probability",
        )

        fig2 = apply_dark_theme(fig2)
        fig2.update_layout(yaxis_tickformat=".0%")

        st.plotly_chart(fig2, use_container_width=True)

        # =========================
        # TOP COUNTRIES
        # =========================
        st.subheader("Top Risk Economies")

        top3 = latest.head(3)

        for _, row in top3.iterrows():
            drivers = top_shap_drivers(
                df[df["country"] == row["country"]].tail(1),
                model,
            )

            st.write(
                f"- **{row['country']}** — {row['crisis_prob']:.1%} risk | Drivers: {', '.join(drivers)}"
            )

        st.markdown("---")

        # =========================
        # BENCHMARK
        # =========================
        st.subheader("Country Benchmark vs Global")

        selected = st.selectbox("Select Country", latest["country"].unique())

        sel_prob = float(
            latest[latest["country"] == selected]["crisis_prob"].values[0]
        )

        diff = sel_prob - avg

        if diff > 0.01:
            label = "Significantly above global risk"
        elif diff < -0.01:
            label = "Significantly below global risk"
        else:
            label = "In line with global average"

        st.metric(
            label="Country vs Global",
            value=selected,
            delta=f"{diff:+.2%}",
        )

        # =========================
        # EXECUTIVE SUMMARY (🔥 NEW)
        # =========================
        st.markdown("---")
        st.subheader("Executive Summary")

        if high > total * 0.3:
            st.error("Global risk environment is elevated. Multiple economies show instability signals.")
        elif high > 0:
            st.warning("Selective risk pockets identified. Monitoring required.")
        else:
            st.success("Global macroeconomic environment appears stable.")

    except Exception as e:
        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
