from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (
    load_data,
    load_model,
    load_explainer,
    add_probabilities,
    apply_dark_theme,
    humanize_feature,
    align_features,
)

# =========================
# SHAP DRIVERS
# =========================
def top_shap_drivers(row, explainer, top_n=2):
    try:
        aligned = align_features(row).iloc[[0]]

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

        return drivers

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
        explainer = load_explainer()

        # ✅ SINGLE SOURCE OF TRUTH
        df = add_probabilities(df)

        latest = (
            df.sort_values("month")
            .groupby("country")
            .tail(1)
            .sort_values("crisis_prob", ascending=False)
        )

        # ================= KPI =================
        total = latest["country"].nunique()
        high = (latest["risk_level"] == "HIGH").sum()
        avg = latest["crisis_prob"].mean()

        top = latest.iloc[0]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Countries", total)
        c2.metric("High Risk", high)
        c3.metric("Avg Risk", f"{avg:.2%}")
        c4.metric("Top Risk", top["country"], f"{top['crisis_prob']:.1%}")

        # ================= DISTRIBUTION =================
        st.subheader("Risk Distribution")

        dist = latest["risk_level"].value_counts().reset_index()
        dist.columns = ["risk", "count"]

        fig = px.bar(dist, x="risk", y="count")
        fig = apply_dark_theme(fig)

        st.plotly_chart(fig, use_container_width=True)

        # ================= TREND =================
        st.subheader("Global Trend")

        trend = df.groupby("month")["crisis_prob"].mean().reset_index()

        fig2 = px.line(trend, x="month", y="crisis_prob")
        fig2 = apply_dark_theme(fig2)

        fig2.update_layout(yaxis_tickformat=".0%")

        st.plotly_chart(fig2, use_container_width=True)

        # ================= TOP COUNTRIES =================
        st.subheader("Top Risk Countries")

        top3 = latest.head(3)

        for _, row in top3.iterrows():
            drivers = top_shap_drivers(
                df[df["country"] == row["country"]].tail(1),
                explainer,
            )

            st.write(
                f"- **{row['country']}** — {row['crisis_prob']:.1%} | {', '.join(drivers)}"
            )

        # ================= BENCHMARK =================
        st.subheader("Benchmark")

        selected = st.selectbox("Country", latest["country"].unique())

        sel_prob = float(latest[latest["country"] == selected]["crisis_prob"])

        diff = sel_prob - avg

        if diff > 0.002:
            label = "Above global risk"
        elif diff < -0.002:
            label = "Below global risk"
        else:
            label = "In line"

        st.write(f"{selected}: {label}")

    except Exception as e:
        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
