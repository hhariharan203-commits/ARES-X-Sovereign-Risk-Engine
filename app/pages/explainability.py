from __future__ import annotations

import plotly.express as px
import streamlit as st

# ✅ FIXED IMPORT
from utils import load_shap, humanize_feature, apply_dark_theme


def main():
    st.title("Explainability")

    shap_imp = load_shap()

    if shap_imp.empty:
        st.info("No SHAP importance data available.")
        return

    # =========================
    # ✅ HUMANIZE FEATURE NAMES
    # =========================
    shap_imp = shap_imp.copy()
    shap_imp["feature"] = shap_imp["feature"].apply(humanize_feature)

    # Sort properly
    shap_imp = shap_imp.sort_values("mean_abs_shap", ascending=True)

    # =========================
    # 📊 BAR CHART
    # =========================
    fig = px.bar(
        shap_imp,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title="Global Feature Importance (SHAP)",
        labels={"mean_abs_shap": "Mean |SHAP|", "feature": "Feature"},
    )

    fig = apply_dark_theme(fig)

    fig.update_layout(
        title_font=dict(size=18),
        xaxis_title="Mean |SHAP Impact|",
        yaxis_title="Feature",
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # 📋 TABLE
    # =========================
    st.subheader("Feature Importance Table")

    st.dataframe(
        shap_imp.sort_values("mean_abs_shap", ascending=False),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
