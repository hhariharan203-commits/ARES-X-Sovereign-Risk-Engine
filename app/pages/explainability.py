from __future__ import annotations

import plotly.express as px
import streamlit as st

from utils import load_shap, humanize_feature, apply_dark_theme


def main():
    st.title("Explainability")

    shap_imp = load_shap()

    # =========================
    # SAFETY CHECKS
    # =========================
    if shap_imp.empty:
        st.warning("No SHAP importance data available.")
        return

    required_cols = {"feature", "mean_abs_shap"}
    if not required_cols.issubset(set(shap_imp.columns)):
        st.error("Invalid SHAP data format.")
        return

    # =========================
    # CLEAN DATA
    # =========================
    shap_imp = shap_imp.copy()

    # Remove NaN safely
    shap_imp = shap_imp.dropna(subset=["feature", "mean_abs_shap"])

    # Humanize feature names
    shap_imp["feature"] = shap_imp["feature"].apply(humanize_feature)

    # Sort
    shap_imp = shap_imp.sort_values("mean_abs_shap", ascending=False)

    # =========================
    # LIMIT TOP FEATURES (UX FIX)
    # =========================
    top_n = 15
    shap_top = shap_imp.head(top_n).sort_values("mean_abs_shap", ascending=True)

    # =========================
    # 📊 BAR CHART
    # =========================
    fig = px.bar(
        shap_top,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title=f"Top {top_n} Global Feature Importance (SHAP)",
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
    # 📌 INSIGHT SECTION (NEW)
    # =========================
    st.subheader("Key Insights")

    top_features = shap_imp.head(3)["feature"].tolist()

    if top_features:
        st.write(
            f"Top global risk drivers are **{', '.join(top_features)}**, "
            "indicating these factors have the strongest influence on crisis probability."
        )

    # =========================
    # 📋 TABLE
    # =========================
    st.subheader("Feature Importance Table")

    st.dataframe(
        shap_imp,
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
