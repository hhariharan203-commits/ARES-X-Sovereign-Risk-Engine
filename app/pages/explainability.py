from __future__ import annotations

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import plotly.express as px
import streamlit as st
import pandas as pd

from utils import (
    load_shap,
    humanize_feature,
    apply_dark_theme,
    load_model,
    load_feature_cols,
)


# =========================
# FEATURE GROUPING (NEW 🔥)
# =========================
def categorize_feature(feat: str) -> str:
    feat = feat.lower()

    if "gdp" in feat:
        return "Economic Growth"
    elif "inflation" in feat or "interest" in feat:
        return "Monetary Conditions"
    elif "export" in feat or "import" in feat:
        return "Trade Exposure"
    elif "unemployment" in feat:
        return "Labor Market"
    else:
        return "Other"


# =========================
# MAIN
# =========================
def main():
    st.title("Explainability — Global Risk Drivers")

    try:
        shap_imp = load_shap()

        # =========================
        # FALLBACK
        # =========================
        if shap_imp.empty:
            model = load_model()

            if hasattr(model, "feature_importances_"):
                feats = load_feature_cols()

                shap_imp = pd.DataFrame({
                    "feature": feats[: len(model.feature_importances_)],
                    "mean_abs_shap": model.feature_importances_,
                })
            else:
                st.warning("No feature importance available")
                return

        # =========================
        # CLEAN
        # =========================
        shap_imp.columns = shap_imp.columns.str.strip().str.lower()

        if not {"feature", "mean_abs_shap"}.issubset(shap_imp.columns):
            st.error("Invalid SHAP format")
            return

        shap_imp = shap_imp.dropna(subset=["feature", "mean_abs_shap"])

        # =========================
        # NORMALIZE
        # =========================
        total = shap_imp["mean_abs_shap"].sum()

        if total > 0:
            shap_imp["contribution_pct"] = (
                shap_imp["mean_abs_shap"] / total
            ) * 100
        else:
            shap_imp["contribution_pct"] = 0

        # =========================
        # HUMANIZE + CATEGORY
        # =========================
        shap_imp["category"] = shap_imp["feature"].apply(categorize_feature)
        shap_imp["feature"] = shap_imp["feature"].apply(humanize_feature)

        shap_imp = shap_imp.sort_values("mean_abs_shap", ascending=False)

        # =========================
        # TOP FEATURES
        # =========================
        top_n = 15
        shap_top = shap_imp.head(top_n).sort_values("mean_abs_shap")

        # =========================
        # CHART
        # =========================
        fig = px.bar(
            shap_top,
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            color="category",
            title="Top Global Risk Drivers",
            labels={
                "mean_abs_shap": "Impact Strength",
                "feature": "Factor",
            },
        )

        fig = apply_dark_theme(fig)

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # CATEGORY IMPORTANCE (NEW 🔥)
        # =========================
        st.subheader("Risk Contribution by Category")

        cat = (
            shap_imp.groupby("category")["contribution_pct"]
            .sum()
            .reset_index()
            .sort_values("contribution_pct", ascending=False)
        )

        fig2 = px.pie(
            cat,
            names="category",
            values="contribution_pct",
            title="Macro Risk Breakdown",
        )

        fig2 = apply_dark_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

        # =========================
        # EXECUTIVE INSIGHTS
        # =========================
        st.subheader("Executive Insights")

        top3 = shap_imp.head(3)

        for _, row in top3.iterrows():
            st.write(
                f"- **{row['feature']}** contributes "
                f"{row['contribution_pct']:.1f}% to risk"
            )

        # =========================
        # STRATEGIC INTERPRETATION
        # =========================
        st.subheader("Strategic Interpretation")

        dominant = cat.iloc[0]["category"] if not cat.empty else None

        if dominant:
            st.write(
                f"The model indicates that **{dominant}** is the dominant driver "
                "of sovereign risk. Policy focus in this area can significantly "
                "reduce financial instability."
            )

        st.write(
            "Overall, sovereign risk is driven by a combination of macroeconomic "
            "instability, monetary tightening, and external trade exposure."
        )

        # =========================
        # TABLE
        # =========================
        st.subheader("Detailed Feature Importance")

        st.dataframe(
            shap_imp[["feature", "category", "contribution_pct"]],
            use_container_width=True,
        )

    except Exception as e:
        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
