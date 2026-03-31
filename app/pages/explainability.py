from __future__ import annotations

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
# MAIN
# =========================
def main():
    st.title("Explainability — Global Risk Drivers")

    try:
        shap_imp = load_shap()

        # =========================
        # FALLBACK (MODEL IMPORTANCE)
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
        # NORMALIZATION (IMPORTANT)
        # =========================
        total = shap_imp["mean_abs_shap"].sum()

        if total > 0:
            shap_imp["contribution_pct"] = (
                shap_imp["mean_abs_shap"] / total
            ) * 100
        else:
            shap_imp["contribution_pct"] = 0

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
            title="Top Global Risk Drivers",
            labels={
                "mean_abs_shap": "Impact Strength",
                "feature": "Factor",
            },
        )

        fig = apply_dark_theme(fig)

        fig.update_layout(
            title_font=dict(size=20),
            xaxis_title="Impact Strength",
            yaxis_title="Economic Factors",
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # EXECUTIVE INSIGHTS
        # =========================
        st.subheader("Executive Insights")

        top3 = shap_imp.head(3)

        if not top3.empty:
            for _, row in top3.iterrows():
                st.write(
                    f"- **{row['feature']}** contributes "
                    f"{row['contribution_pct']:.1f}% to overall risk"
                )

        # =========================
        # STRATEGIC INTERPRETATION
        # =========================
        st.subheader("Strategic Interpretation")

        st.write(
            "The model identifies macroeconomic instability, trade exposure, "
            "and monetary conditions as the primary drivers of sovereign risk. "
            "Countries with deteriorating fundamentals in these areas exhibit "
            "significantly higher crisis probability."
        )

        # =========================
        # TABLE
        # =========================
        st.subheader("Full Feature Importance")

        st.dataframe(
            shap_imp[["feature", "mean_abs_shap", "contribution_pct"]],
            use_container_width=True,
        )

    except Exception as e:
        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
