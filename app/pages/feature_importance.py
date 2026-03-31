from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import apply_dark_theme, load_model, load_feature_cols, humanize_feature


BASE_DIR = Path(__file__).resolve().parents[2]
FEAT_PATH = BASE_DIR / "data" / "feature_importance.csv"


# =========================
# GROUPING LOGIC
# =========================
def assign_group(feature: str) -> str:
    f = feature.lower()

    if "gdp" in f:
        return "Economic Growth"
    if "inflation" in f:
        return "Inflation"
    if "interest" in f:
        return "Monetary Policy"
    if "export" in f:
        return "Trade Strength"
    if "import" in f:
        return "Import Dependency"
    if "unemployment" in f:
        return "Labor Market"

    return "Other"


# =========================
# LOAD DATA
# =========================
def load_importance() -> pd.DataFrame:
    if not FEAT_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(FEAT_PATH)
        df.columns = df.columns.str.strip().str.lower()
        return df
    except Exception:
        return pd.DataFrame()


# =========================
# MAIN
# =========================
def main():
    st.title("Feature Importance — Risk Drivers")

    try:
        df = load_importance()

        # =========================
        # FALLBACK
        # =========================
        if df.empty:
            model = load_model()

            if hasattr(model, "feature_importances_"):
                feats = load_feature_cols()
                df = pd.DataFrame({
                    "feature": feats[: len(model.feature_importances_)],
                    "importance": model.feature_importances_,
                })
            else:
                st.warning("No feature importance available")
                return

        # =========================
        # VALIDATION
        # =========================
        if not {"feature", "importance"}.issubset(df.columns):
            st.warning("Feature importance data incomplete")
            return

        df = df.dropna(subset=["feature", "importance"])

        # =========================
        # NORMALIZE (%)
        # =========================
        total = df["importance"].sum()

        df["contribution_pct"] = (
            df["importance"] / total * 100 if total > 0 else 0
        )

        # =========================
        # GROUPING
        # =========================
        df["group"] = df["feature"].apply(assign_group)

        # Human readable names
        df["feature"] = df["feature"].apply(humanize_feature)

        df = df.sort_values("importance", ascending=False)

        # =========================
        # TOP FEATURES CHART
        # =========================
        top = df.head(12).sort_values("importance")

        fig = px.bar(
            top,
            x="importance",
            y="feature",
            orientation="h",
            color="group",
            title="Top Drivers of Crisis Risk",
        )

        fig = apply_dark_theme(fig)

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # GROUP DISTRIBUTION
        # =========================
        st.subheader("Risk Contribution by Category")

        group_df = (
            df.groupby("group")["contribution_pct"]
            .sum()
            .reset_index()
            .sort_values("contribution_pct", ascending=False)
        )

        fig2 = px.pie(
            group_df,
            names="group",
            values="contribution_pct",
            title="Macroeconomic Risk Composition",
        )

        fig2 = apply_dark_theme(fig2)

        st.plotly_chart(fig2, use_container_width=True)

        # =========================
        # EXECUTIVE INSIGHTS
        # =========================
        st.subheader("Executive Insights")

        top_group = group_df.iloc[0]

        st.write(
            f"**{top_group['group']}** is the dominant driver of sovereign risk, "
            f"contributing approximately **{top_group['contribution_pct']:.1f}%** "
            "to overall crisis probability. Secondary drivers include macroeconomic "
            "instability and external trade exposure."
        )

        # =========================
        # TABLE
        # =========================
        st.subheader("Detailed Breakdown")

        st.dataframe(
            df[["feature", "group", "contribution_pct"]],
            use_container_width=True,
        )

    except Exception as e:
        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
