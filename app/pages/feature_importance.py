from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import apply_dark_theme, load_model, load_feature_cols


BASE_DIR = Path(__file__).resolve().parents[2]
FEAT_PATH = BASE_DIR / "data" / "feature_importance.csv"


def load_importance() -> pd.DataFrame:
    if not FEAT_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(FEAT_PATH)
        df.columns = df.columns.str.strip().str.lower()
        return df
    except Exception:
        return pd.DataFrame()


def main():
    st.title("Feature Importance")

    df = load_importance()
    if df.empty:
        model = load_model()
        if hasattr(model, "feature_importances_"):
            feats = load_feature_cols()
            df = pd.DataFrame(
                {"feature": feats[: len(model.feature_importances_)], "importance": model.feature_importances_}
            )
        else:
            st.info("No feature importance data available.")
            return

    required = {"feature", "importance"}
    if not required.issubset(set(df.columns)):
        st.warning("Feature importance data incomplete.")
        return

    df_top = df.head(10)
    fig = px.bar(
        df_top,
        x="importance",
        y="feature",
        orientation="h",
        title="Top 10 Features Driving Crisis Risk",
    )
    fig = apply_dark_theme(fig)
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### What drives risk?")
    st.write(
        "Higher importance indicates stronger influence on predicted crisis probability. "
        "Use this to understand which macro factors are most impactful."
    )


if __name__ == "__main__":
    main()
