from __future__ import annotations

import os
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
METRICS_PATH = os.path.join(BASE_DIR, "data", "model_metrics.csv")


def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    try:
        df = pd.read_csv(METRICS_PATH)
        return df
    except Exception:
        return None


def main():
    st.title("Model Performance")

    df = load_metrics()
    if df is None or df.empty:
        st.warning("Metrics not found. Run the training pipeline to generate metrics.")
        return

    st.dataframe(df)


if __name__ == "__main__":
    main()
