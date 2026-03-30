from __future__ import annotations

import json
import os

import streamlit as st


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
METRICS_PATH = os.path.join(BASE_DIR, "outputs", "model_metrics.json")


def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def main():
    st.title("Model Performance")

    metrics = load_metrics()
    if not metrics:
        st.info("Metrics not found. Run `python src/train_model.py` to generate model_metrics.json.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
    col2.metric("Precision", f"{metrics.get('precision', 0):.3f}")
    col3.metric("Recall", f"{metrics.get('recall', 0):.3f}")

    col4, col5 = st.columns(2)
    col4.metric("F1 Score", f"{metrics.get('f1', 0):.3f}")
    col5.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")

    st.caption("Metrics loaded from outputs/model_metrics.json")


if __name__ == "__main__":
    main()
