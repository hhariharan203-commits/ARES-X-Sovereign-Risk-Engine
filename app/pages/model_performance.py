from __future__ import annotations

import json
import os
import streamlit as st


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
METRICS_PATH = os.path.join(BASE_DIR, "outputs", "model_metrics.json")


# =========================
# LOAD METRICS
# =========================
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


# =========================
# MAIN
# =========================
def main():
    st.title("Model Performance")

    metrics = load_metrics()

    if not metrics:
        st.warning("Metrics not found. Run model training pipeline to generate metrics.")
        return

    accuracy = metrics.get("accuracy", 0)
    precision = metrics.get("precision", 0)
    recall = metrics.get("recall", 0)
    f1 = metrics.get("f1", 0)
    roc_auc = metrics.get("roc_auc", 0)

    # =========================
    # KPI DISPLAY
    # =========================
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy:.3f}")
    col2.metric("Precision", f"{precision:.3f}")
    col3.metric("Recall", f"{recall:.3f}")

    col4, col5 = st.columns(2)
    col4.metric("F1 Score", f"{f1:.3f}")
    col5.metric("ROC-AUC", f"{roc_auc:.3f}")

    # =========================
    # INTERPRETATION (CRITICAL)
    # =========================
    st.subheader("Model Interpretation")

    st.write(
        f"The model achieves a ROC-AUC of **{roc_auc:.2f}**, indicating strong ability to distinguish "
        "between stable and crisis conditions across countries."
    )

    st.write(
        f"Recall is **{recall:.2f}**, which is prioritized in this system to ensure early detection "
        "of financial crises — minimizing the risk of missed events."
    )

    st.write(
        f"Precision is **{precision:.2f}**, reflecting the inherent class imbalance in sovereign crisis data "
        "(crisis events are rare), resulting in more false positives but stronger early warning capability."
    )

    # =========================
    # BUSINESS CONTEXT
    # =========================
    st.subheader("Why This Matters")

    st.write(
        "- Early detection (high recall) helps policymakers and investors react before crises escalate"
    )
    st.write(
        "- Accepting lower precision is intentional to avoid missing critical financial instability signals"
    )
    st.write(
        "- Model is optimized for **risk surveillance**, not exact prediction"
    )

    # =========================
    # DATA NOTE
    # =========================
    st.caption("Metrics generated from training pipeline (XGBoost + SHAP)")



if __name__ == "__main__":
    main()
