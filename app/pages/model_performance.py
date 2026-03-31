from __future__ import annotations

import os
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve

from utils import safe_metric

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
METRICS_PATH = os.path.join(BASE_DIR, "data", "model_metrics.csv")
PERF_DETAIL_PATH = os.path.join(BASE_DIR, "data", "performance_detail.csv")


# ================= LOADERS =================
def load_csv_safe(path):
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        return df
    except Exception:
        return None


def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    df = load_csv_safe(METRICS_PATH)
    if df is None or df.empty:
        return None
    return df.iloc[0].to_dict()


def load_detail():
    if not os.path.exists(PERF_DETAIL_PATH):
        return None
    return load_csv_safe(PERF_DETAIL_PATH)


# ================= MAIN =================
def main():
    st.title("Model Performance")

    metrics = load_metrics()
    detail = load_detail()

    if not metrics:
        st.warning("Metrics not found. Run training pipeline.")
        return

    # Extract metrics safely
    accuracy = float(metrics.get("accuracy", 0))
    precision = float(metrics.get("precision", 0))
    recall = float(metrics.get("recall", 0))
    f1 = float(metrics.get("f1", 0))
    roc_auc = float(metrics.get("roc_auc", 0))

    # ================= KPI CARDS =================
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        safe_metric("Accuracy", f"{accuracy:.2f}")
    with col2:
        safe_metric("Precision", f"{precision:.2f}")
    with col3:
        safe_metric("Recall", f"{recall:.2f}")
    with col4:
        safe_metric("F1 Score", f"{f1:.2f}")
    with col5:
        safe_metric("ROC-AUC", f"{roc_auc:.2f}")

    st.markdown("---")

    # ================= INTERPRETATION =================
    if roc_auc > 0.95:
        st.error("⚠️ Overfitting risk detected")
    elif 0.75 <= roc_auc <= 0.90:
        st.success("✅ Model performance is realistic and reliable")
    else:
        st.warning("⚠️ Model performance is weak → needs improvement")

    # ================= BENCHMARK =================
    st.subheader("Benchmark Comparison")
    st.write("Industry Expected ROC-AUC: 0.75 – 0.85")
    st.write(f"Current Model ROC-AUC: {roc_auc:.2f}")

    # ================= CONFUSION MATRIX + ROC =================
    if detail is None or detail.empty:
        st.warning("No performance detail data available.")
    else:
        # Target detection
        target_col = None
        for col in ["crisis_target", "crisis"]:
            if col in detail.columns:
                target_col = col
                break

        if target_col is None:
            st.warning("Target column not found in dataset.")
        else:
            y_true = detail[target_col]

            # Safe predictions
            y_pred = None
            if "y_pred" in detail.columns:
                y_pred = detail["y_pred"]
            elif "pred" in detail.columns:
                y_pred = detail["pred"]

            y_prob = None
            if "y_prob" in detail.columns:
                y_prob = detail["y_prob"]
            elif "crisis_prob" in detail.columns:
                y_prob = detail["crisis_prob"]

            if y_pred is None or y_prob is None:
                st.warning("Prediction columns missing.")
            else:
                st.markdown("---")

                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)

                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

                # ROC Curve
                st.subheader("ROC Curve")
                fpr, tpr, _ = roc_curve(y_true, y_prob)

                fig2, ax2 = plt.subplots()
                ax2.plot(fpr, tpr)
                ax2.plot([0, 1], [0, 1], linestyle="--")
                ax2.set_xlabel("False Positive Rate")
                ax2.set_ylabel("True Positive Rate")
                ax2.set_title("ROC Curve")
                st.pyplot(fig2)

    # ================= EXECUTIVE SUMMARY =================
    st.markdown("---")
    st.subheader("Executive Summary")

    if roc_auc > 0.95:
        st.write("Model likely overfitted. Needs validation before deployment.")
    elif roc_auc >= 0.75:
        st.write("Model shows strong predictive capability with realistic performance.")
    else:
        st.write("Model performance is weak. Feature engineering or data improvement required.")


if __name__ == "__main__":
    main()
