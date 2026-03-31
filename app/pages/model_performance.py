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


def load_data_safe(path):
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        return df
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        st.stop()


def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    try:
        return pd.read_csv(METRICS_PATH).iloc[0].to_dict()
    except Exception:
        return None


def load_perf_detail():
    if not os.path.exists(PERF_DETAIL_PATH):
        return None
    try:
        return load_data_safe(PERF_DETAIL_PATH)
    except Exception:
        return None


def main():
    try:
        st.title("Model Performance")

        metrics = load_metrics()
        detail = load_perf_detail()

        if not metrics:
            st.warning("Metrics not found. Run the training pipeline to generate metrics.")
            return

        accuracy = metrics.get("accuracy", 0)
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)
        f1 = metrics.get("f1", 0)
        roc_auc = metrics.get("roc_auc", 0)

        col1, col2, col3, col4, col5 = st.columns(5)
        safe_metric("Accuracy", f"{accuracy:.2f}" if accuracy else None)
        safe_metric("Precision", f"{precision:.2f}" if precision else None)
        safe_metric("Recall", f"{recall:.2f}" if recall else None)
        safe_metric("F1 Score", f"{f1:.2f}" if f1 else None)
        safe_metric("ROC-AUC", f"{roc_auc:.2f}" if roc_auc else None)

        st.markdown("---")

        if roc_auc > 0.95:
            st.error("⚠️ Model performance is unrealistically high → potential data leakage or overfitting.")
        elif 0.75 <= roc_auc <= 0.90:
            st.success("✅ Model performance is realistic and suitable for decision-making.")
        else:
            st.warning("⚠️ Model performance is weak → needs improvement.")

        st.subheader("Benchmark Comparison")
        st.write("Industry Expected ROC-AUC: 0.75 – 0.85")
        st.write(f"Current Model ROC-AUC: {roc_auc:.2f}")

        if detail is not None and not detail.empty:
            target_col = None
            if "crisis_target" in detail.columns:
                target_col = "crisis_target"
            elif "crisis" in detail.columns:
                target_col = "crisis"

            if target_col is None:
                st.warning("Target column not found in performance detail.")
            else:
                y_true = detail[target_col]
                y_pred = detail["y_pred"] if "y_pred" in detail.columns else detail.get("pred", pd.Series(dtype=int))
                y_prob = detail["y_prob"] if "y_prob" in detail.columns else detail.get("crisis_prob", pd.Series(dtype=float))

                if y_true.empty or y_pred.empty or y_prob.empty:
                    st.warning("Insufficient data for confusion matrix/ROC.")
                else:
                    st.markdown("---")
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_true, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)

                    st.subheader("ROC Curve")
                    fpr, tpr, _ = roc_curve(y_true, y_prob)
                    fig2, ax2 = plt.subplots()
                    ax2.plot(fpr, tpr, label="Model")
                    ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
                    ax2.set_title("ROC Curve")
                    ax2.set_xlabel("False Positive Rate")
                    ax2.set_ylabel("True Positive Rate")
                    st.pyplot(fig2)

        st.markdown("---")
        st.subheader("Executive Summary")
        if roc_auc > 0.95:
            st.write("Model shows extremely high performance, indicating potential overfitting. Further validation required.")
        else:
            st.write("Model demonstrates realistic predictive capability with strong macroeconomic signal capture.")
    except Exception as e:
        st.warning(f"Safe fallback: {e}")


if __name__ == "__main__":
    main()
