import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, auc

from app.utils import safe_metric

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
    df = load_csv_safe(METRICS_PATH)
    if df is None or df.empty:
        return None
    return df.iloc[0].to_dict()


def load_detail():
    return load_csv_safe(PERF_DETAIL_PATH)


# ================= MAIN =================
def main():
    st.title("Model Performance")

    metrics = load_metrics()
    detail = load_detail()

    if not metrics:
        st.warning("Metrics not found. Run training pipeline.")
        return

    # ================= METRICS =================
    accuracy = float(metrics.get("accuracy", 0))
    precision = float(metrics.get("precision", 0))
    recall = float(metrics.get("recall", 0))
    f1 = float(metrics.get("f1", 0))
    roc_auc = float(metrics.get("roc_auc", 0))

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", f"{accuracy:.2f}")
    col2.metric("Precision", f"{precision:.2f}")
    col3.metric("Recall", f"{recall:.2f}")
    col4.metric("F1 Score", f"{f1:.2f}")
    col5.metric("ROC-AUC", f"{roc_auc:.2f}")

    st.markdown("---")

    # ================= INTERPRETATION =================
    if roc_auc > 0.95:
        st.error("⚠️ Performance too high → possible overfitting/data leakage")
    elif 0.75 <= roc_auc <= 0.90:
        st.success("✅ Model performance is realistic and reliable")
    else:
        st.warning("⚠️ Model performance is weak")

    # ================= BENCHMARK =================
    st.subheader("Benchmark Comparison")

    st.write("Industry ROC-AUC Range: **0.75 – 0.85**")
    st.write(f"Model ROC-AUC: **{roc_auc:.2f}**")

    # ================= VISUALS =================
    if detail is None or detail.empty:
        st.warning("No performance detail data available.")
    else:
        # Detect target column
        target_col = None
        for col in ["crisis_target", "crisis"]:
            if col in detail.columns:
                target_col = col
                break

        if target_col is None:
            st.warning("Target column not found.")
            return

        y_true = detail[target_col]

        y_pred = detail.get("y_pred") or detail.get("pred")
        y_prob = detail.get("y_prob") or detail.get("crisis_prob")

        if y_pred is None or y_prob is None:
            st.warning("Prediction columns missing.")
            return

        # ================= CONFUSION MATRIX =================
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)

        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Pred 0", "Pred 1"]
        )

        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Confusion Matrix"
        )

        st.plotly_chart(fig_cm, use_container_width=True)

        # ================= ROC CURVE =================
        st.subheader("ROC Curve")

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_score = auc(fpr, tpr)

        fig_roc = go.Figure()

        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"Model (AUC = {roc_score:.2f})"
            )
        )

        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash"),
                name="Random"
            )
        )

        fig_roc.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )

        st.plotly_chart(fig_roc, use_container_width=True)

    # ================= EXECUTIVE SUMMARY =================
    st.markdown("---")
    st.subheader("Executive Summary")

    if roc_auc > 0.95:
        st.write(
            "Model performance appears unusually high, indicating potential overfitting or data leakage. "
            "Further validation required before deployment."
        )
    elif roc_auc >= 0.75:
        st.write(
            "Model demonstrates strong predictive capability with realistic performance. "
            "Suitable for risk monitoring and decision support."
        )
    else:
        st.write(
            "Model performance is below acceptable threshold. "
            "Requires feature engineering, better data, or model tuning."
        )


if __name__ == "__main__":
    main()
