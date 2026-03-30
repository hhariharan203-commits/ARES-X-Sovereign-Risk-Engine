from __future__ import annotations

import plotly.express as px
import streamlit as st

from utils import load_shap


def main():
    st.title("Explainability")
    shap_imp = load_shap()

    if shap_imp.empty:
        st.info("No SHAP importance data available.")
        return

    fig = px.bar(
        shap_imp,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title="Global Feature Importance (SHAP)",
    )
    fig.update_layout(xaxis_title="Mean |SHAP|", yaxis_title="Feature")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(shap_imp, use_container_width=True)


if __name__ == "__main__":
    main()
