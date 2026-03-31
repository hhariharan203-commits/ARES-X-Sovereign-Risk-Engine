from __future__ import annotations

import streamlit as st

# IMPORTANT: absolute import
from app.utils import load_data, load_model


# =========================
# PAGE CONFIG (MUST BE FIRST)
# =========================
st.set_page_config(
    page_title="ARES-X Control Center",
    layout="wide"
)

# =========================
# DARK THEME
# =========================
st.markdown(
    """
<style>
body {
    background-color: #0e1117;
}
h1, h2, h3, h4 {
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)


def main():
    st.title("ARES-X Control Center")

    st.markdown(
        """
**AI-Powered Sovereign Risk Intelligence Platform**

ARES-X integrates macroeconomic data, machine learning, and explainable AI  
to deliver early warning signals for financial crises across global economies.
"""
    )

    st.markdown("---")

    # =========================
    # LOAD DATA SAFELY
    # =========================
    try:
        df = load_data()
        data_ok = not df.empty
    except Exception:
        df = None
        data_ok = False

    try:
        model = load_model()
        model_ok = True
    except Exception:
        model_ok = False

    # =========================
    # DYNAMIC METRICS
    # =========================
    col1, col2, col3, col4 = st.columns(4)

    if data_ok:
        countries = df["country"].nunique() if "country" in df.columns else 0

        if "month" in df.columns:
            years = (
                f"{df['month'].dt.year.min()} – {df['month'].dt.year.max()}"
                if not df["month"].isna().all()
                else "N/A"
            )
        else:
            years = "N/A"

    else:
        countries = "N/A"
        years = "N/A"

    col1.metric("Countries Covered", countries)
    col2.metric("Time Period", years)
    col3.metric("Model Status", "Loaded" if model_ok else "Missing")
    col4.metric("System Status", "Operational" if (data_ok and model_ok) else "Partial")

    st.markdown("---")

    # =========================
    # NAVIGATION GUIDE
    # =========================
    st.subheader("Platform Modules")

    st.markdown(
        """
- **Global Risk** → Monitor cross-country crisis probabilities and global trends  
- **Country Intelligence** → Deep dive into individual economies with scenario simulation  
- **Executive Dashboard** → High-level risk signals and strategic insights  
- **Explainability** → Understand key drivers using SHAP-based feature importance  
- **Forecast** → Forward-looking crisis probability with scenario analysis  
"""
    )

    st.markdown("---")

    # =========================
    # BUSINESS VALUE
    # =========================
    st.subheader("Business Impact")

    st.markdown(
        """
- Identify high-risk sovereign economies using macroeconomic indicators  
- Provide early warning signals for financial instability  
- Support scenario-based policy and investment decisions  
- Combine machine learning with explainable AI for transparency  
"""
    )

    st.markdown("---")

    # =========================
    # SYSTEM HEALTH (NEW 🔥)
    # =========================
    st.subheader("System Health")

    if not data_ok:
        st.warning("⚠️ Dataset not loaded correctly")
    if not model_ok:
        st.warning("⚠️ Model not loaded")

    if data_ok and model_ok:
        st.success("✅ All systems operational")

    st.markdown("---")

    # =========================
    # FOOTER
    # =========================
    st.caption("Data Sources: World Bank | FRED | GDELT | Google Trends")
    st.caption("Model: XGBoost | Explainability: SHAP")


if __name__ == "__main__":
    main()
