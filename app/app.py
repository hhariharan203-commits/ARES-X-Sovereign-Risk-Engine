from __future__ import annotations

import streamlit as st

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
    # =========================
    # HEADER
    # =========================
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
    # KEY HIGHLIGHTS (NEW 🔥)
    # =========================
    col1, col2, col3 = st.columns(3)

    col1.metric("Countries Covered", "24+")
    col2.metric("Time Period", "2000–2024")
    col3.metric("Model Type", "XGBoost + SHAP")

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
    # BUSINESS VALUE (VERY IMPORTANT)
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
    # FOOTER
    # =========================
    st.caption("Data Sources: World Bank | FRED | GDELT | Google Trends")
    st.caption("Model: XGBoost | Explainability: SHAP")


if __name__ == "__main__":
    main()
