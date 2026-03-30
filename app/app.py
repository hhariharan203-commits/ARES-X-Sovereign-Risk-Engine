from __future__ import annotations

import streamlit as st

st.markdown(
    """
<style>
body {
    background-color: #0e1117;
}

h1, h2, h3 {
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)


def main():
    st.set_page_config(page_title="ARES-X Control Center", layout="wide")
    st.title("ARES-X Control Center")
    st.markdown(
        """
        Welcome to the ARES-X multi-page dashboard. Use the sidebar to navigate between:
        - **Global Risk** for cross-country risk snapshots  
        - **Country Intelligence** for deep dives  
        - **Explainability** for SHAP-based insights  
        - **Forecast** for forward-looking crisis risk
        """
    )
    st.markdown("---")
    st.caption("Data: World Bank API | Countries: 24 | Period: 2000–2024")


if __name__ == "__main__":
    main()
