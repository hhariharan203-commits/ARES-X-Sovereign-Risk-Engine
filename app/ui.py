import streamlit as st

# ✅ ADD THIS (missing earlier)
PLOTLY_THEME = "plotly_dark"


def apply_theme():
    st.markdown("""
        <style>
        body {
            background-color: #0E1117;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)


def render_sidebar(system=None):
    st.sidebar.title("ARES-X")

    menu = [
        "Home",
        "Global Risk",
        "Country Intelligence",
        "Forecast",
        "Explainability",
        "Model Performance",
        "Portfolio Impact",
        "Regime Detection",
        "Scenario Lab"
    ]

    choice = st.sidebar.radio("Navigation", menu)
    st.session_state["view"] = choice


def render_home(system):
    st.title("ARES-X Sovereign Risk Intelligence")
    st.write("Elite Macro Risk Engine Running")
