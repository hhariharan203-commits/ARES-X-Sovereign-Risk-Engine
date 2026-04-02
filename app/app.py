"""
ARES-X: Sovereign Risk Intelligence Engine
Top-tier decision intelligence platform for macro risk.
"""

import streamlit as st
import utils
from ui import apply_theme, render_sidebar, render_home

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ARES-X | Sovereign Risk Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# INIT SYSTEM (CRITICAL)
# ─────────────────────────────────────────────
@st.cache_resource
def init_system():
    df = utils.load_dataset()
    model = utils.load_model()
    scaler = utils.load_scaler()
    features = utils.load_feature_cols()
    metrics = utils.load_model_metrics()

    return {
        "df": df,
        "model": model,
        "scaler": scaler,
        "features": features,
        "metrics": metrics
    }


system = init_system()

# ─────────────────────────────────────────────
# SESSION STATE (GLOBAL CONTROL)
# ─────────────────────────────────────────────
if "country" not in st.session_state:
    st.session_state["country"] = "USA"

if "view" not in st.session_state:
    st.session_state["view"] = "Home"

# ─────────────────────────────────────────────
# APPLY UI
# ─────────────────────────────────────────────
apply_theme()

# ─────────────────────────────────────────────
# SIDEBAR (GLOBAL NAVIGATION)
# ─────────────────────────────────────────────
render_sidebar(system)

# ─────────────────────────────────────────────
# ROUTER (IMPORTANT)
# ─────────────────────────────────────────────
view = st.session_state["view"]

if view == "Home":
    render_home(system)

elif view == "Global Risk":
    from pages.global_risk import render
    render(system)

elif view == "Country Intelligence":
    from pages.country_intelligence import render
    render(system)

elif view == "Forecast":
    from pages.forecast import render
    render(system)

elif view == "Explainability":
    from pages.explainability import render
    render(system)

elif view == "Model Performance":
    from pages.model_performance import render
    render(system)

elif view == "Portfolio Impact":
    from pages.portfolio_impact import render
    render(system)

elif view == "Regime Detection":
    from pages.regime_detection import render
    render(system)

elif view == "Scenario Lab":
    from pages.scenario_lab import render
    render(system)

else:
    st.error("Unknown view selected")
