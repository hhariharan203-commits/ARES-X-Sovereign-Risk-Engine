import streamlit as st
import utils
from ui import render_sidebar, render_home, render_global_risk, render_country

# ─────────────────────────────
st.set_page_config(page_title="ARES-X", layout="wide")

# ─────────────────────────────
@st.cache_resource
def init_system():
    return {
        "df": utils.load_dataset(),
        "model": utils.load_model(),
        "scaler": utils.load_scaler(),
        "features": utils.load_feature_cols(),
        "metrics": utils.load_model_metrics()
    }

system = init_system()

# ─────────────────────────────
if "view" not in st.session_state:
    st.session_state["view"] = "Home"

render_sidebar()

view = st.session_state["view"]

if view == "Home":
    render_home(system)

elif view == "Global Risk":
    render_global_risk(system)

elif view == "Country Intelligence":
    render_country(system)
