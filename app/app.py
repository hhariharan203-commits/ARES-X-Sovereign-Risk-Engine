"""
ARES-X Sovereign Risk Intelligence System
Main Application (Final Production Version)
"""

import streamlit as st
from pathlib import Path
import sys

# ─────────────────────────────────────────────
# PATH FIX (IMPORTANT)
# ─────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import utils
import ui
import intelligence
import report

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ARES-X Sovereign Risk Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

ui.apply_theme()

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = utils.load_data()
countries = sorted(df["country"].unique())

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ ARES-X")
    st.caption("Sovereign Risk Intelligence System")

    selected_country = st.selectbox("Select Country", countries)

    st.session_state["country"] = selected_country

    st.divider()

    st.success("Model: Active")
    st.success("Data: Loaded")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🌍 Sovereign Risk Intelligence Platform")
st.caption("AI-driven macro risk analytics · Decision intelligence · Portfolio impact")

# ─────────────────────────────────────────────
# CURRENT COUNTRY ANALYSIS
# ─────────────────────────────────────────────
row = utils.latest(df, selected_country)

score, tier = utils.predict_full(row)

col1, col2 = st.columns(2)

with col1:
    st.metric("Risk Score", f"{score:.3f}")

with col2:
    st.metric("Risk Tier", tier)

# ─────────────────────────────────────────────
# INTELLIGENCE ENGINE (KEY WOW SECTION)
# ─────────────────────────────────────────────
intel = intelligence.generate_intelligence(row.iloc[0], score)

st.markdown("## 🧠 Strategic Intelligence Brief")

st.markdown(f"### Regime: {intel['regime']}")

st.markdown("### Key Risk Drivers")
if intel["drivers"]:
    for d in intel["drivers"]:
        st.write(f"- {d}")
else:
    st.write("No major risk drivers detected")

st.markdown("### Recommended Action")
st.write(intel["action"])

# ─────────────────────────────────────────────
# GLOBAL RISK RANKING
# ─────────────────────────────────────────────
st.markdown("## 🌐 Global Risk Ranking")

ranking = utils.global_risk(df)[["country", "risk_score"]]
st.dataframe(ranking.head(10), use_container_width=True)

# ─────────────────────────────────────────────
# REPORT GENERATION (EXECUTIVE FEATURE)
# ─────────────────────────────────────────────
st.markdown("## 📄 Executive Report")

if st.button("Generate Risk Report"):
    file_path = report.generate_report(selected_country, score, intel)

    with open(file_path, "rb") as f:
        st.download_button(
            label="Download Report",
            data=f,
            file_name=file_path,
            mime="application/pdf"
        )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("ARES-X · Built for Strategic Risk Intelligence · Production Ready System")
