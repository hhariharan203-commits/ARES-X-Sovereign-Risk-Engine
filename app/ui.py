"""
Sovereign Risk Decision Intelligence Engine
UI Components — Production Grade (Final)
Institutional — BlackRock / Goldman Sachs
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Tuple

from intelligence import (
    DECISION_COLORS,
    REGIME_COLORS,
    global_risk_summary,
    run_scenario,
)

# ─── Design Tokens ─────────────────────────────────────────

DARK_BG      = "#0A0C14"
CARD_BG      = "#0F1220"
BORDER_COLOR = "#1E2235"
TEXT_PRIMARY = "#E8EAF0"
TEXT_MUTED   = "#6B7190"
ACCENT_BLUE  = "#3B82F6"
ACCENT_TEAL  = "#00C896"

# ─── CSS ──────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    html, body { background-color:#0A0C14; color:#E8EAF0; }
    .risk-card {
        background:#0F1220;
        border:1px solid #1E2235;
        border-radius:8px;
        padding:20px;
        margin-bottom:12px;
    }
    .section-label {
        font-size:10px;
        letter-spacing:0.15em;
        color:#6B7190;
        text-transform:uppercase;
    }
    </style>
    """, unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame) -> str:
    with st.sidebar:
        nav = st.radio(
            "NAVIGATION",
            ["Overview", "Global Risk", "Country Intelligence", "Scenario Lab"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        st.write(f"Countries: {len(df)}")
        st.write(f"Avg Risk: {df['risk_score'].mean():.2%}")

    return nav

# ─── Overview ─────────────────────────────────────────────

def render_overview(df: pd.DataFrame):
    summary = global_risk_summary(df)

    st.title("Global Sovereign Risk")

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk", f"{summary['avg_risk']:.2%}")
    c2.metric("Countries", summary["total_countries"])
    c3.metric("Status", summary["status"])

    st.markdown("---")

    top = df.sort_values("risk_score", ascending=False).head(10)

    fig = go.Figure(go.Bar(
        x=top["risk_score"],
        y=top["country"],
        orientation="h",
        marker_color=[DECISION_COLORS.get(d) for d in top["decision"]]
    ))
    st.plotly_chart(fig, use_container_width=True)

# ─── Country Intelligence ─────────────────────────────────

def render_country_intelligence(df, model, scaler, feature_cols):

    country = st.selectbox("Country", df["country"].unique())

    row = df[df["country"] == country].iloc[0]

    risk = row["risk_score"]
    decision = row["decision"]
    regime = row["regime"]
    reasoning = row["reasoning"]
    impact = row["impact"]
    action = row["action"]
    drivers = row["drivers"]

    color = DECISION_COLORS.get(decision)

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk", f"{risk:.2%}")
    c2.metric("Decision", decision)
    c3.metric("Regime", regime)

    st.markdown("---")

    # Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk * 100,
        gauge={"axis": {"range": [0, 100]}},
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Drivers
    names = [d[0] for d in drivers]
    vals = [d[1] for d in drivers]

    fig = go.Figure(go.Bar(
        x=vals,
        y=names,
        orientation="h"
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Decision Block
    st.markdown(f"""
    <div class='risk-card' style='border-left:3px solid {color};'>

    <div class='section-label'>WHAT IS HAPPENING</div>
    <p>{reasoning}</p>

    <div class='section-label'>WHY IT MATTERS</div>
    <p>{impact}</p>

    <div class='section-label'>WHAT TO DO</div>
    <p style='color:{color};'>{action}</p>

    </div>
    """, unsafe_allow_html=True)

# ─── Scenario Lab ─────────────────────────────────────────

def render_scenario_lab(df, model, scaler, feature_cols):

    country = st.selectbox("Country", df["country"].unique(), key="sc")

    inflation = st.slider("Inflation Shock", -10.0, 10.0, 0.0)
    gdp = st.slider("GDP Shock", -10.0, 5.0, 0.0)
    rate = st.slider("Rate Shock", -5.0, 10.0, 0.0)

    if st.button("Run Scenario"):

        row = df[df["country"] == country].iloc[0]

        result = run_scenario(
            model,
            scaler,
            feature_cols,
            row,
            {"inflation": inflation, "gdp": gdp, "interest": rate},
            country,
        )

        st.metric("Before", f"{result['before']:.2%}")
        st.metric("After", f"{result['after']:.2%}")
        st.metric("Delta", f"{result['delta']:.2%}")        
def render_global_risk(df: pd.DataFrame):
    summary = global_risk_summary(df)

    st.markdown("## Global Risk Overview")

    top10 = df.sort_values("risk_score", ascending=False).head(10)

    fig = go.Figure(go.Bar(
        x=top10["risk_score"],
        y=top10["country"],
        orientation="h",
        marker_color=[DECISION_COLORS.get(d) for d in top10["decision"]],
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.write("### Summary")
    st.write(f"Average Risk: {summary['avg_risk']:.2%}")
    st.write(f"Status: {summary['status']}")
