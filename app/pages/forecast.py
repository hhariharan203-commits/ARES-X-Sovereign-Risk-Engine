"""
Forecast Engine — Decision-Driven Risk Projection
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from app.ui import apply_theme, render_sidebar, PLOTLY_THEME
from app.utils import load_dataset, load_model, load_scaler, load_feature_cols, filter_country, get_country_list
from app.intelligence import forecast_risk_trajectory

st.set_page_config(page_title="ARES-X | Forecast", page_icon="⬡", layout="wide")
apply_theme()
render_sidebar()

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("Forward Risk Projection")
st.caption("Model-driven trajectory of sovereign risk over future periods")

# ─────────────────────────────────────────
# LOAD SYSTEM
# ─────────────────────────────────────────
df = load_dataset()
model = load_model()
scaler = load_scaler()
feature_cols = load_feature_cols()
countries = get_country_list(df)

col1, col2 = st.columns([3,1])

with col1:
    country = st.selectbox("Country", countries)

with col2:
    steps = st.selectbox("Forecast Horizon", [1,2,3], index=2)

cdf = filter_country(df, country)

# ─────────────────────────────────────────
# FORECAST
# ─────────────────────────────────────────
forecast_df = forecast_risk_trajectory(
    cdf, model, scaler, feature_cols, country, steps
)

historical = forecast_df[~forecast_df["Projected"]]
projected = forecast_df[forecast_df["Projected"]]

last_hist = historical.iloc[-1]
last_proj = projected.iloc[-1]

delta = last_proj["Risk Score"] - last_hist["Risk Score"]

# ─────────────────────────────────────────
# 🔥 TRAJECTORY CLASSIFICATION
# ─────────────────────────────────────────
if delta > 0.05:
    trend = "Rising Risk"
elif delta < -0.05:
    trend = "Falling Risk"
else:
    trend = "Stable"

# ─────────────────────────────────────────
# 🔥 DECISION (FUTURE-BASED)
# ─────────────────────────────────────────
if trend == "Rising Risk":
    decision = "Preemptive Risk Reduction Required"
    action = "Reduce exposure early. Increase hedging before deterioration materialises."
elif trend == "Falling Risk":
    decision = "Opportunity Building Phase"
    action = "Gradually increase exposure. Monitor confirmation of recovery."
else:
    decision = "Hold Strategy"
    action = "Maintain positioning. No major allocation shifts required."

# ─────────────────────────────────────────
# KPI BLOCK
# ─────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

c1.metric("Current Risk", f"{last_hist['Risk Score']:.3f}")
c2.metric("Forecast Risk", f"{last_proj['Risk Score']:.3f}", delta=f"{delta:+.3f}")
c3.metric("Trend", trend)
c4.metric("Forecast Level", last_proj["Risk Level"])

st.divider()

# ─────────────────────────────────────────
# 🔥 EXECUTIVE DECISION BLOCK
# ─────────────────────────────────────────
st.markdown("## Strategic Outlook")

st.markdown(f"""
### Decision  
➡ **{decision}**

### What is happening  
Risk is expected to move from **{last_hist['Risk Level']}** to **{last_proj['Risk Level']}**,  
indicating a **{trend.lower()} trajectory**.

### Why it matters  
Forward-looking risk shifts directly impact capital allocation, sovereign exposure,
and hedging strategy.

### What to do  
{action}
""")

st.divider()

# ─────────────────────────────────────────
# CHART
# ─────────────────────────────────────────
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=historical["Year"],
    y=historical["Risk Score"],
    mode="lines+markers",
    name="Historical"
))

fig.add_trace(go.Scatter(
    x=[historical["Year"].iloc[-1]] + projected["Year"].tolist(),
    y=[historical["Risk Score"].iloc[-1]] + projected["Risk Score"].tolist(),
    mode="lines+markers",
    name="Projected",
    line=dict(dash="dash")
))

fig.update_layout(**PLOTLY_THEME, height=400)

st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────
# TABLE
# ─────────────────────────────────────────
st.markdown("## Forecast Detail")

display = forecast_df.copy()
display["Type"] = display["Projected"].map({True: "Projected", False: "Historical"})

st.dataframe(display[["Year","Type","Risk Score","Risk Level"]], use_container_width=True)
