"""
Scenario Lab — Decision Simulation Engine
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from ui import apply_theme, render_sidebar, PLOTLY_THEME
from utils import load_dataset, load_model, load_scaler, load_feature_cols, filter_country, get_country_list
from intelligence import compute_scenario_intelligence

st.set_page_config(page_title="ARES-X | Scenario Lab", layout="wide")

apply_theme()
render_sidebar()

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("Macro Scenario Simulation Engine")

# ─────────────────────────────────────────
# LOAD SYSTEM
# ─────────────────────────────────────────
df = load_dataset()
model = load_model()
scaler = load_scaler()
features = load_feature_cols()

countries = get_country_list(df)

col1, col2 = st.columns([3,1])

with col1:
    country = st.selectbox("Country", countries)

with col2:
    year = st.selectbox("Year", sorted(df["year"].dropna().unique(), reverse=True))

cdf = filter_country(df, country)
row = cdf[cdf["year"] == year].iloc[0]

# ─────────────────────────────────────────
# SHOCK INPUTS
# ─────────────────────────────────────────
st.markdown("## Apply Macro Shock")

shock = {}

gdp = st.slider("GDP Growth", -10.0, 10.0, float(row["gdp_growth_lag1"]))
inf = st.slider("Inflation", 0.0, 20.0, float(row["inflation_lag1"]))
unemp = st.slider("Unemployment", 0.0, 25.0, float(row["unemployment_lag1"]))

shock["gdp_growth_lag1"] = gdp
shock["inflation_lag1"] = inf
shock["unemployment_lag1"] = unemp

# ─────────────────────────────────────────
# COMPUTE
# ─────────────────────────────────────────
scenario = compute_scenario_intelligence(
    row, shock, model, scaler, features, country, year
)

delta = scenario.delta_risk

# ─────────────────────────────────────────
# 🔥 SEVERITY CLASSIFICATION
# ─────────────────────────────────────────
if delta > 0.15:
    severity = "Systemic Risk Shock"
elif delta > 0.08:
    severity = "Severe Deterioration"
elif delta > 0.03:
    severity = "Moderate Risk Increase"
elif delta < -0.05:
    severity = "Improvement Scenario"
else:
    severity = "Neutral Impact"

# ─────────────────────────────────────────
# 🔥 DECISION ENGINE
# ─────────────────────────────────────────
if delta > 0.1:
    decision = "Immediate De-Risking Required"
    action = "Exit or hedge exposure aggressively"
elif delta > 0.05:
    decision = "Reduce Exposure"
    action = "Cut positions and increase defensive assets"
elif delta < -0.05:
    decision = "Opportunity Signal"
    action = "Consider increasing allocation gradually"
else:
    decision = "No Action"
    action = "Maintain current positioning"

# ─────────────────────────────────────────
# KPI
# ─────────────────────────────────────────
c1, c2, c3 = st.columns(3)

c1.metric("Baseline Risk", round(scenario.baseline.risk_score,3))
c2.metric("Shocked Risk", round(scenario.shocked.risk_score,3), delta=round(delta,3))
c3.metric("Severity", severity)

st.divider()

# ─────────────────────────────────────────
# 🔥 WAR ROOM DECISION BLOCK
# ─────────────────────────────────────────
st.markdown("## Strategic Scenario Outcome")

st.markdown(f"""
### Decision  
➡ **{decision}**

### What is happening  
Risk changes by **{round(delta,3)}**, indicating **{severity}**.

### Why it matters  
Macro shocks can rapidly reprice sovereign risk and trigger capital flight,
FX volatility, and bond repricing.

### What to do  
{action}
""")

st.divider()

# ─────────────────────────────────────────
# VISUAL
# ─────────────────────────────────────────
fig = go.Figure()

fig.add_bar(name="Baseline", x=["Risk"], y=[scenario.baseline.risk_score])
fig.add_bar(name="Shocked", x=["Risk"], y=[scenario.shocked.risk_score])

fig.update_layout(**PLOTLY_THEME, barmode="group")

st.plotly_chart(fig, use_container_width=True)
