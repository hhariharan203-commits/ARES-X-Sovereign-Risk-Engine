"""
Portfolio Impact — Institutional Decision Engine
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from ui import apply_theme, render_sidebar
from utils import load_dataset, load_model, load_scaler, load_feature_cols
from intelligence import compute_global_risk_table, compute_portfolio_intelligence

st.set_page_config(page_title="ARES-X | Portfolio Impact", layout="wide")

apply_theme()
render_sidebar()

st.title("Portfolio Risk & Capital Allocation Engine")

# ─────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────
df = load_dataset()
model = load_model()
scaler = load_scaler()
feature_cols = load_feature_cols()

gdf = compute_global_risk_table(df, model, scaler, feature_cols)

countries = gdf["Country"].tolist()

# ─────────────────────────────────────────
# PORTFOLIO INPUT
# ─────────────────────────────────────────
selected = st.multiselect("Select Countries", countries, default=countries[:5])

exposures = {}
for c in selected:
    exposures[c] = st.number_input(f"{c} Exposure (%)", 0.0, 100.0, 10.0)

exposures = {k: v for k, v in exposures.items() if v > 0}

if not exposures:
    st.stop()

portfolio = compute_portfolio_intelligence(exposures, gdf)

st.divider()

# ─────────────────────────────────────────
# KPI
# ─────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Portfolio Risk", round(portfolio.weighted_risk,4))
c2.metric("Stress Level", portfolio.stress_level)
c3.metric("VaR ($100)", round(portfolio.var_estimate,2))

# ─────────────────────────────────────────
# 🔥 RISK GOVERNANCE LOGIC
# ─────────────────────────────────────────
high_risk = portfolio.country_contributions[
    portfolio.country_contributions["Risk Score"] > 0.7
]

concentration = portfolio.country_contributions[
    portfolio.country_contributions["Exposure (%)"] > 25
]

alerts = []

if len(high_risk) > 0:
    alerts.append("High exposure to risky economies detected")

if len(concentration) > 0:
    alerts.append("Portfolio concentration risk detected")

if portfolio.weighted_risk > 0.75:
    alerts.append("Portfolio in CRITICAL risk zone")

# ─────────────────────────────────────────
# 🔥 ALERT SYSTEM
# ─────────────────────────────────────────
st.markdown("## Risk Alerts")

if alerts:
    for a in alerts:
        st.error(f"⚠ {a}")
else:
    st.success("Portfolio risk within acceptable range")

st.divider()

# ─────────────────────────────────────────
# 🔥 REBALANCING ENGINE
# ─────────────────────────────────────────
st.markdown("## Recommended Rebalancing")

reduce = high_risk["Country"].tolist()
increase = portfolio.country_contributions.sort_values("Risk Score").head(3)["Country"].tolist()

st.markdown(f"""
### Reduce Exposure  
{', '.join(reduce) if reduce else "None"}

### Increase Allocation  
{', '.join(increase)}
""")

# ─────────────────────────────────────────
# 🔥 CIO DECISION BLOCK
# ─────────────────────────────────────────
st.markdown("## Executive Decision")

if portfolio.weighted_risk > 0.8:
    decision = "Aggressively de-risk portfolio"
elif portfolio.weighted_risk > 0.6:
    decision = "Rebalance and reduce exposure"
else:
    decision = "Portfolio within acceptable limits"

st.markdown(f"➡ **{decision}**")

# ─────────────────────────────────────────
# VISUAL
# ─────────────────────────────────────────
fig = px.pie(
    portfolio.country_contributions,
    names="Country",
    values="Exposure (%)"
)

st.plotly_chart(fig, use_container_width=True)
