"""
Regime Detection — Macro Structure Intelligence
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from ui import apply_theme, render_sidebar, PLOTLY_THEME
from utils import load_dataset, load_model, load_scaler, load_feature_cols, get_latest_row_per_country
from intelligence import compute_risk_intelligence

st.set_page_config(page_title="ARES-X | Regime Detection", layout="wide")

apply_theme()
render_sidebar()

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("Global Macro Regime Intelligence")

# ─────────────────────────────────────────
# LOAD SYSTEM
# ─────────────────────────────────────────
df = load_dataset()
model = load_model()
scaler = load_scaler()
features = load_feature_cols()

# ─────────────────────────────────────────
# COMPUTE REGIMES
# ─────────────────────────────────────────
latest = get_latest_row_per_country(df)

records = []

for _, row in latest.iterrows():
    country = row["country"]
    intel = compute_risk_intelligence(row, model, scaler, features, country)

    records.append({
        "Country": country,
        "Regime": intel.regime,
        "Risk Score": intel.risk_score,
        "Risk Level": intel.risk_level
    })

regime_df = pd.DataFrame(records)

# ─────────────────────────────────────────
# 🔥 GLOBAL INSIGHT (KEY ADDITION)
# ─────────────────────────────────────────
stress_regimes = ["Crisis Zone", "Stagflation Risk", "Inflation Stress"]

stress_pct = regime_df["Regime"].isin(stress_regimes).mean()

if stress_pct > 0.5:
    global_state = "Global Macro Stress"
    decision = "Defensive Allocation Strategy"
    action = "Reduce EM exposure, increase safe assets (USD, Gold, Bonds)"
elif stress_pct > 0.3:
    global_state = "Moderate Instability"
    decision = "Selective Allocation"
    action = "Focus on stable economies, avoid fragile regimes"
else:
    global_state = "Stable Global Backdrop"
    decision = "Growth Allocation"
    action = "Increase risk exposure and capital deployment"

# ─────────────────────────────────────────
# 🔥 EXECUTIVE BLOCK
# ─────────────────────────────────────────
st.markdown("## Global Decision Signal")

st.metric("Stress Regime Share", f"{stress_pct:.2%}")
st.metric("Global State", global_state)

st.markdown(f"""
### Decision  
➡ **{decision}**

### What is happening  
{round(stress_pct*100,1)}% of countries are in stress-driven regimes.

### Why it matters  
Macro regimes define the structural environment in which risk evolves.  
High concentration in stress regimes signals systemic fragility.

### What to do  
{action}
""")

st.divider()

# ─────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────
st.markdown("## Regime Distribution")

dist = regime_df["Regime"].value_counts().reset_index()
dist.columns = ["Regime", "Count"]

fig = px.bar(dist, x="Regime", y="Count")

st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────
# TABLE
# ─────────────────────────────────────────
st.markdown("## Country Classification")

st.dataframe(
    regime_df.sort_values("Risk Score", ascending=False),
    use_container_width=True
)
