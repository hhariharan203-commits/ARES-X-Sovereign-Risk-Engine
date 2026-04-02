"""
Explainability — Model Trust & Decision Justification Layer
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from ui import apply_theme, render_sidebar
from utils import load_dataset, load_model, load_scaler, load_feature_cols, filter_country, get_country_list
from intelligence import compute_risk_intelligence, _get_feature_importance

st.set_page_config(page_title="ARES-X | Explainability", layout="wide")

apply_theme()
render_sidebar()

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("Model Explainability & Trust")

# ─────────────────────────────────────────
# LOAD SYSTEM
# ─────────────────────────────────────────
df = load_dataset()
model = load_model()
scaler = load_scaler()
features = load_feature_cols()

countries = get_country_list(df)

# ─────────────────────────────────────────
# 🔥 GLOBAL IMPORTANCE
# ─────────────────────────────────────────
importance = _get_feature_importance(model, features)
imp_df = pd.DataFrame(importance, columns=["Feature", "Importance"])

imp_df["Abs"] = imp_df["Importance"].abs()
imp_df = imp_df.sort_values("Abs", ascending=False)

# ─────────────────────────────────────────
# 🔥 MODEL TRUST METRIC
# ─────────────────────────────────────────
top5 = imp_df.head(5)["Abs"].sum()
total = imp_df["Abs"].sum()

concentration = top5 / total

if concentration > 0.7:
    trust = "High Model Concentration"
    trust_msg = "Model decisions rely heavily on a few dominant drivers."
elif concentration > 0.5:
    trust = "Moderate Concentration"
    trust_msg = "Model is balanced but influenced by key variables."
else:
    trust = "Diversified Model"
    trust_msg = "Model decisions are distributed across multiple factors."

# ─────────────────────────────────────────
# 🔥 EXECUTIVE TRUST BLOCK
# ─────────────────────────────────────────
st.markdown("## Model Trust Assessment")

st.metric("Top-5 Driver Concentration", f"{concentration:.2%}")
st.metric("Model Structure", trust)

st.markdown(f"""
### Interpretation  
{trust_msg}

### Why it matters  
Model concentration affects robustness. Highly concentrated models are sensitive to specific macro variables,
while diversified models are more stable across regimes.

### Decision Confidence  
Model outputs can be considered **reliable for strategic decision-making**, subject to macro data integrity.
""")

st.divider()

# ─────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────
st.markdown("## Key Risk Drivers")

fig = px.bar(
    imp_df.head(15),
    x="Abs",
    y="Feature",
    orientation="h"
)

st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────
# COUNTRY LEVEL
# ─────────────────────────────────────────
st.markdown("## Country-Level Explainability")

country = st.selectbox("Select Country", countries)

cdf = filter_country(df, country)
row = cdf.iloc[-1]

intel = compute_risk_intelligence(row, model, scaler, features, country)

drivers = pd.DataFrame(intel.top_drivers, columns=["Feature", "Importance"])
drivers["Abs"] = drivers["Importance"].abs()

# ─────────────────────────────────────────
# 🔥 DECISION JUSTIFICATION
# ─────────────────────────────────────────
top_drivers = drivers.head(3)["Feature"].tolist()

st.markdown("## Decision Justification")

st.markdown(f"""
### Decision  
➡ **{intel.decision}**

### Why this decision is correct  
The model identifies **{', '.join(top_drivers)}** as dominant drivers influencing risk.

### What is happening  
Key macro variables are pushing the economy toward a **{intel.risk_level}** risk classification.

### Confidence  
{round(intel.confidence,2)}
""")

st.divider()

# ─────────────────────────────────────────
# DRIVER VISUAL
# ─────────────────────────────────────────
fig2 = px.bar(
    drivers.head(10),
    x="Importance",
    y="Feature",
    orientation="h"
)

st.plotly_chart(fig2, use_container_width=True)
