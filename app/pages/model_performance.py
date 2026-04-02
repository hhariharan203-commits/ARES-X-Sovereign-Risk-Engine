"""
Model Performance — Decision Validation Layer
"""

import streamlit as st
import pandas as pd

from app.ui import apply_theme, render_sidebar
from app.utils import load_model_metrics

st.set_page_config(page_title="ARES-X | Model Performance", layout="wide")

apply_theme()
render_sidebar()

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("Model Validation & Reliability")

metrics = load_model_metrics()

auc = metrics.get("roc_auc", 0)
f1 = metrics.get("f1", 0)
cv_auc = metrics.get("cv_auc", 0)

# ─────────────────────────────────────────
# 🔥 MODEL APPROVAL LOGIC
# ─────────────────────────────────────────
if auc > 0.85 and f1 > 0.75:
    approval = "Approved for Decision Use"
    risk_level = "Low Model Risk"
elif auc > 0.75:
    approval = "Conditionally Approved"
    risk_level = "Moderate Model Risk"
else:
    approval = "Not Reliable"
    risk_level = "High Model Risk"

# ─────────────────────────────────────────
# KPI
# ─────────────────────────────────────────
c1, c2, c3 = st.columns(3)

c1.metric("ROC-AUC", round(auc,4))
c2.metric("F1 Score", round(f1,4))
c3.metric("CV AUC", round(cv_auc,4))

st.divider()

# ─────────────────────────────────────────
# 🔥 EXECUTIVE VALIDATION BLOCK
# ─────────────────────────────────────────
st.markdown("## Model Decision Status")

st.markdown(f"""
### Approval  
➡ **{approval}**

### Model Risk Level  
{risk_level}

### What is happening  
The model demonstrates a discrimination capability (AUC) of **{round(auc,3)}**
and classification balance (F1) of **{round(f1,3)}**.

### Why it matters  
Model errors directly translate into incorrect capital allocation decisions.

- False positives → unnecessary de-risking  
- False negatives → hidden exposure to crisis  

### What to do  
{"Model can be deployed in production decision workflows." if approval == "Approved for Decision Use"
else "Use model with caution and combine with analyst judgement." if approval == "Conditionally Approved"
else "Model should not be used for decision-making until improved."}
""")

st.divider()

# ─────────────────────────────────────────
# 🔥 ERROR INTERPRETATION
# ─────────────────────────────────────────
st.markdown("## Risk of Model Failure")

st.markdown("""
### False Positive Risk  
Model flags risk where none exists → leads to missed investment opportunities.

### False Negative Risk  
Model misses risk → leads to capital loss and exposure.

### Practical Interpretation  
This model should be used as a **decision support system**, not a fully autonomous system.
""")
