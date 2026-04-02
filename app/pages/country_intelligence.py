"""
Country Intelligence — Decision Core Screen
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from ui import apply_theme

# ─────────────────────────────────────────────
def render(system):

    apply_theme()

    df = system["df"]
    model = system["model"]
    scaler = system["scaler"]
    features = system["features"]

    from intelligence import compute_risk_intelligence

    countries = sorted(df["country"].unique())
    country = st.session_state["country"]

    # ─────────────────────────────────────────
    # SELECT DATA
    # ─────────────────────────────────────────
    cdf = df[df["country"] == country].sort_values("year")
    row = cdf.iloc[-1]

    intel = compute_risk_intelligence(
        row, model, scaler, features, country, row.get("year")
    )

    # ─────────────────────────────────────────
    # HEADER
    # ─────────────────────────────────────────
    st.title(f"{country} — Risk Intelligence")

    # ─────────────────────────────────────────
    # 🔥 DECISION BLOCK (WOW SECTION)
    # ─────────────────────────────────────────
    st.markdown("## Strategic Decision")

    st.metric("Risk Score", round(intel.risk_score, 3))
    st.metric("Risk Level", intel.risk_level)

    st.markdown(f"""
### Decision  
➡ **{intel.decision}**

### Confidence  
{round(intel.confidence,2)}

---
""")

    # ─────────────────────────────────────────
    # STORYTELLING
    # ─────────────────────────────────────────
    st.markdown("## What is happening")
    st.write(intel.reasoning)

    st.markdown("## Why it matters")
    st.write(
        "Macro instability directly impacts capital allocation, FX exposure, "
        "and sovereign risk positioning."
    )

    st.markdown("## What to do")
    st.write(intel.action)

    st.divider()

    # ─────────────────────────────────────────
    # FEATURE DRIVERS
    # ─────────────────────────────────────────
    st.markdown("## Key Drivers")

    drivers = pd.DataFrame(intel.top_drivers, columns=["Feature", "Importance"])

    fig = px.bar(
        drivers.head(10),
        x="Importance",
        y="Feature",
        orientation="h"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────
    # HISTORY
    # ─────────────────────────────────────────
    st.markdown("## Risk History")

    scores = []

    for _, r in cdf.iterrows():
        i = compute_risk_intelligence(r, model, scaler, features, country)
        scores.append(i.risk_score)

    cdf["risk"] = scores

    fig2 = px.line(cdf, x="year", y="risk")

    st.plotly_chart(fig2, use_container_width=True)
