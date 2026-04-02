"""
Global Risk — Decision Intelligence View
"""

import streamlit as st
import plotly.express as px
import pandas as pd

from app.ui import apply_theme

# ─────────────────────────────────────────────
def render(system):

    apply_theme()

    df = system["df"]
    model = system["model"]
    scaler = system["scaler"]
    features = system["features"]

    from app.intelligence import compute_global

    gdf = compute_global(df, model, scaler, features)

    # ─────────────────────────────────────────
    # HEADER
    # ─────────────────────────────────────────
    st.title("Global Sovereign Risk")

    st.markdown("""
### What is happening
Global macro conditions are being evaluated across all countries using the trained model.

### Why it matters
Cross-country instability drives:
- capital flows
- currency volatility
- sovereign risk exposure

### What to do
Focus on highest-risk economies and adjust exposure accordingly.
""")

    # ─────────────────────────────────────────
    # KPIs
    # ─────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    col1.metric("Countries", len(gdf))
    col2.metric("Avg Risk", round(gdf["Risk Score"].mean(), 3))
    col3.metric("Highest Risk", gdf.iloc[0]["Country"])

    st.divider()

    # ─────────────────────────────────────────
    # DECISION LAYER (NEW)
    # ─────────────────────────────────────────
    top = gdf.head(3)

    st.markdown("## Key Risk Signals")

    for _, row in top.iterrows():
        st.markdown(f"""
**{row['Country']}**
- Risk Score: {round(row['Risk Score'],3)}
- Level: {row['Risk Level']}
- Decision: {row['Decision']}
""")

    # ─────────────────────────────────────────
    # MAP
    # ─────────────────────────────────────────
    fig = px.choropleth(
        gdf,
        locations="Country",
        locationmode="country names",
        color="Risk Score",
        hover_data=["Risk Level"],
        color_continuous_scale="Reds"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────
    # TABLE
    # ─────────────────────────────────────────
    st.markdown("## Full Ranking")

    st.dataframe(gdf, use_container_width=True)
