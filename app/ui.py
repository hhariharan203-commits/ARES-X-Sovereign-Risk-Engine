"""
ui.py — Elite UI Layer (Decision-first interface)
"""

import streamlit as st

# ─────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────
ARES_CSS = """
<style>
body { background-color:#0a0c10; color:#e8eaf0; font-family: 'IBM Plex Sans', sans-serif; }

.ares-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #e8eaf0;
}

.ares-sub {
    font-size: 0.9rem;
    color: #8892a4;
}

.ares-card {
    background:#111318;
    padding:1rem;
    border-radius:6px;
    border:1px solid #1e2433;
}

.badge {
    padding:4px 8px;
    border-radius:4px;
    font-size:0.7rem;
    font-family:monospace;
}

.low {background:#2ed57322;color:#2ed573;}
.moderate {background:#00d4ff22;color:#00d4ff;}
.elevated {background:#ffd32a22;color:#ffd32a;}
.high {background:#ff6b3522;color:#ff6b35;}
.critical {background:#ff2d5522;color:#ff2d55;}
</style>
"""

def apply_theme():
    st.markdown(ARES_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR (SMART NAVIGATION)
# ─────────────────────────────────────────────
def render_sidebar(system):

    df = system["df"]
    countries = sorted(df["country"].unique())

    with st.sidebar:

        st.markdown("### ⬡ ARES-X")
        st.caption("Decision Intelligence Engine")

        st.divider()

        # Country selector (GLOBAL CONTROL)
        country = st.selectbox("Select Country", countries)
        st.session_state["country"] = country

        st.divider()

        # Navigation (STATE DRIVEN)
        views = [
            "Home",
            "Global Risk",
            "Country Intelligence",
            "Forecast",
            "Regime Detection",
            "Scenario Lab",
            "Portfolio Impact",
            "Explainability",
            "Model Performance",
        ]

        selected = st.radio("Navigation", views)

        st.session_state["view"] = selected

        st.divider()

        st.caption("System: Live Model Active")


# ─────────────────────────────────────────────
# HOME (WOW PAGE)
# ─────────────────────────────────────────────
def render_home(system):

    df = system["df"]
    metrics = system["metrics"]

    st.markdown('<div class="ares-title">ARES-X</div>', unsafe_allow_html=True)
    st.markdown('<div class="ares-sub">Sovereign Risk Decision Intelligence</div>', unsafe_allow_html=True)

    st.divider()

    # ── KEY SYSTEM METRICS ──
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Countries", df["country"].nunique())
    col2.metric("Observations", len(df))
    col3.metric("Features", metrics.get("n_features", "-"))
    col4.metric("Model AUC", round(metrics.get("roc_auc", 0), 3))

    st.divider()

    # ── STORYTELLING ──
    st.markdown("### What this system does")

    st.markdown("""
- Converts macroeconomic data → **risk scores**
- Translates model output → **investment decisions**
- Simulates shocks → **future risk**
- Evaluates exposure → **portfolio impact**
""")

    st.markdown("### Why it matters")

    st.markdown("""
Traditional dashboards show data.  
ARES-X tells you **what to do with it**.
""")

    st.markdown("### What to do next")

    st.markdown("""
👉 Go to **Country Intelligence**  
👉 Analyze a specific economy  
👉 Act on model-driven decisions  
""")
