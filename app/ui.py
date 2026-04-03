"""
ui.py — Shared UI components, cards, layout blocks, reusable sections.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from utils import regime_color, risk_color, decision_color, REGIME_COLORS


# ── Theme constants ───────────────────────────────────────────────────────────

BG_DARK       = "#0D1117"
BG_CARD       = "#161B22"
BG_CARD2      = "#1C2333"
ACCENT_BLUE   = "#00B4D8"
ACCENT_CYAN   = "#00E5FF"
ACCENT_GREEN  = "#00E676"
ACCENT_AMBER  = "#FFC107"
ACCENT_RED    = "#FF5252"
TEXT_PRIMARY  = "#E6EDF3"
TEXT_MUTED    = "#8B949E"
GRID_COLOR    = "#21262D"

PLOTLY_LAYOUT = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    font          = dict(color=TEXT_PRIMARY, family="Inter, sans-serif"),
    xaxis         = dict(gridcolor=GRID_COLOR, color=TEXT_MUTED, showgrid=True),
    yaxis         = dict(gridcolor=GRID_COLOR, color=TEXT_MUTED, showgrid=True),
    legend        = dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_MUTED)),
    margin        = dict(l=10, r=10, t=40, b=10),
)


def apply_plotly_style(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ── Global CSS ────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0D1117;
        color: #E6EDF3;
    }

    .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0D1117;
        border-right: 1px solid #21262D;
    }
    [data-testid="stSidebar"] .css-1d391kg { padding-top: 1rem; }

    /* Metric cards */
    .kpi-card {
        background: #161B22;
        border: 1px solid #21262D;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .kpi-label {
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        color: #8B949E;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00E5FF;
        line-height: 1.1;
    }
    .kpi-delta {
        font-size: 0.8rem;
        color: #8B949E;
        margin-top: 0.2rem;
    }

    /* Insight card */
    .insight-card {
        background: #161B22;
        border: 1px solid #21262D;
        border-left: 3px solid #00E5FF;
        border-radius: 8px;
        padding: 1rem 1.4rem;
        margin-bottom: 0.8rem;
    }
    .insight-title {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        color: #00E5FF;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .insight-body {
        font-size: 0.9rem;
        color: #C9D1D9;
        line-height: 1.6;
    }

    /* Decision badge */
    .badge-buy       { background:#00E676; color:#0D1117; padding:4px 16px; border-radius:20px; font-weight:700; font-size:1rem; }
    .badge-hold      { background:#FFC107; color:#0D1117; padding:4px 16px; border-radius:20px; font-weight:700; font-size:1rem; }
    .badge-defensive { background:#FF5252; color:#fff;    padding:4px 16px; border-radius:20px; font-weight:700; font-size:1rem; }

    /* Section header */
    .section-header {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        color: #8B949E;
        text-transform: uppercase;
        margin: 1.5rem 0 0.8rem 0;
        border-bottom: 1px solid #21262D;
        padding-bottom: 0.4rem;
    }

    /* Regime pill */
    .regime-pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Streamlit overrides */
    .stSelectbox > div > div { background: #161B22; border: 1px solid #21262D; }
    .stSlider > div > div > div { background: #00E5FF; }
    div[data-testid="metric-container"] { background: #161B22; border-radius: 8px; padding: 0.8rem; }
    </style>
    """, unsafe_allow_html=True)


# ── Component builders ────────────────────────────────────────────────────────

def kpi_card(label: str, value: str, delta: str = "", color: str = ACCENT_CYAN):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color}">{value}</div>
        <div class="kpi-delta">{delta}</div>
    </div>
    """, unsafe_allow_html=True)


def insight_card(title: str, body: str, accent: str = ACCENT_CYAN):
    st.markdown(f"""
    <div class="insight-card" style="border-left-color:{accent}">
        <div class="insight-title">{title}</div>
        <div class="insight-body">{body}</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(text: str):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def decision_badge(decision: str):
    cls = f"badge-{decision.lower()}"
    st.markdown(f'<span class="{cls}">{decision}</span>', unsafe_allow_html=True)


def regime_pill(regime: str):
    color = regime_color(regime)
    st.markdown(
        f'<span class="regime-pill" style="background:{color}22; color:{color}; border:1px solid {color}44">{regime}</span>',
        unsafe_allow_html=True
    )


def page_header(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div style="margin-bottom:1.5rem">
        <h1 style="font-size:1.8rem;font-weight:700;color:#E6EDF3;margin:0">{title}</h1>
        {"" if not subtitle else f'<p style="color:#8B949E;font-size:0.9rem;margin-top:0.3rem">{subtitle}</p>'}
    </div>
    """, unsafe_allow_html=True)


# ── Chart builders ────────────────────────────────────────────────────────────

def gauge_chart(value: float, title: str = "Risk Score", max_val: float = 100) -> go.Figure:
    color = ACCENT_GREEN if value < 40 else (ACCENT_AMBER if value < 70 else ACCENT_RED)
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = value,
        title = {"text": title, "font": {"color": TEXT_MUTED, "size": 13}},
        number = {"font": {"color": color, "size": 36}},
        gauge = {
            "axis":  {"range": [0, max_val], "tickcolor": TEXT_MUTED, "tickfont": {"color": TEXT_MUTED}},
            "bar":   {"color": color},
            "bgcolor": BG_CARD2,
            "bordercolor": GRID_COLOR,
            "steps": [
                {"range": [0, 40],  "color": "#0D3326"},
                {"range": [40, 70], "color": "#2C2000"},
                {"range": [70, 100],"color": "#2C0D0D"},
            ],
            "threshold": {"line": {"color": "#fff", "width": 2}, "thickness": 0.75, "value": value},
        },
    ))
    fig.update_layout(**{**PLOTLY_LAYOUT, "height": 240, "margin": dict(l=20, r=20, t=40, b=10)})
    return fig


def line_chart(df: pd.DataFrame, x: str, y_cols: list, title: str = "", colors: list = None) -> go.Figure:
    default_colors = [ACCENT_CYAN, ACCENT_GREEN, ACCENT_AMBER, ACCENT_RED, "#B388FF"]
    fig = go.Figure()
    for i, col in enumerate(y_cols):
        c = (colors[i] if colors and i < len(colors) else default_colors[i % len(default_colors)])
        fig.add_trace(go.Scatter(
            x=df[x], y=df[col], name=col,
            line=dict(color=c, width=2),
            mode="lines",
        ))
    fig.update_layout(**{**PLOTLY_LAYOUT, "title": {"text": title, "font": {"size": 14, "color": TEXT_MUTED}}, "height": 340})
    return fig


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str = "", color_col: str = None) -> go.Figure:
    if color_col and color_col in df.columns:
        colors = df[color_col].map(lambda v: ACCENT_GREEN if v >= 2 else (ACCENT_AMBER if v >= 0 else ACCENT_RED)).tolist()
    else:
        colors = ACCENT_CYAN

    fig = go.Figure(go.Bar(
        x=df[x], y=df[y],
        marker_color=colors,
        marker_line_width=0,
    ))
    fig.update_layout(**{**PLOTLY_LAYOUT, "title": {"text": title, "font": {"size": 14, "color": TEXT_MUTED}}, "height": 380})
    return fig


def heatmap_chart(z_data, x_labels, y_labels, title: str = "") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=z_data,
        x=x_labels,
        y=y_labels,
        colorscale=[[0, "#1A1A2E"], [0.5, "#0077B6"], [1, "#00E5FF"]],
        showscale=True,
        colorbar=dict(tickfont=dict(color=TEXT_MUTED)),
    ))
    fig.update_layout(**{
        **PLOTLY_LAYOUT,
        "title": {"text": title, "font": {"size": 14, "color": TEXT_MUTED}},
        "height": 500,
        "xaxis": {**PLOTLY_LAYOUT["xaxis"], "tickangle": -35},
    })
    return fig


def pie_chart(labels: list, values: list, title: str = "") -> go.Figure:
    colors = [ACCENT_CYAN, ACCENT_GREEN, ACCENT_AMBER, ACCENT_RED, "#B388FF", "#FF80AB", "#80D8FF", "#CCFF90"]
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors[:len(labels)], line=dict(color=BG_DARK, width=2)),
        textfont=dict(color=TEXT_PRIMARY, size=11),
    ))
    fig.update_layout(**{**PLOTLY_LAYOUT, "title": {"text": title, "font": {"size": 14, "color": TEXT_MUTED}}, "height": 340})
    return fig


def sidebar_logo():
    st.sidebar.markdown("""
    <div style="text-align:center;padding:1rem 0 1.5rem 0">
        <div style="font-size:1.6rem;font-weight:800;color:#00E5FF;letter-spacing:0.15em">ARES-X</div>
        <div style="font-size:0.65rem;color:#8B949E;letter-spacing:0.2em;margin-top:0.2rem">MACRO INTELLIGENCE TERMINAL</div>
    </div>
    """, unsafe_allow_html=True)