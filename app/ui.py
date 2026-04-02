"""
Sovereign Risk Decision Intelligence Engine
UI Components — Production Grade
Institutional — BlackRock / Goldman Sachs
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

from intelligence import (
    DECISION_COLORS,
    REGIME_COLORS,
    risk_to_decision,
    risk_to_regime,
    compute_top_drivers,
    global_risk_summary,
    run_scenario,
)


# ─── Design Tokens ────────────────────────────────────────────────────────────

DARK_BG      = "#0A0C14"
CARD_BG      = "#0F1220"
BORDER_COLOR = "#1E2235"
TEXT_PRIMARY = "#E8EAF0"
TEXT_MUTED   = "#6B7190"
ACCENT_BLUE  = "#3B82F6"
ACCENT_TEAL  = "#00C896"


# ─── CSS Injection ────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #0A0C14;
        color: #E8EAF0;
    }

    [data-testid="stSidebar"] {
        background: #070910 !important;
        border-right: 1px solid #1A1D2E;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        letter-spacing: 0.08em;
        color: #6B7190 !important;
        cursor: pointer;
        transition: color 0.2s;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        color: #E8EAF0 !important;
    }

    [data-testid="stMetric"] {
        background: #0F1220;
        border: 1px solid #1E2235;
        border-radius: 6px;
        padding: 16px 20px;
    }
    [data-testid="stMetric"] label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        letter-spacing: 0.12em;
        color: #6B7190 !important;
        text-transform: uppercase;
    }
    [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 28px !important;
        font-weight: 600;
        color: #E8EAF0 !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
    }

    [data-testid="stSelectbox"] select,
    .stSelectbox > div > div {
        background: #0F1220 !important;
        border: 1px solid #1E2235 !important;
        color: #E8EAF0 !important;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        border-radius: 4px;
    }

    [data-testid="stSlider"] { padding: 8px 0; }
    .stSlider label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.08em;
        color: #6B7190;
        text-transform: uppercase;
    }

    .block-container {
        padding: 2rem 2.5rem 2rem 2.5rem !important;
        max-width: 1400px;
    }

    hr { border-color: #1E2235 !important; margin: 1.5rem 0; }

    #MainMenu, footer, header { visibility: hidden; }

    .js-plotly-plot { border-radius: 6px; }

    .risk-card {
        background: #0F1220;
        border: 1px solid #1E2235;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }
    .section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        letter-spacing: 0.15em;
        color: #6B7190;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .headline-number {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 48px;
        font-weight: 600;
        line-height: 1;
    }
    .stAlert {
        background: #0F1220 !important;
        border: 1px solid #1E2235 !important;
        border-radius: 6px;
    }
    </style>
    """, unsafe_allow_html=True)


# ─── Shared Plotly Theme ──────────────────────────────────────────────────────

def _layout(title: str = "", height: int = 360) -> dict:
    return dict(
        title=dict(
            text=title,
            font=dict(family="IBM Plex Mono", size=12, color="#6B7190"),
            x=0, xanchor="left",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans", color="#8B8FA8", size=11),
        height=height,
        margin=dict(l=0, r=0, t=36, b=0),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            color="#4A4F6A",
            tickfont=dict(family="IBM Plex Mono", size=10),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#1A1D2E",
            zeroline=False,
            color="#4A4F6A",
            tickfont=dict(family="IBM Plex Mono", size=10),
        ),
        showlegend=False,
    )


# ─── Sidebar ─────────────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame) -> str:
    with st.sidebar:
        st.markdown("""
        <div style='padding: 24px 0 32px 0;'>
            <div style='font-family: IBM Plex Mono, monospace; font-size: 10px;
                        letter-spacing: 0.2em; color: #3B82F6; text-transform: uppercase;
                        margin-bottom: 6px;'>SOVEREIGN RISK</div>
            <div style='font-family: IBM Plex Mono, monospace; font-size: 16px;
                        font-weight: 600; color: #E8EAF0; line-height: 1.3;'>
                Decision Intelligence<br>Engine
            </div>
            <div style='font-family: IBM Plex Mono, monospace; font-size: 9px;
                        color: #3A3F5C; margin-top: 8px; letter-spacing: 0.1em;'>
                v2.4.1 · INSTITUTIONAL
            </div>
        </div>
        <hr style='border-color: #1A1D2E; margin-bottom: 24px;'>
        """, unsafe_allow_html=True)

        nav = st.radio(
            "NAVIGATION",
            ["Overview", "Global Risk", "Country Intelligence", "Scenario Lab"],
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color: #1A1D2E; margin: 24px 0;'>", unsafe_allow_html=True)

        total    = len(df)
        exits    = int((df["decision"] == "Exit").sum())
        invests  = int((df["decision"] == "Invest").sum())
        avg_risk = float(df["risk_score"].mean())

        st.markdown(f"""
        <div style='font-family: IBM Plex Mono, monospace;'>
            <div style='font-size: 9px; letter-spacing: 0.15em; color: #3A3F5C;
                        text-transform: uppercase; margin-bottom: 12px;'>LIVE SIGNALS</div>
            <div style='display: flex; justify-content: space-between;
                        padding: 8px 0; border-bottom: 1px solid #14172A;'>
                <span style='font-size: 11px; color: #6B7190;'>Universe</span>
                <span style='font-size: 11px; color: #E8EAF0;'>{total} countries</span>
            </div>
            <div style='display: flex; justify-content: space-between;
                        padding: 8px 0; border-bottom: 1px solid #14172A;'>
                <span style='font-size: 11px; color: #6B7190;'>Avg Risk</span>
                <span style='font-size: 11px; color: #E8EAF0;'>{avg_risk:.2%}</span>
            </div>
            <div style='display: flex; justify-content: space-between;
                        padding: 8px 0; border-bottom: 1px solid #14172A;'>
                <span style='font-size: 11px; color: #FF3B5C;'>Exit Signals</span>
                <span style='font-size: 11px; color: #FF3B5C; font-weight: 600;'>{exits}</span>
            </div>
            <div style='display: flex; justify-content: space-between; padding: 8px 0;'>
                <span style='font-size: 11px; color: #6B7190;'>Invest Signals</span>
                <span style='font-size: 11px; color: #00C896; font-weight: 600;'>{invests}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-family: IBM Plex Mono, monospace; font-size: 9px;
                    color: #2A2F45; text-align: center; letter-spacing: 0.08em;'>
            FOR INSTITUTIONAL USE ONLY<br>NOT FOR PUBLIC DISTRIBUTION
        </div>
        """, unsafe_allow_html=True)

    return nav


# ─── Overview ────────────────────────────────────────────────────────────────

def render_overview(df: pd.DataFrame):
    # Compute summary ONCE and reuse across all sub-components
    summary = global_risk_summary(df)

    st.markdown(f"""
    <div style='margin-bottom: 28px;'>
        <div class='section-label'>EXECUTIVE BRIEFING — GLOBAL SOVEREIGN RISK</div>
        <div style='font-family: IBM Plex Mono, monospace; font-size: 22px;
                    font-weight: 600; color: #E8EAF0; margin-top: 4px;'>
            Market Status:
            <span style='color: {summary["status_color"]};'>{summary["status"]}</span>
        </div>
        <div style='font-family: IBM Plex Sans, sans-serif; font-size: 14px;
                    color: #8B8FA8; margin-top: 8px; max-width: 740px; line-height: 1.6;'>
            {summary["headline"]}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI strip
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("GLOBAL RISK SCORE", f"{summary['avg_risk']:.2%}")
    with c2:
        st.metric("TOTAL UNIVERSE", f"{summary['total_countries']} CTRY")
    with c3:
        exit_n = summary["decision_dist"]["Exit"]
        st.metric("EXIT SIGNALS", str(exit_n),
                  delta="Critical exposure" if exit_n > 5 else "Contained")
    with c4:
        st.metric("INVEST SIGNALS", str(summary["decision_dist"]["Invest"]))
    with c5:
        st.metric("GLOBAL DECISION", summary["decision"])

    st.markdown("<hr>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.6, 1])
    with col_left:
        _render_decision_distribution_chart(df)
    with col_right:
        _render_regime_breakdown(df)

    st.markdown("<hr>", unsafe_allow_html=True)
    _render_risk_spectrum(df)


def _render_decision_distribution_chart(df: pd.DataFrame):
    order = ["Invest", "Hold", "Reduce", "Exit"]
    dist  = (
        df.groupby("decision")["risk_score"]
        .agg(["mean", "count"])
        .reindex(order)
        .reset_index()
    )
    dist.columns = ["decision", "avg_risk", "count"]
    dist = dist.fillna(0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dist["decision"],
        y=dist["count"],
        marker_color=[DECISION_COLORS.get(d, "#8B8FA8") for d in dist["decision"]],
        marker_line_width=0,
        text=dist["count"].astype(int),
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=11, color="#8B8FA8"),
        hovertemplate="<b>%{x}</b><br>Countries: %{y}<extra></extra>",
        width=0.55,
    ))

    layout = _layout("DECISION SIGNAL DISTRIBUTION", height=300)
    layout["xaxis"]["tickfont"] = dict(family="IBM Plex Mono", size=12, color="#E8EAF0")
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_regime_breakdown(df: pd.DataFrame):
    regime_counts = df["regime"].value_counts()
    total         = max(len(df), 1)

    st.markdown("""
    <div class='section-label' style='margin-bottom: 14px;'>REGIME BREAKDOWN</div>
    """, unsafe_allow_html=True)

    for regime, color in REGIME_COLORS.items():
        count = int(regime_counts.get(regime, 0))
        pct   = count / total * 100
        st.markdown(f"""
        <div style='margin-bottom: 12px;'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 4px;'>
                <span style='font-family: IBM Plex Mono, monospace; font-size: 11px;
                             color: {color}; text-transform: uppercase; letter-spacing: 0.08em;'>
                    {regime}
                </span>
                <span style='font-family: IBM Plex Mono, monospace; font-size: 11px; color: #E8EAF0;'>
                    {count} <span style='color: #4A4F6A;'>({pct:.0f}%)</span>
                </span>
            </div>
            <div style='background: #1A1D2E; border-radius: 2px; height: 3px;'>
                <div style='background: {color}; width: {pct:.1f}%; height: 3px;
                             border-radius: 2px; opacity: 0.8;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def _render_risk_spectrum(df: pd.DataFrame):
    st.markdown("""
    <div class='section-label'>RISK SPECTRUM — TOP 30 COUNTRIES (RANKED)</div>
    """, unsafe_allow_html=True)

    sorted_df = df.nlargest(30, "risk_score")
    country_col = "country" if "country" in sorted_df.columns else sorted_df.columns[0]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sorted_df[country_col],
        y=sorted_df["risk_score"],
        marker_color=[DECISION_COLORS.get(d, "#8B8FA8") for d in sorted_df["decision"]],
        marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>Risk: %{y:.2%}<extra></extra>",
    ))

    layout = _layout("", height=240)
    layout["yaxis"]["tickformat"] = ".0%"
    layout["margin"] = dict(l=0, r=0, t=10, b=60)
    layout["xaxis"]["tickangle"]  = -45
    layout["xaxis"]["tickfont"]   = dict(size=9, family="IBM Plex Mono")
    fig.update_layout(**layout)
    fig.add_hline(y=0.50, line_dash="dash", line_color="#FF8C42", line_width=1, opacity=0.4,
                  annotation_text="Reduce", annotation_position="right",
                  annotation_font=dict(family="IBM Plex Mono", size=9, color="#FF8C42"))
    fig.add_hline(y=0.70, line_dash="dash", line_color="#FF3B5C", line_width=1, opacity=0.4,
                  annotation_text="Exit", annotation_position="right",
                  annotation_font=dict(family="IBM Plex Mono", size=9, color="#FF3B5C"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─── Global Risk ─────────────────────────────────────────────────────────────

def render_global_risk(df: pd.DataFrame):
    summary = global_risk_summary(df)

    st.markdown("""
    <div class='section-label'>GLOBAL RISK INTELLIGENCE</div>
    <div style='font-family: IBM Plex Mono, monospace; font-size: 18px;
                font-weight: 600; color: #E8EAF0; margin-bottom: 24px;'>
        Top Risk Exposures &amp; Market Positioning
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.8, 1])
    with col1:
        _render_top10_bar(summary["top10"])
    with col2:
        _render_top10_table(summary["top10"])

    st.markdown("<hr>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    avg = summary["avg_risk"]
    top10 = summary["top10"]
    worst_country = str(top10.iloc[0]["country"]) if len(top10) > 0 else "N/A"
    worst_score   = float(top10.iloc[0]["risk_score"]) if len(top10) > 0 else 0.0

    with c1:
        st.metric("GLOBAL AVG RISK", f"{avg:.2%}",
                  delta="↑ Elevated" if avg > 0.5 else "↓ Contained")
    with c2:
        st.metric("HIGHEST RISK", worst_country, delta=f"{worst_score:.2%}")
    with c3:
        exit_n = summary["decision_dist"]["Exit"]
        st.metric("EXIT MANDATES", str(exit_n),
                  delta="Immediate action required" if exit_n > 0 else "None triggered")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Executive briefing block — uses pre-computed summary, no recompute
    dd = summary["decision_dist"]
    st.markdown(f"""
    <div class='risk-card'>
        <div class='section-label'>GLOBAL RISK ASSESSMENT</div>
        <div style='font-family: IBM Plex Sans, sans-serif; font-size: 14px;
                    color: #C4C8DC; line-height: 1.7; margin-top: 8px;'>
            Global sovereign risk index stands at
            <strong style='color: {summary["status_color"]};'>{summary["avg_risk"]:.2%}</strong>
            — signaling a
            <strong style='color: {summary["status_color"]};'>{summary["status"]}</strong> environment.
            {summary["headline"]}
            Of {summary["total_countries"]} tracked sovereigns,
            <strong style='color: #FF3B5C;'>{dd["Exit"]}</strong> are exit-rated,
            <strong style='color: #FF8C42;'>{dd["Reduce"]}</strong> warrant reduction,
            <strong style='color: #F5C518;'>{dd["Hold"]}</strong> are on hold, and
            <strong style='color: #00C896;'>{dd["Invest"]}</strong> present active opportunity.
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_top10_bar(top10: pd.DataFrame):
    country_col = "country" if "country" in top10.columns else top10.columns[0]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top10["risk_score"],
        y=top10[country_col],
        orientation="h",
        marker_color=[DECISION_COLORS.get(d, "#8B8FA8") for d in top10["decision"]],
        marker_line_width=0,
        text=[f"{s:.2%}" for s in top10["risk_score"]],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=10, color="#8B8FA8"),
        hovertemplate="<b>%{y}</b><br>Risk Score: %{x:.2%}<extra></extra>",
    ))

    layout = _layout("TOP 10 HIGHEST RISK SOVEREIGNS", height=340)
    layout["yaxis"]["autorange"] = "reversed"
    layout["xaxis"]["tickformat"] = ".0%"
    layout["xaxis"]["range"] = [0, 1.15]
    layout["margin"] = dict(l=0, r=60, t=36, b=0)
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_top10_table(top10: pd.DataFrame):
    st.markdown("""
    <div class='section-label' style='margin-bottom: 10px;'>DECISION MATRIX</div>
    """, unsafe_allow_html=True)

    country_col = "country" if "country" in top10.columns else top10.columns[0]

    for _, row in top10.iterrows():
        d_color = DECISION_COLORS.get(str(row["decision"]), "#8B8FA8")
        r_color = REGIME_COLORS.get(str(row.get("regime", "")), "#8B8FA8")
        country = str(row.get(country_col, ""))[:18]
        st.markdown(f"""
        <div style='display: flex; justify-content: space-between; align-items: center;
                    padding: 7px 0; border-bottom: 1px solid #12152A;'>
            <div>
                <div style='font-family: IBM Plex Mono, monospace; font-size: 11px;
                            color: #C4C8DC;'>{country}</div>
                <div style='font-family: IBM Plex Mono, monospace; font-size: 9px;
                            color: {r_color}; letter-spacing: 0.06em;'>{row.get("regime", "—")}</div>
            </div>
            <div style='text-align: right;'>
                <div style='font-family: IBM Plex Mono, monospace; font-size: 12px;
                            color: {d_color}; font-weight: 600;'>
                    {row["risk_score"]:.2%}
                </div>
                <div style='font-family: IBM Plex Mono, monospace; font-size: 9px;
                            color: {d_color}; letter-spacing: 0.1em;'>
                    {row["decision"]}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Country Intelligence ────────────────────────────────────────────────────

def render_country_intelligence(df: pd.DataFrame, model, scaler, feature_cols: List[str]):
    country_col = "country" if "country" in df.columns else df.columns[0]
    countries   = sorted(df[country_col].dropna().unique().tolist())

    st.markdown("""
    <div class='section-label'>COUNTRY INTELLIGENCE MODULE</div>
    <div style='font-family: IBM Plex Mono, monospace; font-size: 16px;
                font-weight: 600; color: #E8EAF0; margin-bottom: 20px;'>
        Single-Country Deep Dive
    </div>
    """, unsafe_allow_html=True)

    selected = st.selectbox(
        "SELECT SOVEREIGN", countries,
        label_visibility="collapsed",
        key="country_select",
    )

    if not selected:
        return

    # ── Pull pre-enriched row — NO recomputation ──────────────────────────────
    mask = df[country_col] == selected
    if not mask.any():
        st.error(f"No data found for {selected}.")
        return

    row = df[mask].iloc[0]

    risk_score   = float(row["risk_score"])
    decision     = str(row["decision"])
    regime       = str(row["regime"])
    conf_label   = str(row["confidence_label"])
    conf_val     = float(row["confidence_value"])
    reasoning    = str(row["reasoning"])
    action       = str(row["action"])
    drivers      = list(row["drivers"])     # pre-computed list of (feat, impact)

    d_color = DECISION_COLORS.get(decision, "#8B8FA8")
    r_color = REGIME_COLORS.get(regime, "#8B8FA8")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("RISK SCORE", f"{risk_score:.2%}")
    with c2:
        st.metric("DECISION", decision)
    with c3:
        st.metric("REGIME", regime)
    with c4:
        st.metric("CONFIDENCE", f"{conf_val:.0%}")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        _render_risk_gauge(risk_score, selected, d_color)
        _render_driver_chart(drivers)

    with col_right:
        _render_decision_block(
            country    = selected,
            risk_score = risk_score,
            decision   = decision,
            regime     = regime,
            conf_label = conf_label,
            conf_val   = conf_val,
            d_color    = d_color,
            r_color    = r_color,
            reasoning  = reasoning,
            action     = action,
        )
        _render_country_positioning(risk_score, df)


def _render_risk_gauge(risk_score: float, country: str, d_color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(risk_score * 100, 1),
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "RISK INDEX",
               "font": {"family": "IBM Plex Mono", "size": 10, "color": "#6B7190"}},
        number={"valueformat": ".1f",
                "font": {"family": "IBM Plex Mono", "size": 32, "color": d_color}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 0,
                "tickcolor": "#1E2235",
                "tickfont": {"family": "IBM Plex Mono", "size": 9, "color": "#4A4F6A"},
            },
            "bar":       {"color": d_color, "thickness": 0.25},
            "bgcolor":   "#0F1220",
            "borderwidth": 0,
            "steps": [
                {"range": [0,   30],  "color": "#0A1A14"},
                {"range": [30,  50],  "color": "#1A1800"},
                {"range": [50,  70],  "color": "#1A1000"},
                {"range": [70,  100], "color": "#1A0A0A"},
            ],
            "threshold": {
                "line":      {"color": d_color, "width": 2},
                "thickness": 0.8,
                "value":     risk_score * 100,
            },
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono", color="#8B8FA8"),
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_driver_chart(drivers: List[Tuple[str, float]]):
    """Render pre-computed driver list — no model calls here."""
    if not drivers:
        return

    st.markdown("""
    <div class='section-label' style='margin-top: 12px;'>TOP RISK DRIVERS</div>
    """, unsafe_allow_html=True)

    names  = [d[0].replace("_", " ").upper()[:22] for d in drivers[:5]]
    values = [d[1] for d in drivers[:5]]
    colors = ["#FF3B5C" if v > 0 else "#00C896" for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>Impact: %{x:.4f}<extra></extra>",
    ))

    layout = _layout("", height=200)
    layout["margin"]                  = dict(l=0, r=10, t=10, b=0)
    layout["xaxis"]["zeroline"]       = True
    layout["xaxis"]["zerolinecolor"]  = "#2A2F45"
    layout["xaxis"]["zerolinewidth"]  = 1
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_decision_block(
    country:    str,
    risk_score: float,
    decision:   str,
    regime:     str,
    conf_label: str,
    conf_val:   float,
    d_color:    str,
    r_color:    str,
    reasoning:  str,
    action:     str,
):
    """
    Renders the full decision intelligence block.
    All strings passed in — nothing recomputed inside.
    Covers: What is happening / Why it matters / What to do.
    """
    icons = {"Invest": "▲", "Hold": "◆", "Reduce": "▼", "Exit": "✕"}
    icon  = icons.get(decision, "◆")

    st.markdown(f"""
    <div class='risk-card' style='border-left: 3px solid {d_color};'>

        <div style='display: flex; justify-content: space-between;
                    align-items: flex-start; margin-bottom: 16px;'>
            <div>
                <div class='section-label'>INVESTMENT DECISION</div>
                <div style='font-family: IBM Plex Mono, monospace; font-size: 28px;
                            font-weight: 600; color: {d_color}; line-height: 1; margin-top: 4px;'>
                    {icon} {decision.upper()}
                </div>
            </div>
            <div style='text-align: right;'>
                <div class='section-label'>CONFIDENCE</div>
                <div style='font-family: IBM Plex Mono, monospace; font-size: 18px;
                            color: #E8EAF0;'>{conf_val:.0%}</div>
                <div style='font-family: IBM Plex Mono, monospace; font-size: 9px;
                            color: #6B7190;'>{conf_label.upper()}</div>
            </div>
        </div>

        <div style='margin-bottom: 14px;'>
            <div class='section-label'>REGIME</div>
            <span style='font-family: IBM Plex Mono, monospace; font-size: 11px;
                         color: {r_color}; font-weight: 600;'>● {regime.upper()}</span>
        </div>

        <div style='margin-bottom: 14px;'>
            <div class='section-label'>WHAT IS HAPPENING</div>
            <div style='font-family: IBM Plex Sans, sans-serif; font-size: 12px;
                        color: #A0A4BC; line-height: 1.65;'>
                {reasoning}
            </div>
        </div>

        <div style='background: #141728; border-radius: 4px; padding: 12px 14px;'>
            <div class='section-label'>WHAT TO DO</div>
            <div style='font-family: IBM Plex Mono, monospace; font-size: 11px;
                        color: {d_color}; line-height: 1.65; margin-top: 4px;'>
                {action}
            </div>
        </div>

    </div>
    """, unsafe_allow_html=True)


def _render_country_positioning(risk_score: float, df: pd.DataFrame):
    st.markdown("""
    <div class='section-label' style='margin-top: 16px;'>POSITIONING vs UNIVERSE</div>
    """, unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["risk_score"],
        nbinsx=25,
        marker_color="#1E2235",
        marker_line_width=0,
        hovertemplate="Risk: %{x:.2%}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(
        x=risk_score,
        line_color="#3B82F6",
        line_width=2,
        line_dash="dot",
        annotation_text="SELECTED",
        annotation_font=dict(family="IBM Plex Mono", size=9, color="#3B82F6"),
        annotation_position="top",
    )

    layout = _layout("", height=180)
    layout["margin"] = dict(l=0, r=0, t=20, b=0)
    layout["xaxis"]["tickformat"] = ".0%"
    layout["bargap"]    = 0.05
    layout["showlegend"] = False
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─── Scenario Lab ─────────────────────────────────────────────────────────────

def render_scenario_lab(df: pd.DataFrame, model, scaler, feature_cols: List[str]):
    country_col = "country" if "country" in df.columns else df.columns[0]
    countries   = sorted(df[country_col].dropna().unique().tolist())

    st.markdown("""
    <div class='section-label'>SCENARIO SIMULATION LAB</div>
    <div style='font-family: IBM Plex Mono, monospace; font-size: 16px;
                font-weight: 600; color: #E8EAF0; margin-bottom: 6px;'>
        Macro Stress Testing Engine
    </div>
    <div style='font-family: IBM Plex Sans, sans-serif; font-size: 13px;
                color: #6B7190; margin-bottom: 24px;'>
        Simulate macro shocks and observe sovereign risk dynamics in real time.
    </div>
    """, unsafe_allow_html=True)

    col_ctrl, col_result = st.columns([1, 1.6])

    with col_ctrl:
        st.markdown("<div class='section-label'>PARAMETERS</div>", unsafe_allow_html=True)

        selected = st.selectbox(
            "SOVEREIGN", countries,
            key="scenario_country",
            label_visibility="collapsed",
        )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-family: IBM Plex Mono, monospace; font-size: 9px;
                    letter-spacing: 0.15em; color: #3A3F5C; text-transform: uppercase;
                    margin-bottom: 16px;'>SHOCK PARAMETERS</div>
        """, unsafe_allow_html=True)

        inflation_shock = st.slider(
            "INFLATION SHOCK (pp)", -10.0, 15.0, 0.0, 0.5,
            help="Change in inflation rate (percentage points)",
            key="s_inflation",
        )
        gdp_shock = st.slider(
            "GDP GROWTH SHOCK (pp)", -10.0, 5.0, 0.0, 0.5,
            help="Change in GDP growth rate (percentage points)",
            key="s_gdp",
        )
        rate_shock = st.slider(
            "INTEREST RATE SHOCK (pp)", -5.0, 10.0, 0.0, 0.25,
            help="Change in policy rate (percentage points)",
            key="s_rate",
        )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        run = st.button(
            "▶  RUN SIMULATION",
            use_container_width=True,
            type="primary",
            key="run_scenario",
        )

    # ── Compute scenario result — only when button pressed ───────────────────
    result = None
    if run and selected:
        shocks = {"inflation": inflation_shock, "gdp": gdp_shock, "interest": rate_shock}
        row    = df[df[country_col] == selected].iloc[0]
        result = run_scenario(model, scaler, feature_cols, row, shocks, selected)
        st.session_state["last_scenario"] = result
    elif not run:
        result = st.session_state.get("last_scenario")

    with col_result:
        if result:
            _render_scenario_result(result)
        else:
            st.markdown("""
            <div style='display: flex; align-items: center; justify-content: center;
                        height: 300px; border: 1px dashed #1E2235; border-radius: 8px;'>
                <div style='text-align: center;'>
                    <div style='font-family: IBM Plex Mono, monospace; font-size: 11px;
                                color: #3A3F5C; letter-spacing: 0.1em;'>
                        SET PARAMETERS AND RUN SIMULATION
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    if result:
        st.markdown("<hr>", unsafe_allow_html=True)
        _render_scenario_waterfall(result)


def _render_scenario_result(result: Dict):
    before_color = DECISION_COLORS.get(result["before_decision"], "#8B8FA8")
    after_color  = DECISION_COLORS.get(result["after_decision"],  "#8B8FA8")
    delta        = result["delta"]
    delta_sign   = "+" if delta >= 0 else ""
    impact_color = result["impact_color"]

    st.markdown(f"""
    <div style='font-family: IBM Plex Mono, monospace; font-size: 10px;
                letter-spacing: 0.15em; color: #3A3F5C; text-transform: uppercase;
                margin-bottom: 16px;'>{result["country"]} — SIMULATION OUTPUT</div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("BEFORE", f"{result['before_score']:.2%}",
                  delta=result["before_decision"])
    with c2:
        st.metric("AFTER SHOCK", f"{result['after_score']:.2%}",
                  delta=f"{delta_sign}{delta:.2%}")
    with c3:
        st.metric("IMPACT", result["impact_label"])

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    if result["decision_changed"]:
        st.markdown(f"""
        <div style='background: #1A0A0A; border: 1px solid #FF3B5C;
                    border-radius: 6px; padding: 14px 18px; margin-bottom: 12px;'>
            <div style='font-family: IBM Plex Mono, monospace; font-size: 9px;
                        letter-spacing: 0.15em; color: #FF3B5C; margin-bottom: 6px;'>
                ⚠ DECISION CHANGE TRIGGERED
            </div>
            <div style='display: flex; align-items: center; gap: 16px;'>
                <span style='font-family: IBM Plex Mono, monospace; font-size: 16px;
                             color: {before_color}; font-weight: 600;'>
                    {result["before_decision"]}
                </span>
                <span style='color: #3A3F5C; font-size: 18px;'>→</span>
                <span style='font-family: IBM Plex Mono, monospace; font-size: 16px;
                             color: {after_color}; font-weight: 600;'>
                    {result["after_decision"]}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background: #0A1A14; border: 1px solid #00C896;
                    border-radius: 6px; padding: 14px 18px; margin-bottom: 12px;'>
            <div style='font-family: IBM Plex Mono, monospace; font-size: 9px;
                        letter-spacing: 0.15em; color: #00C896; margin-bottom: 6px;'>
                DECISION UNCHANGED
            </div>
            <div style='font-family: IBM Plex Mono, monospace; font-size: 16px;
                        color: {after_color}; font-weight: 600;'>
                {result["after_decision"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    before_r_color = REGIME_COLORS.get(result["before_regime"], "#8B8FA8")
    after_r_color  = REGIME_COLORS.get(result["after_regime"],  "#8B8FA8")

    st.markdown(f"""
    <div style='display: flex; gap: 12px;'>
        <div style='flex: 1; background: #0F1220; border: 1px solid #1E2235;
                    border-radius: 6px; padding: 12px 14px;'>
            <div class='section-label'>BEFORE REGIME</div>
            <div style='font-family: IBM Plex Mono, monospace; font-size: 12px;
                        color: {before_r_color}; font-weight: 600; margin-top: 4px;'>
                {result["before_regime"]}
            </div>
        </div>
        <div style='flex: 1; background: #0F1220; border: 1px solid #1E2235;
                    border-radius: 6px; padding: 12px 14px;'>
            <div class='section-label'>AFTER REGIME</div>
            <div style='font-family: IBM Plex Mono, monospace; font-size: 12px;
                        color: {after_r_color}; font-weight: 600; margin-top: 4px;'>
                {result["after_regime"]}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_scenario_waterfall(result: Dict):
    st.markdown("""
    <div class='section-label'>RISK DELTA WATERFALL</div>
    """, unsafe_allow_html=True)

    before = result["before_score"]
    delta  = result["delta"]
    after  = result["after_score"]

    categories  = ["Baseline Risk", "Macro Shock Impact", "Post-Shock Risk"]
    bar_values  = [before, abs(delta), after]
    bar_colors  = ["#3B82F6", "#FF3B5C" if delta > 0 else "#00C896", "#F5C518"]

    fig = go.Figure()
    for cat, val, col in zip(categories, bar_values, bar_colors):
        fig.add_trace(go.Bar(
            x=[cat],
            y=[val],
            name=cat,
            marker_color=col,
            marker_line_width=0,
            text=[f"{val:.2%}"],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=11, color="#8B8FA8"),
            hovertemplate=f"<b>{cat}</b><br>Value: {val:.2%}<extra></extra>",
            width=0.5,
        ))

    layout = _layout("", height=240)
    layout["showlegend"]           = False
    layout["yaxis"]["tickformat"]  = ".0%"
    layout["yaxis"]["range"]       = [0, 1.1]
    layout["margin"] = dict(l=0, r=0, t=30, b=0)
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
