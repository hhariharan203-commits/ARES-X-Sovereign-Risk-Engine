"""
main.py — ARES-X Macro Intelligence Terminal
Full application routing for all pages.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ── Page config (must be first) ───────────────────────────────────────────────
st.set_page_config(
    page_title="ARES-X | Macro Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── App imports ───────────────────────────────────────────────────────────────
from ui import (
    inject_css, sidebar_logo, page_header, section_header,
    kpi_card, insight_card, decision_badge, regime_pill,
    gauge_chart, line_chart, bar_chart, heatmap_chart, pie_chart,
    apply_plotly_style, ACCENT_CYAN, ACCENT_GREEN, ACCENT_AMBER,
    ACCENT_RED, TEXT_MUTED, BG_CARD,
)
from data_api import (
    load_dataset, load_metrics, load_model, load_feature_cols,
    get_countries, get_latest, get_country_series,
)
from forecast import forecast_all, forecast_country, forecast_timeseries
from intelligence import generate_country_intelligence, generate_global_intelligence
from risk_engine import country_risk, global_risk_table, macro_score
from portfolio import get_allocation, country_rank_table
from explainability import get_feature_importance, get_category_importance
from scenario_lab import run_scenario
from decision_terminal import make_decision, bulk_decisions
from report import generate_country_report, generate_global_report
from utils import fmt_pct, fmt_signed, regime_label, regime_color, risk_color, decision_color

inject_css()

# ── Sidebar navigation ────────────────────────────────────────────────────────
sidebar_logo()

PAGES = [
    "📊  Dashboard",
    "📈  Forecast",
    "🌍  Country Intelligence",
    "🗺️  Heatmap",
    "⚖️  Multi-Country Comparison",
    "🔄  Macro Regime Dashboard",
    "⚠️  Risk Monitor",
    "💼  Portfolio Insights",
    "🧪  Scenario Lab",
    "🎯  Decision Terminal",
    "🔬  Explainability",
    "📋  Reports",
]

st.sidebar.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)
page = st.sidebar.radio("", PAGES, label_visibility="collapsed")

# ── Load core data (cached) ───────────────────────────────────────────────────
df         = load_dataset()
metrics    = load_metrics()
countries  = get_countries(df)
latest_df  = get_latest(df)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊  Dashboard":
    page_header("Global Macro Dashboard", "Real-time macro intelligence powered by ML forecasting")

    forecasts  = forecast_all()
    global_int = generate_global_intelligence()
    risk_table = global_risk_table()

    avg_gdp    = round(float(forecasts["predicted_gdp"].mean()), 2)
    avg_risk   = round(float(risk_table["Risk Score"].mean()), 1)
    avg_macro  = round(float(risk_table.apply(
        lambda r: macro_score(r["Country"]), axis=1
    ).mean()), 1) if not risk_table.empty else 50.0

    n_countries = len(countries)
    regime      = global_int.get("regime", "N/A")

    # Top KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("Avg Forecast GDP", f"{avg_gdp:.1f}%", f"Model R² {metrics.get('r2', 0):.3f}")
    with c2: kpi_card("Avg Risk Score",   f"{avg_risk:.0f}",  "0–100 scale", color=ACCENT_AMBER if avg_risk > 50 else ACCENT_GREEN)
    with c3: kpi_card("Global Regime",    regime,             global_int.get("avg_inflation", 0) and f"Inflation {global_int['avg_inflation']:.1f}%", color=regime_color(regime))
    with c4: kpi_card("Countries",        str(n_countries),   "in dataset")
    with c5: kpi_card("VIX Level",        f"{global_int.get('global_vix', 0):.1f}", "Market risk sentiment", color=ACCENT_RED if global_int.get("global_vix", 20) > 25 else ACCENT_CYAN)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([3, 2])

    with left:
        section_header("Top Countries by Forecast GDP Growth")
        top15 = forecasts.head(15)
        fig   = bar_chart(top15, "country", "predicted_gdp", "Forecast GDP Growth (%)", color_col="predicted_gdp")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        section_header("Global Executive Summary")
        summary = global_int.get("summary", "").replace("**", "")
        insight_card("MACRO INTELLIGENCE BRIEF", summary)

        section_header("Strategic Actions")
        for a in global_int.get("actions", []):
            insight_card("", a, accent=ACCENT_GREEN)

    section_header("Model Performance")
    m1, m2, m3, m4 = st.columns(4)
    with m1: kpi_card("R² Score",   f"{metrics.get('r2', 0):.4f}",  "Variance explained", color=ACCENT_GREEN)
    with m2: kpi_card("RMSE",       f"{metrics.get('rmse', 0):.4f}", "Root mean square error")
    with m3: kpi_card("MAE",        f"{metrics.get('mae', 0):.4f}",  "Mean absolute error")
    with m4: kpi_card("Train RMSE", f"{metrics.get('train_rmse', 0):.4f}", "In-sample fit")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Forecast":
    page_header("GDP Forecast Engine", "XGBoost model predictions for next-period GDP growth")

    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        country = st.selectbox("Select Country", countries)

    fc     = forecast_country(country)
    series = forecast_timeseries(country)
    intel  = generate_country_intelligence(country)

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Current GDP",    fmt_pct(fc.get("current_gdp", 0)),   "Latest observation")
    with c2: kpi_card("Forecast GDP",   fmt_pct(fc.get("predicted_gdp", 0)), "Model prediction", color=ACCENT_GREEN if fc.get("predicted_gdp", 0) > 0 else ACCENT_RED)
    with c3: kpi_card("Δ vs Current",   fmt_signed(fc.get("delta", 0)),       "Forecast change",  color=ACCENT_GREEN if fc.get("delta", 0) > 0 else ACCENT_RED)
    with c4: kpi_card("Model Confidence", f"{fc.get('confidence', 0):.1f}%", f"RMSE ±{fc.get('rmse', 0):.3f}")

    section_header("Historical GDP + Model Fitted Values")
    if not series.empty:
        fig = line_chart(
            series, "month",
            ["actual_gdp", "model_fitted"],
            f"{country} — GDP Growth: Actual vs Model",
            colors=[ACCENT_CYAN, ACCENT_AMBER],
        )
        st.plotly_chart(fig, use_container_width=True)

    section_header("Executive Intelligence")
    insight_card("FORECAST SUMMARY", intel.get("summary", "").replace("**", ""))

    section_header("Key GDP Drivers")
    for d in intel.get("drivers", [])[:3]:
        insight_card("", d)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — COUNTRY INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌍  Country Intelligence":
    page_header("Country Intelligence", "Deep-dive macro analysis by country")

    country = st.selectbox("Select Country", countries)

    fc    = forecast_country(country)
    rk    = country_risk(country)
    intel = generate_country_intelligence(country)
    ms    = macro_score(country)

    sig = intel.get("signals", {})

    section_header("Key Macro Indicators")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: kpi_card("GDP Growth",    fmt_pct(sig.get("gdp", 0)),          sig.get("gdp_trend", ""))
    with c2: kpi_card("Inflation",     fmt_pct(sig.get("inflation", 0)),    sig.get("inflation_trend", ""))
    with c3: kpi_card("Unemployment",  fmt_pct(sig.get("unemployment", 0)), sig.get("unemp_trend", ""))
    with c4: kpi_card("Interest Rate", fmt_pct(sig.get("interest_rate", 0)), sig.get("rate_trend", ""))
    with c5: kpi_card("Trade Balance", f"{sig.get('trade_balance', 0):.1f}", "Exports − Imports")
    with c6: kpi_card("VIX",           f"{sig.get('vix', 0):.1f}",          sig.get("vix_trend", ""), color=ACCENT_RED if sig.get("vix", 20) > 25 else ACCENT_CYAN)

    col_chart, col_intel = st.columns([3, 2])

    with col_chart:
        section_header("Macro Trend Series")
        cs = get_country_series(df, country)
        if not cs.empty:
            cols_to_plot = [c for c in ["gdp_growth", "inflation", "unemployment"] if c in cs.columns]
            if cols_to_plot:
                fig = line_chart(cs, "month", cols_to_plot,
                                 f"{country} — Macro Indicators",
                                 colors=[ACCENT_CYAN, ACCENT_AMBER, ACCENT_GREEN])
                st.plotly_chart(fig, use_container_width=True)

    with col_intel:
        section_header("Macro Regime")
        regime = intel.get("regime", "Unknown")
        regime_pill(regime)
        st.markdown(f"""<p style="color:#8B949E;font-size:0.85rem;margin-top:0.5rem">
            Macro Score: <strong style="color:#00E5FF">{ms:.0f}</strong> / 100 &nbsp;|&nbsp;
            Risk Score: <strong style="color:#FFC107">{rk['risk_score']:.0f}</strong> / 100
        </p>""", unsafe_allow_html=True)

        section_header("Executive Summary")
        insight_card("COUNTRY BRIEF", intel.get("summary", "").replace("**", ""))

    section_header("Key Drivers")
    col_d1, col_d2 = st.columns(2)
    drivers = intel.get("drivers", [])
    for i, d in enumerate(drivers):
        target = col_d1 if i % 2 == 0 else col_d2
        with target:
            insight_card(f"DRIVER {i+1}", d)

    section_header("Suggested Actions")
    for a in intel.get("actions", []):
        insight_card("ACTION", a, accent=ACCENT_GREEN)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️  Heatmap":
    page_header("Macro Heatmap", "Country × Feature intensity visualization")

    FEATURES = ["gdp_growth", "inflation", "unemployment", "interest_rate", "vix"]
    feat_labels = ["GDP Growth", "Inflation", "Unemployment", "Interest Rate", "VIX"]

    top_n = st.slider("Number of countries", 10, min(40, len(countries)), 20)
    selected_countries = countries[:top_n]

    hm_data = []
    for c in selected_countries:
        row_data = get_country_series(df, c)
        if row_data.empty:
            hm_data.append([0] * len(FEATURES))
        else:
            latest = row_data.iloc[-1]
            hm_data.append([float(latest.get(f, 0)) for f in FEATURES])

    z = np.array(hm_data)
    fig = heatmap_chart(z, feat_labels, selected_countries, "Macro Signals Heatmap")
    st.plotly_chart(fig, use_container_width=True)

    section_header("Reading the Heatmap")
    insight_card("HOW TO READ", "Brighter cells indicate higher values. Compare rows to identify outlier countries. Scan columns to spot macro factor concentrations across markets.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MULTI-COUNTRY COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️  Multi-Country Comparison":
    page_header("Multi-Country Comparison", "Side-by-side macro and risk comparison")

    selected = st.multiselect("Select countries to compare", countries, default=countries[:5])

    if not selected:
        st.info("Select at least one country to begin.")
    else:
        records = []
        for c in selected:
            fc  = forecast_country(c)
            rk  = country_risk(c)
            ms  = macro_score(c)
            rec = {
                "Country":       c,
                "Forecast GDP":  fc.get("predicted_gdp", 0),
                "Risk Score":    rk["risk_score"],
                "Risk Level":    rk["risk_label"],
                "Macro Score":   ms,
                "Inflation":     rk["inflation"],
                "Unemployment":  rk["unemployment"],
                "Regime":        regime_label(rk["gdp"], rk["inflation"]),
            }
            records.append(rec)

        comp_df = pd.DataFrame(records)

        # Cards
        section_header("Overview")
        cols = st.columns(len(selected))
        for i, (_, row) in enumerate(comp_df.iterrows()):
            with cols[i]:
                color = regime_color(row["Regime"])
                kpi_card(row["Country"], f"{row['Forecast GDP']:.1f}%",
                         f"Risk {row['Risk Score']:.0f} | {row['Regime']}", color=color)

        col_a, col_b = st.columns(2)

        with col_a:
            section_header("Forecast GDP Growth Comparison")
            fig = go.Figure(go.Bar(
                x=comp_df["Country"],
                y=comp_df["Forecast GDP"],
                marker_color=[ACCENT_GREEN if v > 0 else ACCENT_RED for v in comp_df["Forecast GDP"]],
            ))
            apply_plotly_style(fig)
            fig.update_layout(height=320, title_text="Forecast GDP (%)")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            section_header("Risk Score Comparison")
            fig2 = go.Figure(go.Bar(
                x=comp_df["Country"],
                y=comp_df["Risk Score"],
                marker_color=[
                    ACCENT_GREEN if v < 40 else (ACCENT_AMBER if v < 70 else ACCENT_RED)
                    for v in comp_df["Risk Score"]
                ],
            ))
            apply_plotly_style(fig2)
            fig2.update_layout(height=320, title_text="Risk Score (0–100)")
            st.plotly_chart(fig2, use_container_width=True)

        section_header("Detailed Comparison Table")
        st.dataframe(
            comp_df.style.format({
                "Forecast GDP": "{:.2f}%",
                "Risk Score":   "{:.1f}",
                "Macro Score":  "{:.1f}",
                "Inflation":    "{:.2f}%",
                "Unemployment": "{:.2f}%",
            }),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — MACRO REGIME DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔄  Macro Regime Dashboard":
    page_header("Macro Regime Dashboard", "Classification of countries by economic regime")

    regimes_map = {}
    for c in countries:
        cs = get_country_series(df, c)
        if cs.empty:
            continue
        latest = cs.iloc[-1]
        r = regime_label(float(latest.get("gdp_growth", 0)), float(latest.get("inflation", 3)))
        regimes_map[c] = r

    regime_df = pd.DataFrame(
        [{"Country": c, "Regime": r} for c, r in regimes_map.items()]
    )

    regime_counts = regime_df["Regime"].value_counts().reset_index()
    regime_counts.columns = ["Regime", "Count"]

    col_pie, col_table = st.columns([2, 3])

    with col_pie:
        section_header("Regime Distribution")
        colors_pie = [regime_color(r) for r in regime_counts["Regime"]]
        fig = go.Figure(go.Pie(
            labels=regime_counts["Regime"],
            values=regime_counts["Count"],
            hole=0.55,
            marker=dict(colors=colors_pie, line=dict(color="#0D1117", width=2)),
        ))
        apply_plotly_style(fig)
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        section_header("Regime by Country")
        for regime in ["Expansion", "Overheating", "Slowdown", "Stagflation", "Recession"]:
            group = regime_df[regime_df["Regime"] == regime]["Country"].tolist()
            if group:
                color = regime_color(regime)
                st.markdown(f"""
                <div style="margin-bottom:0.6rem;padding:0.7rem 1rem;background:#161B22;
                border-left:3px solid {color};border-radius:6px">
                    <span style="color:{color};font-size:0.75rem;font-weight:600">{regime.upper()}</span>
                    <br><span style="color:#C9D1D9;font-size:0.85rem">{', '.join(group)}</span>
                </div>
                """, unsafe_allow_html=True)

    section_header("Regime Summary Cards")
    regime_descriptions = {
        "Expansion":   "GDP > 2.5% | Inflation < 4% — Risk-on, cyclical leadership expected.",
        "Overheating": "GDP > 2.5% | Inflation ≥ 4% — Tightening risk; duration caution warranted.",
        "Slowdown":    "0 < GDP ≤ 2.5% — Moderate caution; selective quality bias recommended.",
        "Stagflation": "Low growth + high inflation — Commodities and inflation-linked assets preferred.",
        "Recession":   "GDP ≤ 0% — Defensive only; sovereign bonds and cash preservation mode.",
    }
    rcols = st.columns(5)
    for i, (regime, desc) in enumerate(regime_descriptions.items()):
        with rcols[i]:
            count = int(regime_counts[regime_counts["Regime"] == regime]["Count"].sum())
            kpi_card(regime, str(count), desc[:45] + "…", color=regime_color(regime))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — RISK MONITOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️  Risk Monitor":
    page_header("Risk Monitor", "Macro risk surveillance across all countries")

    country = st.selectbox("Select Country for Deep-Dive", countries)
    rk      = country_risk(country)
    ms      = macro_score(country)

    col_gauge, col_breakdown, col_stats = st.columns([2, 2, 2])

    with col_gauge:
        section_header("Risk Score Gauge")
        fig = gauge_chart(rk["risk_score"], f"{country} Risk Score")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"""
        <div style="text-align:center;margin-top:-1rem">
            <span style="background:{risk_color(rk['risk_label'])}22;color:{risk_color(rk['risk_label'])};
            padding:4px 14px;border-radius:20px;font-weight:600;font-size:0.85rem">
            {rk['risk_label']} RISK
            </span>
        </div>
        """, unsafe_allow_html=True)

    with col_breakdown:
        section_header("Risk Components")
        comps  = rk["components"]
        labels = list(comps.keys())
        vals   = list(comps.values())
        fig2   = go.Figure(go.Bar(
            x=vals, y=labels, orientation="h",
            marker_color=[ACCENT_GREEN if v < 40 else (ACCENT_AMBER if v < 70 else ACCENT_RED) for v in vals],
        ))
        apply_plotly_style(fig2)
        fig2.update_layout(height=260, xaxis_title="Risk Score (0–100)")
        st.plotly_chart(fig2, use_container_width=True)

    with col_stats:
        section_header("Macro Signals")
        kpi_card("GDP Growth",    fmt_pct(rk["gdp"]),          "", color=ACCENT_GREEN if rk["gdp"] > 0 else ACCENT_RED)
        kpi_card("Inflation",     fmt_pct(rk["inflation"]),    "")
        kpi_card("Unemployment",  fmt_pct(rk["unemployment"]), "")
        kpi_card("Trade Balance", f"{rk['trade_balance']:.2f}", "")

    section_header("Global Risk Rankings")
    risk_table = global_risk_table()
    st.dataframe(
        risk_table.style.format({
            "Risk Score":   "{:.1f}",
            "GDP Growth":   "{:.2f}%",
            "Inflation":    "{:.2f}%",
            "Unemployment": "{:.2f}%",
            "VIX":          "{:.1f}",
        }),
        use_container_width=True,
        height=320,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — PORTFOLIO INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💼  Portfolio Insights":
    page_header("Portfolio Insights", "Regime-driven asset allocation recommendations")

    country = st.selectbox("Select Country", countries)
    alloc   = get_allocation(country)
    rank_df = country_rank_table(top_n=15)

    col_pie, col_detail = st.columns([2, 3])

    with col_pie:
        section_header(f"{country} — Recommended Allocation")
        labels = list(alloc["allocation"].keys())
        vals   = list(alloc["allocation"].values())
        fig    = pie_chart(labels, vals, f"{country} Asset Allocation")
        st.plotly_chart(fig, use_container_width=True)

        kpi_card("Regime",     alloc["regime"],                     "", color=regime_color(alloc["regime"]))
        kpi_card("Risk Score", f"{alloc['risk_score']:.0f} / 100", "")

    with col_detail:
        section_header("Allocation Breakdown")
        for asset, pct in alloc["allocation"].items():
            bar_width = int(pct * 2.5)
            st.markdown(f"""
            <div style="margin-bottom:0.6rem">
                <div style="display:flex;justify-content:space-between;margin-bottom:0.2rem">
                    <span style="color:#C9D1D9;font-size:0.85rem">{asset}</span>
                    <span style="color:#00E5FF;font-weight:600;font-size:0.85rem">{pct:.1f}%</span>
                </div>
                <div style="background:#21262D;border-radius:4px;height:6px">
                    <div style="background:#00E5FF;width:{bar_width}%;height:6px;border-radius:4px"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        section_header("Top Countries by Portfolio Score")
        if not rank_df.empty:
            st.dataframe(
                rank_df[["Country", "Macro Score", "Predicted GDP", "Risk Level", "Regime", "Portfolio Score"]]
                .style.format({
                    "Macro Score":   "{:.1f}",
                    "Predicted GDP": "{:.2f}%",
                    "Portfolio Score": "{:.1f}",
                }),
                use_container_width=True,
                height=300,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 9 — SCENARIO LAB
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧪  Scenario Lab":
    page_header("Scenario Lab", "Simulate macro shocks and see the model's re-forecast")

    col_left, col_right = st.columns([2, 3])

    with col_left:
        country = st.selectbox("Country", countries)

        cs      = get_country_series(df, country)
        latest  = cs.iloc[-1] if not cs.empty else pd.Series()

        def _default(col, fallback):
            try:
                return float(latest.get(col, fallback))
            except Exception:
                return fallback

        section_header("Adjust Macro Levers")

        sc_inflation  = st.slider("Inflation (%)",     0.0, 15.0, _default("inflation", 3.0), 0.1)
        sc_rate       = st.slider("Interest Rate (%)", 0.0, 20.0, _default("interest_rate", 3.0), 0.1)
        sc_unemp      = st.slider("Unemployment (%)",  0.0, 20.0, _default("unemployment", 5.0), 0.1)
        sc_vix        = st.slider("VIX Level",         5.0, 80.0, _default("vix", 20.0), 0.5)
        sc_exports    = st.slider("Exports",            0.0, 200.0, _default("exports", 50.0), 1.0)
        sc_imports    = st.slider("Imports",            0.0, 200.0, _default("imports", 50.0), 1.0)

    overrides = {
        "inflation":    sc_inflation,
        "interest_rate": sc_rate,
        "unemployment": sc_unemp,
        "vix":          sc_vix,
        "exports":      sc_exports,
        "imports":      sc_imports,
    }

    result = run_scenario(country, overrides)

    with col_right:
        section_header("Scenario Results")

        if "error" in result:
            st.error(result["error"])
        else:
            r1, r2, r3 = st.columns(3)
            delta_gdp  = result["delta_gdp"]
            delta_risk = result["delta_risk"]
            with r1:
                kpi_card("Baseline GDP",  fmt_pct(result["baseline_gdp"]), "Current model forecast")
            with r2:
                color = ACCENT_GREEN if result["scenario_gdp"] > 0 else ACCENT_RED
                kpi_card("Scenario GDP",  fmt_pct(result["scenario_gdp"]), fmt_signed(delta_gdp) + " vs baseline", color=color)
            with r3:
                color = ACCENT_RED if delta_risk > 0 else ACCENT_GREEN
                kpi_card("Scenario Risk", f"{result['scenario_risk']:.1f}", fmt_signed(delta_risk) + " vs baseline", color=color)

            section_header("Scenario Regime")
            regime_pill(result.get("scenario_regime", "N/A"))

            section_header("Baseline vs Scenario Comparison")
            labels = ["Baseline GDP", "Scenario GDP"]
            vals   = [result["baseline_gdp"], result["scenario_gdp"]]
            colors = [ACCENT_CYAN, ACCENT_GREEN if delta_gdp > 0 else ACCENT_RED]

            fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=colors))
            apply_plotly_style(fig)
            fig.update_layout(height=280, title_text="GDP Growth (%)")
            st.plotly_chart(fig, use_container_width=True)

            risk_labels = ["Baseline Risk", "Scenario Risk"]
            risk_vals   = [result["baseline_risk"], result["scenario_risk"]]
            risk_colors = [ACCENT_CYAN, ACCENT_RED if delta_risk > 0 else ACCENT_GREEN]
            fig2 = go.Figure(go.Bar(x=risk_labels, y=risk_vals, marker_color=risk_colors))
            apply_plotly_style(fig2)
            fig2.update_layout(height=280, title_text="Risk Score (0–100)")
            st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 10 — DECISION TERMINAL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯  Decision Terminal":
    page_header("Decision Terminal", "Investment action engine: STRONG BUY / BUY / HOLD / DEFENSIVE")

    col_sel, col_bulk = st.columns([3, 1])
    with col_sel:
        country = st.selectbox("Select Country", countries)
    with col_bulk:
        st.markdown("<br>", unsafe_allow_html=True)
        show_bulk = st.checkbox("Show All Countries")

    dec = make_decision(country)

    col_dec, col_meta = st.columns([2, 3])

    with col_dec:
        section_header("Decision")
        color = decision_color(dec["decision"])
        st.markdown(f"""
        <div style="text-align:center;padding:2.5rem 1rem;background:#161B22;border-radius:14px;
        border:2px solid {color}44;margin-bottom:1rem">
            <div style="font-size:0.7rem;font-weight:600;color:#8B949E;letter-spacing:0.15em;margin-bottom:0.5rem">
            ARES-X RECOMMENDATION
            </div>
            <div style="font-size:3.5rem;font-weight:800;color:{color}">{dec['decision']}</div>
            <div style="color:#8B949E;font-size:0.85rem;margin-top:0.5rem">
            Score: {dec['score']} | Confidence: {dec['confidence']:.0f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        section_header("Macro Context")
        kpi_card("Macro Score",   f"{dec['macro_score']:.0f} / 100", "")
        kpi_card("Risk Score",    f"{dec['risk_score']:.0f} / 100",  "")
        kpi_card("Forecast GDP",  fmt_pct(dec["pred_gdp"]),           "")
        kpi_card("Regime",        dec["regime"],                       "", color=regime_color(dec["regime"]))

    with col_meta:
        section_header("Decision Rationale")
        insight_card("WHY THIS DECISION", dec["rationale"])

        section_header("Supporting Factors")
        for f_ in dec.get("supporting", []):
            st.markdown(f'<div style="color:#C9D1D9;font-size:0.88rem;padding:0.3rem 0">{f_}</div>',
                        unsafe_allow_html=True)

   if show_bulk:
    section_header("All-Country Decision Matrix")
    all_decs = bulk_decisions()

    dec_df = pd.DataFrame([{
        "Country":     d["country"],
        "Decision":    d["decision"],
        "Score":       d["score"],
        "Macro Score": d["macro_score"],
        "Risk Score":  d["risk_score"],
        "Regime":      d["regime"],
        "Forecast GDP": d["pred_gdp"],
    } for d in all_decs])

    strong_buy_df = dec_df[dec_df["Decision"] == "STRONG BUY"]
    buy_df        = dec_df[dec_df["Decision"] == "BUY"]
    hold_df       = dec_df[dec_df["Decision"] == "HOLD"]
    def_df        = dec_df[dec_df["Decision"] == "DEFENSIVE"]

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        kpi_card("STRONG BUY", str(len(strong_buy_df)), "countries", color=ACCENT_GREEN)
    with d2:
        kpi_card("BUY", str(len(buy_df)), "countries", color=ACCENT_GREEN)
    with d3:
        kpi_card("HOLD", str(len(hold_df)), "countries", color=ACCENT_AMBER)
    with d4:
        kpi_card("DEFENSIVE", str(len(def_df)), "countries", color=ACCENT_RED)

    st.dataframe(
        dec_df.sort_values("Score", ascending=False)
        .style.format({
            "Macro Score": "{:.1f}",
            "Risk Score": "{:.1f}",
            "Forecast GDP": "{:.2f}%"
        }),
        use_container_width=True,
        height=350,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 11 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬  Explainability":
    page_header("Model Explainability", "Real XGBoost feature importances from the trained model")

    feat_df = get_feature_importance(top_n=20)
    cat_df  = get_category_importance()

    col_feat, col_cat = st.columns([3, 2])

    with col_feat:
        section_header("Top 20 Feature Importances")
        fig = go.Figure(go.Bar(
            x=feat_df["Importance %"],
            y=feat_df["Feature"],
            orientation="h",
            marker_color=ACCENT_CYAN,
            marker_line_width=0,
        ))
        apply_plotly_style(fig)
        fig.update_layout(
            height=520,
            title_text="Feature Importance (%)",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_cat:
        section_header("Importance by Category")
        colors = [ACCENT_CYAN, ACCENT_GREEN, ACCENT_AMBER, ACCENT_RED, "#B388FF", "#FF80AB"]
        fig2 = go.Figure(go.Pie(
            labels=cat_df["Category"],
            values=cat_df["Total Importance %"],
            hole=0.52,
            marker=dict(colors=colors[:len(cat_df)], line=dict(color="#0D1117", width=2)),
        ))
        apply_plotly_style(fig2)
        fig2.update_layout(height=320)
        st.plotly_chart(fig2, use_container_width=True)

        section_header("Category Breakdown")
        for _, row in cat_df.iterrows():
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:0.3rem 0;
            border-bottom:1px solid #21262D">
                <span style="color:#C9D1D9">{row['Category']}</span>
                <span style="color:#00E5FF;font-weight:600">{row['Total Importance %']:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

    section_header("Feature Details")
    st.dataframe(
        feat_df[["Feature", "Category", "Importance %"]]
        .style.format({"Importance %": "{:.2f}%"}),
        use_container_width=True,
        height=300,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 12 — REPORTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋  Reports":
    page_header("Intelligence Reports", "Download-ready macro reports for any country")

    col_type, col_country = st.columns([2, 3])
    with col_type:
        report_type = st.radio("Report Type", ["Country Report", "Global Report"])
    with col_country:
        if report_type == "Country Report":
            country = st.selectbox("Select Country", countries)

    if st.button("⚡ Generate Report", type="primary"):
        with st.spinner("Generating intelligence report…"):
            if report_type == "Global Report":
                report_text = generate_global_report()
                filename    = "ARES-X_Global_Report.txt"
            else:
                report_text = generate_country_report(country)
                filename    = f"ARES-X_{country}_Report.txt"

        st.markdown(f"""
        <div style="background:#161B22;border:1px solid #21262D;border-radius:8px;
        padding:1.5rem;max-height:600px;overflow-y:auto;font-family:monospace;
        font-size:0.78rem;color:#C9D1D9;white-space:pre-wrap">{report_text}</div>
        """, unsafe_allow_html=True)

        st.download_button(
            label    = "⬇️ Download Report (.txt)",
            data     = report_text,
            file_name = filename,
            mime     = "text/plain",
        )
    else:
        insight_card(
            "HOW TO USE REPORTS",
            "Select a report type and optionally a country, then click Generate Report. "
            "The full structured analysis will appear below and can be downloaded as a .txt file. "
            "Reports include: executive summary, GDP forecast, risk assessment, "
            "portfolio allocation, and investment decision with rationale.",
        )
