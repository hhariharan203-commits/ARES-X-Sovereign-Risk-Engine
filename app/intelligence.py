"""
intelligence.py — Executive-grade macro intelligence engine.
Generates McKinsey/Goldman-quality insights from real signals.
"""

import pandas as pd
import numpy as np
from data_api import get_country_series, load_dataset
from utils import trend_direction, regime_label, momentum_score


# ── Signal extraction ─────────────────────────────────────────────────────────

def _extract_signals(series: pd.DataFrame) -> dict:
    """Extract the latest and trend signals from a country series."""
    if series.empty or len(series) < 3:
        return {}

    latest = series.iloc[-1]

    def _trend(col):
        if col in series.columns:
            return trend_direction(series[col], window=4)
        return "Stable"

    def _mom(col):
        if col in series.columns:
            return momentum_score(series[col], window=6)
        return 0.0

    def _val(col):
        return float(latest.get(col, 0.0))

    gdp          = _val("gdp_growth")
    inflation    = _val("inflation")
    unemployment = _val("unemployment")
    interest     = _val("interest_rate")
    exports      = _val("exports")
    imports      = _val("imports")
    vix          = _val("vix")
    sentiment    = _val("sentiment_mean")

    trade_balance = exports - imports

    return {
        "gdp":               gdp,
        "gdp_trend":         _trend("gdp_growth"),
        "gdp_momentum":      _mom("gdp_growth"),
        "inflation":         inflation,
        "inflation_trend":   _trend("inflation"),
        "unemployment":      unemployment,
        "unemp_trend":       _trend("unemployment"),
        "interest_rate":     interest,
        "rate_trend":        _trend("interest_rate"),
        "exports":           exports,
        "imports":           imports,
        "trade_balance":     trade_balance,
        "trade_trend":       _trend("exports"),
        "vix":               vix,
        "vix_trend":         _trend("vix"),
        "sentiment":         sentiment,
        "regime":            regime_label(gdp, inflation),
        "date":              str(latest["month"]) if "month" in latest.index else "N/A",
    }


# ── Narrative builders ────────────────────────────────────────────────────────

def _gdp_narrative(sig: dict) -> str:
    g = sig["gdp"]
    t = sig["gdp_trend"]
    if g > 3.5 and t == "Rising":
        return f"GDP growth is robust at {g:.1f}% and accelerating, signaling strong aggregate demand and output expansion."
    elif g > 2.0 and t == "Stable":
        return f"GDP growth holds at {g:.1f}%, reflecting sustained but moderating economic momentum."
    elif g > 0 and t == "Falling":
        return f"GDP growth at {g:.1f}% is decelerating — early signs of demand softness warrant monitoring."
    elif g <= 0:
        return f"Negative GDP growth ({g:.1f}%) indicates contractionary pressure; recessionary dynamics are emerging."
    else:
        return f"GDP growth of {g:.1f}% is tepid, suggesting sub-trend expansion with limited upside catalyst."


def _inflation_narrative(sig: dict) -> str:
    i  = sig["inflation"]
    t  = sig["inflation_trend"]
    ir = sig["interest_rate"]
    if i > 6.0 and t == "Rising":
        return f"Inflation is running hot at {i:.1f}% and rising — real rate compression is eroding monetary effectiveness despite a {ir:.1f}% policy rate."
    elif i > 4.0 and t == "Falling":
        return f"Inflation elevated at {i:.1f}% but on a downward path; disinflation trajectory reduces near-term tightening pressure."
    elif i <= 2.5 and t == "Falling":
        return f"Inflation below {i:.1f}% with a falling trend creates space for policy accommodation if growth deteriorates."
    elif 2.0 <= i <= 4.0:
        return f"Inflation at {i:.1f}% remains within a manageable band; central bank policy stance appears appropriately calibrated."
    else:
        return f"Inflation of {i:.1f}% with a {t.lower()} trend keeps monetary policy on a watchful footing."


def _labor_narrative(sig: dict) -> str:
    u = sig["unemployment"]
    t = sig["unemp_trend"]
    if u < 4.0 and t in ["Stable", "Falling"]:
        return f"Labor markets are tight with unemployment at {u:.1f}%; wage pressures may sustain consumer spending but add inflationary risk."
    elif u > 7.0 and t == "Rising":
        return f"Unemployment rising to {u:.1f}% signals deteriorating labor conditions and potential consumer demand weakness ahead."
    elif t == "Falling":
        return f"Unemployment declining to {u:.1f}% supports household income resilience and near-term consumption stability."
    else:
        return f"Unemployment at {u:.1f}% is broadly stable; labor market is providing a neutral influence on aggregate demand."


def _trade_narrative(sig: dict) -> str:
    tb = sig["trade_balance"]
    tt = sig["trade_trend"]
    ex = sig["exports"]
    im = sig["imports"]
    if tb > 0 and tt == "Rising":
        return f"Trade surplus is widening ({ex:.1f} vs {im:.1f}), indicating competitive export performance and positive external demand contribution."
    elif tb < 0 and tt == "Falling":
        return f"Trade deficit widening ({ex:.1f} vs {im:.1f}) — import demand is outpacing exports, creating a drag on net GDP contribution."
    elif tb > 0:
        return f"Positive trade balance ({tb:.1f}) provides a structural buffer against external financing vulnerability."
    else:
        return f"Trade deficit of {abs(tb):.1f} units reflects external sector drag; currency dynamics and competitiveness merit attention."


def _risk_sentiment_narrative(sig: dict) -> str:
    v = sig["vix"]
    t = sig["vix_trend"]
    if v > 30 and t == "Rising":
        return f"Elevated VIX at {v:.1f} and rising signals acute risk-off sentiment; institutional positioning is likely de-risking."
    elif v > 25:
        return f"VIX at {v:.1f} reflects heightened market anxiety — risk premium expansion is compressing equity multiples."
    elif v < 15 and t in ["Stable", "Falling"]:
        return f"VIX at {v:.1f} reflects complacency; volatility compression often precedes sharp regime transitions."
    else:
        return f"VIX at {v:.1f} is within normal range, suggesting orderly market conditions with manageable tail risk."


def _actions(sig: dict) -> list:
    regime   = sig["regime"]
    gdp      = sig["gdp"]
    inflation = sig["inflation"]
    vix      = sig["vix"]
    unemp    = sig["unemployment"]
    tb       = sig["trade_balance"]
    rate     = sig["interest_rate"]

    actions = []

    # Growth-related action
    if gdp > 3.0:
        actions.append("Overweight cyclical equities and credit; growth momentum supports risk asset outperformance.")
    elif gdp > 1.5:
        actions.append("Maintain balanced allocation — blend quality growth with defensive income exposure given moderate expansion.")
    elif gdp <= 0:
        actions.append("Shift toward defensive positioning: increase sovereign bonds, reduce high-beta equity exposure.")
    else:
        actions.append("Consider selective value opportunities while maintaining sufficient liquidity buffer for volatility.")

    # Inflation/rate action
    if inflation > 5.0 and rate < inflation:
        actions.append("Negative real rates favor inflation-linked bonds (TIPS/linkers) and hard asset allocation.")
    elif inflation < 2.5 and sig["rate_trend"] == "Falling":
        actions.append("Duration extension is attractive — rate cuts ahead favor long-end sovereign bond exposure.")
    else:
        actions.append("Maintain moderate duration; inflation-rate equilibrium limits extreme positioning on the yield curve.")

    # Risk/trade action
    if vix > 28:
        actions.append("Hedge tail risk via options overlays; reduce gross exposure until volatility regime normalizes.")
    elif tb < -5:
        actions.append("Monitor currency risk — persistent trade deficits may pressure the exchange rate; hedge FX exposure.")
    elif unemp > 7.0:
        actions.append("Monitor consumer discretionary sector — rising unemployment will weaken retail and credit fundamentals.")
    else:
        actions.append("External sector stable; maintain strategic EM exposure where local growth dynamics are constructive.")

    return actions[:3]


# ── Main interface ─────────────────────────────────────────────────────────────

def generate_country_intelligence(country: str) -> dict:
    """Generate full executive intelligence for a single country."""
    df     = load_dataset()
    series = get_country_series(df, country)

    if series.empty:
        return {"country": country, "error": "No data available."}

    sig = _extract_signals(series)
    if not sig:
        return {"country": country, "error": "Insufficient data for analysis."}

    regime = sig["regime"]

    summary = (
        f"{country} is in a **{regime}** macro regime with GDP at {sig['gdp']:.1f}% "
        f"({sig['gdp_trend'].lower()}) and inflation at {sig['inflation']:.1f}% "
        f"({sig['inflation_trend'].lower()}). "
    )

    if regime == "Expansion":
        summary += "Risk-on bias is supported by constructive fundamentals."
    elif regime == "Recession":
        summary += "Defensive allocation and credit quality preservation are priority."
    elif regime == "Overheating":
        summary += "Monetary tightening risk warrants duration caution and real asset hedge."
    elif regime == "Stagflation":
        summary += "Dual headwinds of weak growth and persistent inflation demand tactical agility."
    else:
        summary += "Selective positioning with asymmetric risk management is recommended."

    drivers = [
        _gdp_narrative(sig),
        _inflation_narrative(sig),
        _labor_narrative(sig),
        _trade_narrative(sig),
        _risk_sentiment_narrative(sig),
    ]

    actions = _actions(sig)

    return {
        "country":  country,
        "regime":   regime,
        "date":     sig["date"],
        "signals":  sig,
        "summary":  summary,
        "drivers":  drivers,
        "actions":  actions,
    }


def generate_global_intelligence(top_n: int = 5) -> dict:
    """Generate a global macro intelligence summary."""
    from forecast import forecast_all

    forecasts = forecast_all()
    if forecasts.empty:
        return {"error": "No forecast data available."}

    avg_gdp   = round(float(forecasts["predicted_gdp"].mean()), 2)
    top_up    = forecasts.nlargest(top_n, "predicted_gdp")["country"].tolist()
    top_down  = forecasts.nsmallest(top_n, "predicted_gdp")["country"].tolist()

    df        = load_dataset()
    global_vix = float(df["vix"].dropna().iloc[-1]) if "vix" in df.columns else 20.0
    avg_inf   = float(df["inflation"].dropna().mean()) if "inflation" in df.columns else 3.0

    if avg_gdp > 3.0 and avg_inf < 4.0:
        global_regime = "Expansion"
        global_summary = (
            f"Global macro is in **Expansion** territory. Average forecast GDP of {avg_gdp:.1f}% "
            f"with contained inflation ({avg_inf:.1f}%) supports a constructive risk-on bias. "
            f"Cyclical leadership is expected across developed and select EM markets."
        )
    elif avg_gdp > 2.0 and avg_inf >= 4.0:
        global_regime = "Overheating"
        global_summary = (
            f"Global growth at {avg_gdp:.1f}% is robust but inflation at {avg_inf:.1f}% signals "
            f"overheating. Central banks may sustain tighter-for-longer policies, compressing multiples."
        )
    elif avg_gdp <= 0:
        global_regime = "Recession"
        global_summary = (
            f"Global GDP contracting to {avg_gdp:.1f}% — recession dynamics are dominant. "
            f"Sovereign bond allocation and quality defensive positioning are appropriate."
        )
    else:
        global_regime = "Slowdown"
        global_summary = (
            f"Global growth slowing to {avg_gdp:.1f}% while inflation at {avg_inf:.1f}% remains sticky. "
            f"Selective positioning with elevated cash buffers is warranted."
        )

    global_actions = [
        f"Overweight markets in the top growth cohort: {', '.join(top_up[:3])}.",
        f"Underweight or hedge exposure to bottom growth markets: {', '.join(top_down[:3])}.",
        "Monitor central bank policy divergence — rate differentials will drive FX and EM flow dynamics.",
    ]

    return {
        "regime":          global_regime,
        "avg_gdp":         avg_gdp,
        "avg_inflation":   round(avg_inf, 2),
        "global_vix":      round(global_vix, 2),
        "top_growth":      top_up,
        "bottom_growth":   top_down,
        "summary":         global_summary,
        "actions":         global_actions,
    }