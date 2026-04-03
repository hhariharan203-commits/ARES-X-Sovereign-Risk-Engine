"""
risk_engine.py — Composite risk scoring and regime classification.
"""

import numpy as np
import pandas as pd
from data_api import get_country_series, load_dataset
from utils import clamp, fmt_risk_label


# ── Risk factor weights ───────────────────────────────────────────────────────

WEIGHTS = {
    "gdp":        -0.30,   # negative GDP → higher risk
    "inflation":   0.20,   # high inflation → higher risk
    "unemployment": 0.20,  # high unemployment → higher risk
    "vix":         0.20,   # high VIX → higher risk
    "trade":      -0.10,   # trade surplus → lower risk
}


def _risk_from_signals(gdp, inflation, unemployment, vix, trade_balance) -> float:
    """
    Compute a 0–100 risk score from macro signals.
    Higher = more risk.
    """
    # Normalize each dimension to [0, 1] using domain-expert bounds
    gdp_risk    = clamp(1 - (gdp + 2) / 8.0, 0, 1)          # [-2, 6] → [1, 0]
    inf_risk    = clamp((inflation - 1) / 8.0, 0, 1)         # [1, 9] → [0, 1]
    unemp_risk  = clamp((unemployment - 2) / 10.0, 0, 1)     # [2, 12] → [0, 1]
    vix_risk    = clamp((vix - 12) / 28.0, 0, 1)             # [12, 40] → [0, 1]
    trade_risk  = clamp(1 - (trade_balance + 10) / 20.0, 0, 1)  # [-10, 10] → [1, 0]

    score = (
        abs(WEIGHTS["gdp"])         * gdp_risk    +
        WEIGHTS["inflation"]        * inf_risk    +
        WEIGHTS["unemployment"]     * unemp_risk  +
        WEIGHTS["vix"]              * vix_risk    +
        abs(WEIGHTS["trade"])       * trade_risk
    )

    return float(clamp(score * 100, 0, 100))


def country_risk(country: str) -> dict:
    """Return full risk profile for a country."""
    df     = load_dataset()
    series = get_country_series(df, country)
    if series.empty:
        return {"country": country, "risk_score": 50.0, "risk_label": "MEDIUM", "components": {}}

    latest = series.iloc[-1]

    gdp          = float(latest.get("gdp_growth",    0.0))
    inflation    = float(latest.get("inflation",     3.0))
    unemployment = float(latest.get("unemployment",  5.0))
    vix          = float(latest.get("vix",          20.0))
    exports      = float(latest.get("exports",       0.0))
    imports      = float(latest.get("imports",       0.0))
    trade_balance = exports - imports

    score = _risk_from_signals(gdp, inflation, unemployment, vix, trade_balance)
    label = fmt_risk_label(score)

    components = {
        "GDP Risk":          round(clamp(1 - (gdp + 2) / 8.0, 0, 1) * 100, 1),
        "Inflation Risk":    round(clamp((inflation - 1) / 8.0, 0, 1) * 100, 1),
        "Unemployment Risk": round(clamp((unemployment - 2) / 10.0, 0, 1) * 100, 1),
        "Volatility Risk":   round(clamp((vix - 12) / 28.0, 0, 1) * 100, 1),
        "Trade Risk":        round(clamp(1 - (trade_balance + 10) / 20.0, 0, 1) * 100, 1),
    }

    return {
        "country":      country,
        "risk_score":   round(score, 1),
        "risk_label":   label,
        "gdp":          gdp,
        "inflation":    inflation,
        "unemployment": unemployment,
        "vix":          vix,
        "trade_balance": round(trade_balance, 2),
        "components":   components,
        "date":         str(latest.get("month", "")),
    }


def global_risk_table() -> pd.DataFrame:
    """Return risk scores for all countries."""
    df        = load_dataset()
    countries = df["country"].unique().tolist()

    records = []
    for c in countries:
        r = country_risk(c)
        records.append({
            "Country":     c,
            "Risk Score":  r["risk_score"],
            "Risk Level":  r["risk_label"],
            "GDP Growth":  r["gdp"],
            "Inflation":   r["inflation"],
            "Unemployment": r["unemployment"],
            "VIX":         r["vix"],
        })

    return (
        pd.DataFrame(records)
        .sort_values("Risk Score", ascending=False)
        .reset_index(drop=True)
    )


def macro_score(country: str) -> float:
    """
    Composite macro health score: 0 (very bad) → 100 (excellent).
    Inverse of risk score with slight weighting toward GDP.
    """
    r  = country_risk(country)
    rs = r["risk_score"]
    gdp_bonus = clamp(r["gdp"] * 5, 0, 15)   # Up to 15-point bonus for strong growth
    return round(clamp(100 - rs + gdp_bonus, 0, 100), 1)