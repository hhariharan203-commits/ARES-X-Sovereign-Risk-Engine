"""
portfolio.py — Risk-aware asset allocation logic driven by macro signals.
"""

import pandas as pd
import numpy as np
from risk_engine import country_risk, macro_score
from forecast import forecast_country
from utils import clamp, regime_label


# ── Asset class mapping by regime ─────────────────────────────────────────────

REGIME_ALLOCATION = {
    "Expansion": {
        "Equities":          45,
        "Corporate Bonds":   20,
        "EM Assets":         15,
        "Commodities":       10,
        "Cash & Short-Term":  5,
        "Sovereign Bonds":    5,
    },
    "Overheating": {
        "Equities":          30,
        "Corporate Bonds":   10,
        "EM Assets":         10,
        "Commodities":       25,
        "Cash & Short-Term": 10,
        "Sovereign Bonds":   15,
    },
    "Slowdown": {
        "Equities":          25,
        "Corporate Bonds":   20,
        "EM Assets":          5,
        "Commodities":        5,
        "Cash & Short-Term": 20,
        "Sovereign Bonds":   25,
    },
    "Stagflation": {
        "Equities":          15,
        "Corporate Bonds":    5,
        "EM Assets":          5,
        "Commodities":       30,
        "Cash & Short-Term": 20,
        "Sovereign Bonds":   25,
    },
    "Recession": {
        "Equities":          10,
        "Corporate Bonds":    5,
        "EM Assets":          0,
        "Commodities":        5,
        "Cash & Short-Term": 30,
        "Sovereign Bonds":   50,
    },
    "Unknown": {
        "Equities":          25,
        "Corporate Bonds":   20,
        "EM Assets":         10,
        "Commodities":       10,
        "Cash & Short-Term": 20,
        "Sovereign Bonds":   15,
    },
}


def get_allocation(country: str) -> dict:
    """
    Return recommended asset allocation dict for a country
    based on its macro regime and risk score.
    """
    risk   = country_risk(country)
    gdp    = risk["gdp"]
    infl   = risk["inflation"]
    regime = regime_label(gdp, infl)

    base   = REGIME_ALLOCATION.get(regime, REGIME_ALLOCATION["Unknown"])

    # Risk adjustment: high risk → shift toward safety
    risk_score = risk["risk_score"]
    if risk_score >= 70:
        # Shift 10% from equities/EM to bonds/cash
        adjustment = {k: v for k, v in base.items()}
        adjustment["Equities"]          = max(0, adjustment["Equities"] - 10)
        adjustment["EM Assets"]         = max(0, adjustment["EM Assets"] - 5)
        adjustment["Sovereign Bonds"]   = adjustment["Sovereign Bonds"] + 10
        adjustment["Cash & Short-Term"] = adjustment["Cash & Short-Term"] + 5
        total = sum(adjustment.values())
        allocation = {k: round(v / total * 100, 1) for k, v in adjustment.items()}
    else:
        allocation = dict(base)

    return {
        "country":    country,
        "regime":     regime,
        "risk_score": risk["risk_score"],
        "allocation": allocation,
    }


def country_rank_table(top_n: int = 10) -> pd.DataFrame:
    """
    Rank countries by macro score + predicted GDP for portfolio priority.
    """
    from data_api import load_dataset
    df        = load_dataset()
    countries = df["country"].unique().tolist()

    records = []
    for c in countries:
        try:
            fc = forecast_country(c)
            ms = macro_score(c)
            rk = country_risk(c)
            records.append({
                "Country":        c,
                "Macro Score":    ms,
                "Predicted GDP":  fc.get("predicted_gdp", 0.0),
                "Risk Score":     rk["risk_score"],
                "Risk Level":     rk["risk_label"],
                "Regime":         regime_label(rk["gdp"], rk["inflation"]),
            })
        except Exception:
            pass

    df_rank = pd.DataFrame(records)
    df_rank["Portfolio Score"] = (
        df_rank["Macro Score"] * 0.5 +
        df_rank["Predicted GDP"].clip(-2, 6).apply(lambda x: (x + 2) / 8 * 100) * 0.5
    ).round(1)

    return df_rank.sort_values("Portfolio Score", ascending=False).head(top_n).reset_index(drop=True)