"""
volatility.py — FINAL FIX (correct country mapping)
"""

from live_data import (
    fetch_vix,
    fetch_country_bond_vol,
    fetch_country_fx_vol,
    fetch_country_equity_vol
)

# 🔥 FIX: COUNTRY CODE → NAME
COUNTRY_CODE_MAP = {
    "ARG": "Argentina",
    "ZAF": "South Africa",
    "TUR": "Turkey",
    "FRA": "France",
    "ESP": "Spain",
    "GBR": "United Kingdom",
    "IND": "India",
    "USA": "United States",
    "DEU": "Germany",
    "JPN": "Japan",
    "BRA": "Brazil",
}

# 🔥 Risk multipliers (using real names)
COUNTRY_RISK_PREMIUM = {
    "United States": 0.9,
    "Germany": 0.8,
    "Japan": 0.85,
    "France": 0.9,
    "United Kingdom": 0.95,

    "India": 1.2,
    "Brazil": 1.4,
    "South Africa": 1.5,

    "Turkey": 1.8,
    "Argentina": 2.2,
}

def get_country_volatility(country: str) -> float:

    try:
        # 🔥 FIX: Convert code → name
        country_name = COUNTRY_CODE_MAP.get(country, country)

        vix   = float(fetch_vix())
        bond  = float(fetch_country_bond_vol(country_name))
        fx    = float(fetch_country_fx_vol(country_name))
        eq    = float(fetch_country_equity_vol(country_name))

        base_score = (
            0.4 * vix +
            0.3 * bond +
            0.2 * fx +
            0.1 * eq
        )

        # 🔥 NOW THIS WILL WORK
        multiplier = COUNTRY_RISK_PREMIUM.get(country_name, 1.1)

        final_score = base_score * multiplier

        return round(final_score, 2)

    except Exception:
        return 20.0
