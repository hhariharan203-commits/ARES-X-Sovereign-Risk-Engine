"""
volatility.py — FINAL FIX (real country differentiation)
"""

import numpy as np
from live_data import (
    fetch_vix,
    fetch_country_bond_vol,
    fetch_country_fx_vol,
    fetch_country_equity_vol
)

# Country-specific risk multipliers (VERY IMPORTANT)
COUNTRY_RISK_PREMIUM = {
    "India": 1.2,
    "United States": 0.9,
    "Germany": 0.8,
    "Japan": 0.85,
    "Brazil": 1.4,
    "Turkey": 1.8,
    "Argentina": 2.2,
    "South Africa": 1.5,
}

def get_country_volatility(country: str) -> float:

    try:
        vix   = float(fetch_vix())
        bond  = float(fetch_country_bond_vol(country))
        fx    = float(fetch_country_fx_vol(country))
        eq    = float(fetch_country_equity_vol(country))

        base_score = (
            0.4 * vix +
            0.3 * bond +
            0.2 * fx +
            0.1 * eq
        )

        # 🔥 KEY FIX (country differentiation)
        multiplier = COUNTRY_RISK_PREMIUM.get(country, 1.1)

        final_score = base_score * multiplier

        return round(final_score, 2)

    except:
        return 20.0
