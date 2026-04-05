"""
volatility.py — Country-level volatility engine (institutional upgrade)
"""

import numpy as np
from live_data import fetch_vix

# ── MOCK / INITIAL (upgrade later with APIs) ─────────────────────

def _bond_vol(country):
    # Placeholder (later: FRED yields std dev)
    return np.random.uniform(10, 25)

def _fx_vol(country):
    # Placeholder (later: FX volatility)
    return np.random.uniform(5, 20)

def _equity_vol(country):
    # Placeholder (later: index volatility)
    return np.random.uniform(10, 30)


# ── MAIN FUNCTION ───────────────────────────────────────────────

def get_country_volatility(country: str) -> float:
    """
    Composite volatility score (0–100)
    """

    try:
        vix = fetch_vix()
        vix = float(vix) if vix else 20.0
    except:
        vix = 20.0

    bond = _bond_vol(country)
    fx   = _fx_vol(country)
    eq   = _equity_vol(country)

    score = (
        0.4 * vix +
        0.3 * bond +
        0.2 * fx +
        0.1 * eq
    )

    return round(score, 2)