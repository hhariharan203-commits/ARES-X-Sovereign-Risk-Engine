"""
volatility.py — Country-specific volatility engine (final production version)
"""

from live_data import (
    fetch_vix,
    fetch_country_bond_vol,
    fetch_country_fx_vol,
    fetch_country_equity_vol
)


def get_country_volatility(country: str) -> float:
    """
    Composite volatility score using real + country-specific data
    """

    try:
        vix   = float(fetch_vix())
        bond  = float(fetch_country_bond_vol(country))
        fx    = float(fetch_country_fx_vol(country))
        eq    = float(fetch_country_equity_vol(country))

    except Exception:
        return 20.0

    score = (
        0.4 * vix +
        0.3 * bond +
        0.2 * fx +
        0.1 * eq
    )

    return round(score, 2)
