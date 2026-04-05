"""
live_data.py — Real-time macro + country-specific volatility (World Bank + FRED)
"""

import requests
import numpy as np

# ─────────────────────────────────────
# CONFIG
# ─────────────────────────────────────

WORLD_BANK_BASE = "https://api.worldbank.org/v2/country"

FRED_API_KEY = "9a64ee435dc1e90529aaa8eb4850aa5d"
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


# ─────────────────────────────────────
# WORLD BANK
# ─────────────────────────────────────

def fetch_gdp(country_code):
    try:
        url = f"{WORLD_BANK_BASE}/{country_code}/indicator/NY.GDP.MKTP.KD.ZG?format=json"
        res = requests.get(url, timeout=5).json()
        data = res[1]

        for x in data:
            if x["value"] is not None:
                return float(x["value"])
    except:
        return None


def fetch_inflation(country_code):
    try:
        url = f"{WORLD_BANK_BASE}/{country_code}/indicator/FP.CPI.TOTL.ZG?format=json"
        res = requests.get(url, timeout=5).json()
        data = res[1]

        for x in data:
            if x["value"] is not None:
                return float(x["value"])
    except:
        return None


# ─────────────────────────────────────
# FRED CORE
# ─────────────────────────────────────

def fetch_fred_series(series_id, limit=30):
    try:
        url = f"{FRED_BASE}?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json&limit={limit}"
        res = requests.get(url, timeout=5).json()

        values = [
            float(x["value"]) for x in res["observations"]
            if x["value"] != "."
        ]

        return values
    except:
        return []


def fetch_fred_latest(series_id):
    data = fetch_fred_series(series_id, limit=10)
    return data[-1] if data else None


# ─────────────────────────────────────
# GLOBAL VOL COMPONENTS
# ─────────────────────────────────────

def fetch_vix():
    return fetch_fred_latest("VIXCLS") or 20.0


def fetch_bond_vol():
    data = fetch_fred_series("DGS10", limit=30)
    return float(np.std(data)) if len(data) > 5 else 10.0


def fetch_fx_vol():
    data = fetch_fred_series("DTWEXBGS", limit=30)
    return float(np.std(data)) if len(data) > 5 else 8.0


def fetch_equity_vol():
    data = fetch_fred_series("SP500", limit=30)

    if len(data) > 5:
        returns = np.diff(data) / data[:-1]
        return float(np.std(returns) * 100)

    return 15.0


# ─────────────────────────────────────
# COUNTRY-SPECIFIC MAPPING
# ─────────────────────────────────────

COUNTRY_FRED_MAP = {
    "United States": {
        "bond": "DGS10",
        "fx": "DTWEXBGS",
    },
    "India": {
        "bond": "INDIRLTLT01STM",
        "fx": "DEXINUS",
    },
    "Germany": {
        "bond": "IRLTLT01DEM156N",
        "fx": "DEXUSEU",
    },
    "Japan": {
        "bond": "IRLTLT01JPM156N",
        "fx": "DEXJPUS",
    }
}


# ─────────────────────────────────────
# COUNTRY-SPECIFIC VOL
# ─────────────────────────────────────

def fetch_country_bond_vol(country):
    series = COUNTRY_FRED_MAP.get(country, {}).get("bond")

    if not series:
        return fetch_bond_vol()

    data = fetch_fred_series(series, limit=30)
    return float(np.std(data)) if len(data) > 5 else fetch_bond_vol()


def fetch_country_fx_vol(country):
    series = COUNTRY_FRED_MAP.get(country, {}).get("fx")

    if not series:
        return fetch_fx_vol()

    data = fetch_fred_series(series, limit=30)
    return float(np.std(data)) if len(data) > 5 else fetch_fx_vol()


def fetch_country_equity_vol(country):
    # fallback (FRED has limited indices)
    return fetch_equity_vol()


# ─────────────────────────────────────
# INTEREST RATE
# ─────────────────────────────────────

def fetch_interest_rate():
    return fetch_fred_latest("FEDFUNDS")
