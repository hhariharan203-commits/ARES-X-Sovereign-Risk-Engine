import requests

# ─────────────────────────────────────
# CONFIG
# ─────────────────────────────────────

WORLD_BANK_BASE = "https://api.worldbank.org/v2/country"

FRED_API_KEY = "9a64ee435dc1e90529aaa8eb4850aa5d"   # 🔥 put your key here
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
# FRED (STRONG ADDITION)
# ─────────────────────────────────────

def fetch_fred(series_id):
    try:
        url = f"{FRED_BASE}?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
        res = requests.get(url, timeout=5).json()

        data = res["observations"]

        for x in reversed(data):
            if x["value"] != ".":
                return float(x["value"])
    except:
        return None


# VIX (real market volatility)
def fetch_vix():
    return fetch_fred("VIXCLS")


# US Interest Rate (proxy global rate)
def fetch_interest_rate():
    return fetch_fred("FEDFUNDS")