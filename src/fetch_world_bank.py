import time
from pathlib import Path

import pandas as pd
import requests


BASE_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
DATE_RANGE = "2000:2024"

# Expanded country coverage (mix of developed & emerging, ISO3 codes)
COUNTRIES = [
    "USA", "CAN", "GBR", "DEU", "FRA", "ESP", "ITA", "AUS", "JPN", "KOR",
    "CHN", "IND", "BRA", "MEX", "ARG", "TUR", "ZAF", "RUS", "IDN", "SAU",
    "ARE", "SGP", "MYS", "THA", "PHL", "VNM", "POL", "CZE", "HUN", "ROU",
]

INDICATORS = {
    "NY.GDP.MKTP.CD": "gdp_current_usd",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct",
    "FP.CPI.TOTL.ZG": "inflation_cpi_pct",
    "SL.UEM.TOTL.ZS": "unemployment_rate_pct",
    "FR.INR.RINR": "interest_rate_pct",
    "NE.EXP.GNFS.ZS": "exports_pct_gdp",
    "NE.IMP.GNFS.ZS": "imports_pct_gdp",
}


def fetch_indicator(country: str, indicator: str) -> list[dict]:
    """Fetch a single indicator for one country over the date range."""
    url = BASE_URL.format(country=country, indicator=indicator)
    params = {"date": DATE_RANGE, "format": "json", "per_page": 20000}

    resp = None
    for attempt in range(2):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            break
        except requests.RequestException:
            if attempt < 1:
                print(f"Retrying for {country} {indicator}...")
                time.sleep(2)
                continue
            else:
                # Give up on this indicator-country pair but continue overall run
                return []

    payload = resp.json()
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        return []
    rows = []
    for entry in payload[1]:
        rows.append(
            {
                "country": entry.get("country", {}).get("id") or country,
                "country_name": entry.get("country", {}).get("value"),
                "indicator": indicator,
                "year": entry.get("date"),
                "value": entry.get("value"),
            }
        )
    return rows


def build_dataframe() -> pd.DataFrame:
    records = []
    total_countries = len(COUNTRIES)
    for idx, country in enumerate(COUNTRIES, start=1):
        print(f"Fetching country {idx} of {total_countries}")
        for indicator in INDICATORS:
            records.extend(fetch_indicator(country, indicator))
            time.sleep(0.2)  # polite rate limiting between calls

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError("No data fetched from World Bank API.")

    # Pivot to wide format with indicators as columns
    wide = (
        df.pivot_table(
            index=["country", "country_name", "year"],
            columns="indicator",
            values="value",
        )
        .reset_index()
    )

    # Rename indicator columns
    wide = wide.rename(columns=INDICATORS)

    # Clean data
    wide = wide.dropna()
    wide["year"] = pd.to_datetime(wide["year"], format="%Y")

    # Sort for readability
    wide = wide.sort_values(["country", "year"]).reset_index(drop=True)
    return wide


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    df = build_dataframe()
    project_root = Path(__file__).resolve().parents[1]
    out_path = project_root / "data" / "world_bank_data.csv"
    save_csv(df, out_path)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
