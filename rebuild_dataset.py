from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
WB_OUT = DATA_DIR / "world_bank_data.csv"
MASTER_OUT = DATA_DIR / "master_dataset.csv"
CLEAN_OUT = DATA_DIR / "clean_master_dataset.csv"

COUNTRIES = [
    "USA", "CHN", "IND", "JPN", "DEU", "GBR", "FRA", "ITA", "BRA", "CAN",
    "RUS", "KOR", "AUS", "ESP", "MEX", "IDN", "NLD", "SAU", "TUR", "CHE",
    "ARG", "SWE", "POL", "BEL", "THA", "IRL", "ISR", "ARE", "NOR", "SGP",
]
DATE_RANGE = "2000:2024"

INDICATORS: Dict[str, str] = {
    "NY.GDP.MKTP.CD": "gdp_current_usd",
    "NE.EXP.GNFS.ZS": "exports_pct_gdp",
    "NE.IMP.GNFS.ZS": "imports_pct_gdp",
    "FP.CPI.TOTL.ZG": "inflation_cpi_pct",
    "SL.UEM.TOTL.ZS": "unemployment_pct",
}

WB_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
TIMEOUT = 30
RETRIES = 3

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def fetch_indicator(country: str, indicator: str) -> Optional[pd.DataFrame]:
    """Fetch one indicator for one country with retries; return DataFrame or None."""
    url = WB_URL.format(country=country, indicator=indicator)
    params = {"date": DATE_RANGE, "format": "json", "per_page": 20000}
    for attempt in range(1, RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
                raise ValueError("Empty payload")
            rows = []
            for entry in payload[1]:
                rows.append({"year": entry.get("date"), "value": entry.get("value")})
            df = pd.DataFrame(rows)
            return df
        except Exception as exc:
            print(f"[WARN] {country} {indicator} attempt {attempt}/{RETRIES} failed: {exc}")
            if attempt == RETRIES:
                return None
            time.sleep(1)
    return None


def build_world_bank_country(country: str) -> pd.DataFrame:
    """Build per-country WB dataset per instructions."""
    merged: Optional[pd.DataFrame] = None
    for code, colname in INDICATORS.items():
        df_ind = fetch_indicator(country, code)
        if df_ind is None:
            print(f"[WARN] Skipping indicator {code} for {country}")
            continue
        df_ind = df_ind.rename(columns={"value": colname})
        df_ind["year"] = pd.to_numeric(df_ind["year"], errors="coerce")
        df_ind = df_ind.dropna(subset=["year"])
        if merged is None:
            merged = df_ind
        else:
            merged = pd.merge(merged, df_ind, on="year", how="outer")

    if merged is None or merged.empty:
        print(f"[WARN] No indicators fetched for {country}; creating empty frame")
        merged = pd.DataFrame(columns=["year"] + list(INDICATORS.values()))

    merged["month"] = pd.to_datetime(merged["year"].astype(int).astype(str) + "-01-01")
    merged = merged.drop(columns=["year"])
    merged = merged.sort_values("month")
    merged["country"] = country

    # Resample yearly -> monthly
    merged = (
        merged.set_index("month")
        .resample("M")
        .ffill()
        .reset_index()
    )

    if "month" not in merged.columns:
        merged = merged.reset_index()

    assert "month" in merged.columns, "month column missing after resample"
    assert "country" in merged.columns, "country column missing after resample"

    # Per-country fill only (already per single country)
    merged = merged.ffill().bfill()

    # Drop rows only if ALL features are missing
    feature_cols = list(INDICATORS.values())
    merged = merged.dropna(how="all", subset=feature_cols)

    merged = merged.reset_index(drop=True)
    merged = merged[["month", "country"] + feature_cols]
    merged = merged.loc[:, ~merged.columns.duplicated()]
    print(f"[INFO] {country}: rows={len(merged)}")
    return merged


def build_world_bank() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for c in COUNTRIES:
        frames.append(build_world_bank_country(c))
    df_world = pd.concat(frames, ignore_index=True)
    assert "month" in df_world.columns and "country" in df_world.columns
    df_world = df_world.drop_duplicates(subset=["country", "month"])
    df_world = df_world.sort_values(["country", "month"]).reset_index(drop=True)
    df_world.to_csv(WB_OUT, index=False)
    print(f"[INFO] World Bank data saved to {WB_OUT} (countries={df_world['country'].nunique()}, rows={len(df_world)})")
    return df_world


def normalize_month(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize any date/month column to 'month' datetime."""
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"])
    elif "date" in df.columns:
        df["month"] = pd.to_datetime(df["date"])
        df = df.drop(columns=["date"])
    elif "week" in df.columns:
        df["month"] = pd.to_datetime(df["week"])
        df = df.drop(columns=["week"])
    return df


def load_fred(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Missing file {path}, using empty frame")
        return pd.DataFrame(columns=["country", "month"])
    df = pd.read_csv(path)
    print("FRED columns:", df.columns.tolist())
    date_candidates = [
        col
        for col in df.columns
        if any(x in col.lower() for x in ["date", "week", "time", "timestamp"])
    ]
    if not date_candidates:
        print(f"[WARN] No date-like column in {path}; returning empty frame")
        return pd.DataFrame(columns=["country", "month"])
    date_col = date_candidates[0]
    print("Detected date column:", date_col)
    df = df.rename(columns={date_col: "month"})
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df.dropna(subset=["month"])
    if df.empty:
        print(f"[WARN] FRED empty after date conversion; returning empty frame")
        return pd.DataFrame(columns=["country", "month"])
    df = df.set_index("month").resample("M").mean(numeric_only=True).reset_index()
    if "country" not in df.columns:
        df["country"] = "USA"
    df = df.loc[:, ~df.columns.duplicated()]
    print("Final shape:", df.shape)
    return df


def load_trends(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Missing file {path}, using empty frame")
        return pd.DataFrame(columns=["country", "month"])
    df = pd.read_csv(path)
    print("TRENDS columns:", df.columns.tolist())
    date_candidates = [
        col
        for col in df.columns
        if any(x in col.lower() for x in ["date", "week", "time", "timestamp", "month"])
    ]
    if not date_candidates:
        print(f"[WARN] No date-like column in {path}; returning empty frame")
        return pd.DataFrame(columns=["country", "month"])
    date_col = date_candidates[0]
    print("Detected date column:", date_col)
    df["month"] = pd.to_datetime(df[date_col], errors="coerce")

    # Ensure country column
    if "country" in df.columns:
        pass
    elif "country_code" in df.columns:
        df["country"] = df["country_code"]
    else:
        raise ValueError("No country column found in google_trends.csv")

    df = df.dropna(subset=["month", "country"])
    if df.empty:
        print(f"[WARN] Trends empty after date conversion; returning empty frame")
        return pd.DataFrame(columns=["country", "month"])

    # Per-country monthly resample, reinserting country after resample
    df_list = []
    for country, group in df.groupby("country"):
        temp = (
            group.set_index("month")
            .resample("ME")
            .mean(numeric_only=True)
            .reset_index()
        )
        temp["country"] = country
        df_list.append(temp)

    df = pd.concat(df_list, ignore_index=True)

    assert "month" in df.columns and "country" in df.columns, "Trends missing keys after processing"
    df = df.loc[:, ~df.columns.duplicated()]
    print("Trends columns after fix:", df.columns.tolist())
    print("Trends shape:", df.shape)
    return df


def load_optional_csv(path: Path, country_default: Optional[str] = None) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Missing file {path}, using empty frame")
        return pd.DataFrame(columns=["country", "month"])
    df = pd.read_csv(path)
    df = normalize_month(df)
    if "month" not in df.columns:
        print(f"[WARN] {path} missing month column; returning empty frame")
        return pd.DataFrame(columns=["country", "month"])
    if "country" not in df.columns and country_default:
        df["country"] = country_default
    if "country" not in df.columns:
        print(f"[WARN] {path} missing country column; returning empty frame")
        return pd.DataFrame(columns=["country", "month"])
    return df


def build_master(df_world: pd.DataFrame) -> pd.DataFrame:
    fred = load_fred(DATA_DIR / "fred_yield.csv")
    news = load_optional_csv(DATA_DIR / "news_sentiment.csv")
    trends = load_trends(DATA_DIR / "google_trends.csv")

    # Start with world bank as left frame
    master = df_world.copy()
    for other, name in [(fred, "fred_yield"), (news, "news_sentiment"), (trends, "google_trends")]:
        if other.empty:
            print(f"[WARN] {name} empty; skipping merge")
            continue
        assert "month" in other.columns and "country" in other.columns, f"{name} missing keys"
        master = master.merge(other, on=["country", "month"], how="left", suffixes=("", f"_{name}"))

    assert "month" in master.columns and "country" in master.columns, "Master missing merge keys"
    master = master.drop_duplicates(subset=["country", "month"])
    master = master.loc[:, ~master.columns.duplicated()]
    master = master.sort_values(["country", "month"]).reset_index(drop=True)
    master.to_csv(MASTER_OUT, index=False)
    print(f"[INFO] Master dataset saved to {MASTER_OUT} shape={master.shape}")
    return master


def clean_master(master: pd.DataFrame) -> pd.DataFrame:
    df = master.copy()
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values(["country", "month"])

    # Per-country fill only
    df = df.groupby("country", group_keys=False).apply(lambda x: x.ffill().bfill()).reset_index(drop=True)
    print("[INFO] Applied per-country forward/backward fill to master")

    df = df.drop_duplicates(subset=["country", "month"])
    df = df.loc[:, ~df.columns.duplicated()]

    # Validation
    assert "country" in df.columns
    assert "month" in df.columns
    assert df.columns.duplicated().sum() == 0

    df.to_csv(CLEAN_OUT, index=False)
    print(f"[INFO] Clean dataset saved to {CLEAN_OUT} shape={df.shape}")
    print(f"Merged datasets: {df.shape}")
    print(f"[INFO] Countries={df['country'].nunique()}, Rows={len(df)}")
    return df


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_world = build_world_bank()
    master = build_master(df_world)
    clean_master(master)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
