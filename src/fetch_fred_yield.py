from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests


FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
SERIES_ID = "DGS10"
START = "2000-01-01"
END = "2023-12-31"


def fetch_fred_series(api_key: str) -> pd.DataFrame:
    """Fetch DGS10 observations from FRED."""
    params = {
        "series_id": SERIES_ID,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": START,
        "observation_end": END,
    }
    resp = requests.get(FRED_API_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    observations = data.get("observations", [])
    df = pd.DataFrame(observations)
    if df.empty:
        raise RuntimeError("No data returned from FRED for series DGS10.")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw FRED observations."""
    # Keep only date and value columns
    df = df[["date", "value"]].copy()

    # Replace '.' placeholders with NaN then drop missing
    df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
    df = df.dropna(subset=["value"])

    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Sort for consistency
    df = df.sort_values("date").reset_index(drop=True)
    df = df.rename(columns={"date": "timestamp", "value": "dgs10_yield_pct"})
    return df


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FRED_API_KEY environment variable is required to fetch data."
        )

    raw = fetch_fred_series(api_key)
    cleaned = clean(raw)

    project_root = Path(__file__).resolve().parents[1]
    out_path = project_root / "data" / "fred_yield.csv"
    save_csv(cleaned, out_path)
    print(f"Saved {len(cleaned)} rows to {out_path}")


if __name__ == "__main__":
    main()
