from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from pytrends.request import TrendReq


KEYWORDS = ["inflation", "currency crisis", "debt crisis"]
COUNTRIES = {
    "IN": "India",
    "BR": "Brazil",
    "TR": "Turkey",
    "AR": "Argentina",
}


def timeframe_last_10_years() -> str:
    end = date.today()
    start = end - timedelta(days=365 * 10)
    return f"{start:%Y-%m-%d} {end:%Y-%m-%d}"


def fetch_country_trends(pytrends: TrendReq, geo: str, country_name: str) -> pd.DataFrame:
    pytrends.build_payload(KEYWORDS, timeframe=timeframe_last_10_years(), geo=geo)
    df = pytrends.interest_over_time()
    if df.empty:
        return pd.DataFrame()

    df = df.reset_index().rename(columns={"date": "timestamp"})
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    df["country_code"] = geo
    df["country"] = country_name
    df = df.dropna()
    return df


def build_dataframe() -> pd.DataFrame:
    pytrends = TrendReq(hl="en-US", tz=0)
    frames = []
    for geo, name in COUNTRIES.items():
        frames.append(fetch_country_trends(pytrends, geo, name))
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        raise RuntimeError("No Google Trends data retrieved.")
    return combined


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    df = build_dataframe()
    project_root = Path(__file__).resolve().parents[1]
    out_path = project_root / "data" / "google_trends.csv"
    save_csv(df, out_path)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
