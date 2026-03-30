from __future__ import annotations

from functools import reduce
from pathlib import Path
from typing import List

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Mapping 2-letter geo codes to 3-letter ISO codes to align across sources
GEO_TO_ISO3 = {"IN": "IND", "BR": "BRA", "TR": "TUR", "AR": "ARG", "US": "USA"}


def load_world_bank(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["year"])
    # Use January 1 as anchor for the year, then expand to monthly and forward-fill.
    df["date"] = df["year"].dt.to_period("Y").dt.to_timestamp()
    value_cols = [c for c in df.columns if c not in {"country", "country_name", "year", "date"}]
    df = df[["country", "date"] + value_cols]

    # Expand to monthly frequency per country and forward-fill annual values.
    monthly = []
    for country, g in df.groupby("country"):
        g = g.set_index("date").resample("MS").ffill().reset_index()
        g["country"] = country
        if "date" in g.columns:
            g.rename(columns={"date": "month"}, inplace=True)
        g["month"] = pd.to_datetime(g["month"])
        monthly.append(g)
    return pd.concat(monthly, ignore_index=True)


def load_fred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["date"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()
    # DGS10 is US-only; add country code for merge.
    df["country"] = "USA"
    monthly = (
        df.groupby(["country", "date"])["dgs10_yield_pct"]
        .mean()
        .reset_index()
    )
    if "date" in monthly.columns:
        monthly.rename(columns={"date": "month"}, inplace=True)
    monthly["month"] = pd.to_datetime(monthly["month"])
    return monthly


def load_news(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.rename(columns={"date": "month", "sentiment_score": "sentiment_mean"})
    if "date" in df.columns:
        df.rename(columns={"date": "month"}, inplace=True)
    df["month"] = pd.to_datetime(df["month"])
    df["country"] = df["country"].str.upper()
    return df[["country", "month", "sentiment_mean"]]


def load_trends(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    # Map to ISO3 for consistency
    df["country"] = df["country_code"].map(GEO_TO_ISO3).fillna(df["country_code"])
    df["date"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()
    value_cols = [c for c in df.columns if c not in {"timestamp", "country_code", "country", "date", "isPartial"}]
    df = (
        df.groupby(["country", "date"])[value_cols]
        .mean()
        .reset_index()
    )
    if "date" in df.columns:
        df.rename(columns={"date": "month"}, inplace=True)
    df["month"] = pd.to_datetime(df["month"])
    return df


def merge_all(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    def _merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        key = "month" if "month" in left.columns and "month" in right.columns else "date"
        return pd.merge(left, right, on=["country", key], how="outer")

    merged = reduce(_merge, dfs)
    # Normalize to common key name
    if "date" in merged.columns and "month" not in merged.columns:
        merged = merged.rename(columns={"date": "month"})
    merged = merged.sort_values(["country", "month"]).reset_index(drop=True)

    numeric_cols = [c for c in merged.columns if c not in {"country", "month"}]
    # Interpolate and forward-fill within each country
    merged[numeric_cols] = (
        merged.groupby("country")[numeric_cols]
        .apply(lambda g: g.interpolate().ffill())
        .reset_index(level=0, drop=True)
    )
    return merged


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    world_bank = load_world_bank(DATA_DIR / "world_bank_data.csv")
    fred = load_fred(DATA_DIR / "fred_yield.csv")
    news = load_news(DATA_DIR / "news_sentiment.csv")
    trends = load_trends(DATA_DIR / "google_trends.csv")

    master = merge_all([world_bank, fred, news, trends])
    out_path = DATA_DIR / "master_dataset.csv"
    save_csv(master, out_path)
    print(f"Saved {len(master)} rows to {out_path}")


if __name__ == "__main__":
    main()
