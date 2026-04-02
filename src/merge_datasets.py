from __future__ import annotations

from pathlib import Path
from typing import List
import pandas as pd

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

WB_PATH     = DATA_DIR / "world_bank_data.csv"
FRED_PATH   = DATA_DIR / "fred_yield.csv"
NEWS_PATH   = DATA_DIR / "news_sentiment.csv"
TRENDS_PATH = DATA_DIR / "google_trends.csv"

OUT_PATH = DATA_DIR / "master_dataset.csv"

# ─────────────────────────────────────────────
# COUNTRY MAPPING
# ─────────────────────────────────────────────
GEO_TO_ISO3 = {
    "IN": "IND",
    "BR": "BRA",
    "TR": "TUR",
    "AR": "ARG",
    "US": "USA"
}

# ─────────────────────────────────────────────
# WORLD BANK (ROBUST)
# ─────────────────────────────────────────────
def load_world_bank(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # ✅ HANDLE BOTH YEAR OR MONTH
    if "year" in df.columns:
        df["date"] = pd.to_datetime(df["year"], errors="coerce")
    elif "month" in df.columns:
        df["date"] = pd.to_datetime(df["month"], errors="coerce")
    else:
        raise ValueError("❌ No 'year' or 'month' column found")

    df = df.dropna(subset=["date"])

    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    value_cols = [
        c for c in df.columns
        if c not in {"country", "country_name", "year", "month", "date"}
    ]

    df = df[["country", "month"] + value_cols]
    df = df.sort_values(["country", "month"])

    return df


# ─────────────────────────────────────────────
# FRED (DAILY → MONTHLY)
# ─────────────────────────────────────────────
def load_fred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])

    df["month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()
    df["country"] = "USA"

    df = (
        df.groupby(["country", "month"], as_index=False)
        .agg({"dgs10_yield_pct": "mean"})
    )

    return df


# ─────────────────────────────────────────────
# NEWS SENTIMENT
# ─────────────────────────────────────────────
def load_news(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])

    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["country"] = df["country"].str.upper()

    df = (
        df.groupby(["country", "month"], as_index=False)
        .agg({"sentiment_score": "mean"})
        .rename(columns={"sentiment_score": "sentiment_mean"})
    )

    return df


# ─────────────────────────────────────────────
# GOOGLE TRENDS
# ─────────────────────────────────────────────
def load_trends(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])

    df["country"] = df["country_code"].map(GEO_TO_ISO3).fillna(df["country_code"])
    df["month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()

    value_cols = [
        c for c in df.columns
        if c not in {"timestamp", "country_code", "country", "month", "isPartial"}
    ]

    df = (
        df.groupby(["country", "month"], as_index=False)[value_cols]
        .mean()
    )

    return df


# ─────────────────────────────────────────────
# MERGE ENGINE (SAFE + REALISTIC)
# ─────────────────────────────────────────────
def merge_all(dfs: List[pd.DataFrame]) -> pd.DataFrame:

    base = dfs[0].copy()

    for df in dfs[1:]:
        base = pd.merge(
            base,
            df,
            on=["country", "month"],
            how="left"
        )

    base = base.sort_values(["country", "month"]).reset_index(drop=True)

    numeric_cols = [
        c for c in base.columns
        if c not in {"country", "month"}
    ]

    # ✅ SAFE INTERPOLATION (NO CONSTANT VALUES)
    base[numeric_cols] = (
        base.groupby("country")[numeric_cols]
        .apply(lambda g: g.interpolate(limit_direction="both"))
        .reset_index(level=0, drop=True)
    )

    return base


# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
def save(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("🚀 Loading datasets...")

    wb = load_world_bank(WB_PATH)
    fred = load_fred(FRED_PATH)
    news = load_news(NEWS_PATH)
    trends = load_trends(TRENDS_PATH)

    print("🔗 Merging datasets...")

    df = merge_all([wb, fred, news, trends])

    save(df, OUT_PATH)

    print("\n✅ MASTER DATASET READY")
    print(f"Shape: {df.shape}")
    print(f"Countries: {df['country'].nunique()}")
    print(f"Date range: {df['month'].min()} → {df['month'].max()}")


if __name__ == "__main__":
    main()