from __future__ import annotations

from pathlib import Path

import pandas as pd


# Paths relative to the src directory so the script can be run from inside src
DATA_PATH = (Path(__file__).resolve().parent / "../data/master_dataset.csv").resolve()
OUT_PATH = (Path(__file__).resolve().parent / "../data/clean_master_dataset.csv").resolve()

# Mapping of possible short country codes to ISO3
COUNTRY_MAP = {
    "US": "USA",
    "IN": "IND",
    "BR": "BRA",
    "UK": "GBR",
    "CN": "CHN",
}


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    print("===== BASIC INFO =====")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df


def fix_dates(df: pd.DataFrame) -> pd.DataFrame:
    print("\nFixing month format...")
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    invalid_dates = df["month"].isna().sum()
    print(f"Invalid month rows: {invalid_dates}")
    if invalid_dates:
        df = df.dropna(subset=["month"])
        print(f"Dropped {invalid_dates} rows with invalid months.")
    # Normalize to month start (monthly frequency)
    df["month"] = df["month"].dt.to_period("M").dt.to_timestamp()
    return df


def standardize_country(df: pd.DataFrame) -> pd.DataFrame:
    print("\nFixing country codes...")
    df = df.copy()
    df["country"] = df["country"].replace(COUNTRY_MAP)
    print(f"Unique countries: {df['country'].nunique()}")
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    print("\nChecking duplicates...")
    df = df.copy()
    duplicates = df.duplicated(subset=["country", "month"]).sum()
    print(f"Duplicate rows: {duplicates}")
    if duplicates:
        df = df.drop_duplicates(subset=["country", "month"])
        print("Duplicates removed.")
    return df


def drop_sparse_columns(df: pd.DataFrame, threshold_ratio: float = 0.6) -> pd.DataFrame:
    print("\nMissing values per column:")
    missing = df.isna().sum()
    print(missing)
    threshold = len(df) * threshold_ratio
    drop_cols = missing[missing > threshold].index.tolist()
    if drop_cols:
        print(f"\nDropping columns with >{int(threshold_ratio*100)}% missing: {drop_cols}")
        df = df.drop(columns=drop_cols)
    return df


def interpolate_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate and forward-fill numeric columns within each country."""
    numeric_cols = [c for c in df.columns if c not in {"country", "month"}]
    if not numeric_cols:
        return df
    df = df.copy()
    df[numeric_cols] = (
        df.groupby("country")[numeric_cols]
        .apply(lambda g: g.interpolate().ffill())
        .reset_index(level=0, drop=True)
    )
    return df


def add_scaling(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if c not in {"country", "month"}]
    df = df.copy()
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std() if df[col].std() else 1
        df[f"{col}_z"] = (df[col] - mean) / std
    return df


def summarize(df: pd.DataFrame) -> None:
    print("\n===== SUMMARY =====")
    print(f"Shape: {df.shape}")
    print(f"Countries ({df['country'].nunique()}): {sorted(df['country'].unique())}")
    print(f"Date range: {df['month'].min()} to {df['month'].max()}")
    print("\nRemaining missing values per column:")
    print(df.isna().sum())
    print("\nSample rows:")
    print(df.head(10))


def main() -> None:
    print("Script started")
    df = load_dataset(DATA_PATH)

    # Ensure required columns exist
    if "country" not in df.columns:
        raise KeyError("Missing country column")

    # Post-load shape
    print(f"Loaded dataset shape: {df.shape}")

    df = fix_dates(df)
    df = standardize_country(df)
    df = drop_duplicates(df)
    df = df.sort_values(by=["country", "month"]).reset_index(drop=True)
    df = drop_sparse_columns(df, threshold_ratio=0.6)
    df = interpolate_numeric(df)
    df = add_scaling(df)

    # Additional summaries
    print(f"Number of unique countries: {df['country'].nunique()}")
    print(f"Date range after cleaning: {df['month'].min()} to {df['month'].max()}")
    print("Missing values summary after cleaning:")
    print(df.isna().sum())

    summarize(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print("Clean dataset saved successfully")
    print(f"\n✅ CLEAN DATASET SAVED: {OUT_PATH}")


if __name__ == "__main__":
    main()
