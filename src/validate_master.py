from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "master_dataset.csv"


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    return df


def ensure_datetime(df: pd.DataFrame, column: str = "date") -> tuple[pd.DataFrame, int]:
    df = df.copy()
    df[column] = pd.to_datetime(df[column], errors="coerce")
    invalid = df[column].isna().sum()
    return df, invalid


def validate(df: pd.DataFrame) -> None:
    print("=== Basic Shape ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n=== Countries ===")
    print(f"Unique countries: {df['country'].nunique()}")
    print(f"Countries: {sorted(df['country'].unique())}")

    print("\n=== Date Range ===")
    print(f"Start: {df['date'].min()}")
    print(f"End:   {df['date'].max()}")

    print("\n=== Missing Values per Column ===")
    print(df.isna().sum())

    print("\n=== Duplicate Rows (country + date) ===")
    dup_mask = df.duplicated(subset=["country", "date"])
    dup_count = dup_mask.sum()
    print(f"Duplicate count: {dup_count}")
    if dup_count:
        print(df.loc[dup_mask, ["country", "date"]].head())

    print("\n=== Sample Rows ===")
    print(df.head(10))


def main() -> None:
    df = load_dataset(DATA_PATH)
    df, invalid_dates = ensure_datetime(df, column="date")

    if invalid_dates:
        print(f"WARNING: {invalid_dates} rows have invalid or missing dates after parsing.")

    validate(df)


if __name__ == "__main__":
    main()
