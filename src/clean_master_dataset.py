from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "master_dataset.csv"
OUT_PATH  = BASE_DIR / "data" / "clean_master_dataset.csv"

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
COUNTRY_MAP = {"US":"USA","IN":"IND","BR":"BRA","UK":"GBR","CN":"CHN"}

COLUMN_RENAMES = {
    "inflation_cpi_pct":  "inflation",
    "unemployment_pct":   "unemployment",
    "gdp_current_usd":    "gdp_growth",
    "exports_pct_gdp":    "exports",
    "imports_pct_gdp":    "imports",
}

BASE_COLS = ["gdp_growth", "inflation", "unemployment", "interest_rate"]

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape}")
    return df

# ─────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────
def fix_dates(df):
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df.dropna(subset=["month"])
    df["year"]  = df["month"].dt.year
    df["month"] = df["month"].dt.month
    return df

def standardize_country(df):
    df = df.copy()
    df["country"] = df["country"].replace(COUNTRY_MAP)
    return df

def rename_columns(df):
    return df.rename(columns=COLUMN_RENAMES)

def remove_duplicates(df):
    return df.drop_duplicates(subset=["country","year","month"])

def force_numeric(df):
    df = df.copy()

    # 🔥 Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    for col in df.columns:
        if col == "country":
            continue

        # 🔥 If column is accidentally a DataFrame → fix it
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def fill_missing(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df.groupby("country")[numeric_cols].transform(
        lambda x: x.interpolate().ffill().bfill()
    )
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df

def add_interest_rate(df):
    df = df.copy()
    if "interest_rate" not in df.columns:
        if "inflation" in df.columns:
            df["interest_rate"] = df["inflation"] + 2
        else:
            df["interest_rate"] = 5.0
    return df

# ─────────────────────────────────────────────
# FEATURE ENGINEERING (MATCH TRAIN MODEL)
# ─────────────────────────────────────────────
def add_lag_features(df):
    df = df.sort_values(["country","year","month"]).copy()

    for col in BASE_COLS:
        if col in df.columns:
            for lag in [1,3,6]:
                df[f"{col}_lag{lag}"] = df.groupby("country")[col].shift(lag)

    return df

def add_rolling_features(df):
    df = df.copy()

    for col in BASE_COLS:
        if col in df.columns:
            df[f"{col}_roll3"] = df.groupby("country")[col].rolling(3).mean().reset_index(0, drop=True)
            df[f"{col}_roll6"] = df.groupby("country")[col].rolling(6).mean().reset_index(0, drop=True)

    return df

def add_momentum(df):
    df = df.copy()

    for col in BASE_COLS:
        if col in df.columns:
            df[f"{col}_momentum"] = df[col] - df.groupby("country")[col].shift(3)

    return df

def add_stress_index(df):
    df = df.copy()

    if set(BASE_COLS).issubset(df.columns):
        df["stress_index"] = (
            df["inflation"] * 0.4 +
            df["unemployment"] * 0.3 -
            df["gdp_growth"] * 0.3
        )

    return df

# ─────────────────────────────────────────────
# LABEL
# ─────────────────────────────────────────────
def create_risk_label(df):
    df = df.copy()

    conditions = [
        df["inflation"] > 7,
        df["gdp_growth"] < 0,
        df["unemployment"] > 10,
        df["interest_rate"] > 8,
    ]

    stress = sum(c.astype(int) for c in conditions)
    df["risk_label"] = (stress >= 2).astype(int)

    print("Risk distribution:")
    print(df["risk_label"].value_counts())

    return df

# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    df = load_dataset(DATA_PATH)

    df = fix_dates(df)
    df = standardize_country(df)
    df = rename_columns(df)
    df = remove_duplicates(df)
    df = force_numeric(df)
    df = add_interest_rate(df)
    df = fill_missing(df)

    # FEATURES (CRITICAL)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_momentum(df)
    df = add_stress_index(df)

    df = create_risk_label(df)

    # Final fill
    for col in df.columns:
        if col != "country":
            df[col] = df[col].fillna(df[col].median())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\n✅ Saved clean dataset → {OUT_PATH}")
    print(f"Rows: {len(df)} | Countries: {df['country'].nunique()}")

# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()