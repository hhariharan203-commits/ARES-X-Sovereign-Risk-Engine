from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "master_dataset.csv"
OUT_PATH  = BASE_DIR / "data" / "clean_master_dataset.csv"

COUNTRY_MAP = {"US":"USA","IN":"IND","BR":"BRA","UK":"GBR","CN":"CHN"}

COLUMN_RENAMES = {
    "inflation_cpi_pct": "inflation",
    "unemployment_pct":  "unemployment",
    "exports_pct_gdp":   "exports",
    "imports_pct_gdp":   "imports",
}

BASE_COLS = ["gdp_growth","inflation","unemployment","interest_rate"]

# ─────────────────────────────
def log(msg):
    print(f"[PIPELINE] {msg}")

# ─────────────────────────────
def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    df = pd.read_csv(path)
    log(f"Loaded dataset: {df.shape}")

    if df.empty:
        raise ValueError("Dataset is empty")

    return df

# ─────────────────────────────
def fix_dates(df):
    df = df.copy()

    if "month" not in df.columns:
        raise ValueError("Missing 'month' column")

    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df.dropna(subset=["month"])

    df["year"]  = df["month"].dt.year
    df["month"] = df["month"].dt.month

    return df

# ─────────────────────────────
def standardize_country(df):
    df = df.copy()

    if "country" not in df.columns:
        raise ValueError("Missing 'country' column")

    df["country"] = df["country"].astype(str).str.upper()
    df["country"] = df["country"].replace(COUNTRY_MAP)

    return df

# ─────────────────────────────
def rename_columns(df):
    return df.rename(columns=COLUMN_RENAMES)

# ─────────────────────────────
def force_numeric(df):
    df = df.loc[:, ~df.columns.duplicated()].copy()

    for col in df.columns:
        if col not in ["country"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# ─────────────────────────────
def compute_gdp_growth(df):
    df = df.sort_values(["country","year","month"]).copy()

    if "gdp_current_usd" in df.columns:
        log("Computing GDP growth from raw GDP")

        df["gdp_growth"] = (
            df.groupby("country")["gdp_current_usd"]
            .pct_change(12)
        ) * 100

        df.drop(columns=["gdp_current_usd"], inplace=True)

    # fallback if missing
    if "gdp_growth" not in df.columns:
        log("WARNING: GDP growth missing → creating proxy")
        df["gdp_growth"] = 0

    # clean extreme values
    df["gdp_growth"] = df["gdp_growth"].replace([np.inf, -np.inf], np.nan)
    df["gdp_growth"] = df["gdp_growth"].clip(-25, 25)

    # robust fill
    df["gdp_growth"] = df.groupby("country")["gdp_growth"].transform(lambda x: x.bfill().ffill())

    return df

# ─────────────────────────────
def add_interest_rate(df):
    if "interest_rate" not in df.columns:
        log("Interest rate missing → creating proxy")
        df["interest_rate"] = df["inflation"] + 2

    return df

# ─────────────────────────────
def fill_missing(df):
    df = df.copy()

    num = df.select_dtypes(include=np.number).columns

    # country-level interpolation
    df[num] = df.groupby("country")[num].transform(
        lambda x: x.interpolate(limit_direction="both")
    )

    # fallback global median
    df[num] = df[num].fillna(df[num].median())

    return df

# ─────────────────────────────
def handle_outliers(df):
    df = df.copy()

    for col in ["inflation","unemployment","interest_rate"]:
        if col in df.columns:
            df[col] = df[col].clip(df[col].quantile(0.01), df[col].quantile(0.99))

    return df

# ─────────────────────────────
def add_features(df):
    df = df.sort_values(["country","year","month"]).copy()

    for col in BASE_COLS:
        if col in df.columns:

            df[f"{col}_lag1"] = df.groupby("country")[col].shift(1)
            df[f"{col}_lag3"] = df.groupby("country")[col].shift(3)

            df[f"{col}_roll3"] = df.groupby("country")[col].transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            )

            df[f"{col}_momentum"] = df[col] - df.groupby("country")[col].shift(3)

    return df

# ─────────────────────────────
def create_label(df):
    df = df.copy()

    stress = (
        (df["inflation"] > 7).astype(int) +
        (df["gdp_growth"] < 0).astype(int) +
        (df["unemployment"] > 10).astype(int) +
        (df["interest_rate"] > 8).astype(int)
    )

    df["risk_now"] = (stress >= 2).astype(int)

    # future prediction
    df["risk_label"] = df.groupby("country")["risk_now"].shift(-3)

    df = df.dropna(subset=["risk_label"])

    log("Risk distribution:")
    print(df["risk_label"].value_counts())

    return df

# ─────────────────────────────
def validate(df):
    log("Running data validation checks")

    if df.isna().sum().sum() > 0:
        raise ValueError("NaNs still exist after processing")

    if (df.select_dtypes(include=np.number) == np.inf).any().any():
        raise ValueError("Infinite values detected")

    log("Validation passed")

# ─────────────────────────────
def main():
    log("START PIPELINE")

    df = load_dataset(DATA_PATH)

    df = fix_dates(df)
    df = standardize_country(df)
    df = rename_columns(df)
    df = force_numeric(df)

    df = compute_gdp_growth(df)
    df = add_interest_rate(df)
    df = fill_missing(df)
    df = handle_outliers(df)

    df = add_features(df)
    df = create_label(df)

    df = df.fillna(df.median(numeric_only=True))

    validate(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    log("CLEAN DATA READY (REAL WORLD PRODUCTION)")

# ─────────────────────────────
if __name__ == "__main__":
    main()