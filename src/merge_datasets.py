from pathlib import Path
import pandas as pd


def main():
    base_path = Path(__file__).resolve().parents[1] / "data"

    # =========================
    # LOAD DATA
    # =========================
    wb = pd.read_csv(base_path / "world_bank_data.csv")
    vix = pd.read_csv(base_path / "vix_data.csv")
    news = pd.read_csv(base_path / "news_sentiment.csv")

    # =========================
    # DATE HANDLING
    # =========================
    wb["month"] = pd.to_datetime(wb["month"])
    vix["month"] = pd.to_datetime(vix["month"])
    news["month"] = pd.to_datetime(news["month"])

    # Align all to month-end
    wb["month"] = wb["month"] + pd.offsets.MonthEnd(0)
    vix["month"] = vix["month"] + pd.offsets.MonthEnd(0)
    news["month"] = news["month"] + pd.offsets.MonthEnd(0)

    print("✅ Dates aligned")

    # =========================
    # MERGE DATASETS (IMPORTANT)
    # =========================
    df = wb.merge(vix, on="month", how="left")
    df = df.merge(news, on="month", how="left")

    print("✅ Data merged")

    # =========================
    # SORT DATA
    # =========================
    df = df.sort_values(["country", "month"])

    # =========================
    # FIX SENTIMENT (KEY STEP)
    # =========================
    df["sentiment_mean"] = df.groupby("country")["sentiment_mean"].ffill().bfill()

    # =========================
    # DROP ONLY CRITICAL MISSING
    # =========================
    df = df.dropna(subset=[
        "gdp_growth",
        "inflation",
        "unemployment",
        "interest_rate",
        "exports",
        "imports",
        "vix"
    ]).reset_index(drop=True)

    print("✅ Sentiment fixed + core data clean")

    # =========================
    # FILTER VALID PERIOD
    # =========================
    df = df[df["month"] >= "2014-01-01"]

    # =========================
    # FEATURE ENGINEERING (LAGS)
    # =========================
    features = [
        "gdp_growth",
        "inflation",
        "unemployment",
        "interest_rate",
        "exports",
        "imports",
        "vix",
        "sentiment_mean",
    ]

    for lag in [1, 2, 3]:
        for col in features:
            df[f"{col}_lag{lag}"] = df.groupby("country")[col].shift(lag)

    # Remove rows where lag not available
    df = df.dropna().reset_index(drop=True)

    print("✅ Lag features created")

    # =========================
    # FINAL DATASET
    # =========================
    final_cols = ["country", "month"] + features + [
        f"{col}_lag{lag}" for col in features for lag in [1, 2, 3]
    ]

    df = df[final_cols]

    # =========================
    # SAVE FILE
    # =========================
    output_path = base_path / "master_dataset.csv"
    df.to_csv(output_path, index=False)

    print("\n🔥 MASTER DATASET READY (PROFESSIONAL MODE)")
    print("Columns:", len(df.columns))
    print("Rows:", len(df))
    print(df.head())


if __name__ == "__main__":
    main()