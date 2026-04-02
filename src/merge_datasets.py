from pathlib import Path
import pandas as pd


def main():
    base_path = Path(__file__).resolve().parents[1] / "data"

    # Load datasets
    wb = pd.read_csv(base_path / "world_bank_data.csv")
    vix = pd.read_csv(base_path / "vix_data.csv")  # your file name
    news = pd.read_csv(base_path / "news_sentiment.csv")

    # Convert to datetime
    wb["month"] = pd.to_datetime(wb["month"])
    vix["month"] = pd.to_datetime(vix["month"])
    news["month"] = pd.to_datetime(news["month"])

    print("✅ Data loaded")

    # Merge datasets
    df = wb.merge(vix, on="month", how="left")
    df = df.merge(news, on="month", how="left")

    # Sort properly
    df = df.sort_values(["country", "month"])

    # Fill missing values per country
    df = (
        df.groupby("country", as_index=False)
        .apply(lambda x: x.ffill().bfill())
        .reset_index(drop=True)
    )

    # Final column order (clean)
    df = df[
        [
            "country",
            "month",
            "gdp_growth",
            "inflation",
            "unemployment",
            "interest_rate",
            "exports",
            "imports",
            "vix",
            "sentiment_mean",
        ]
    ]

    # Save final dataset
    path = base_path / "master_dataset.csv"
    df.to_csv(path, index=False)

    print("\n✅ MASTER DATASET CREATED")
    print("Columns:", df.columns.tolist())
    print("Rows:", len(df))
    print(df.head())


if __name__ == "__main__":
    main()