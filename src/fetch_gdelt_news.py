from pathlib import Path
import pandas as pd

BASE_URL = "http://data.gdeltproject.org/events/{date}.export.CSV.zip"

# Fetch multiple days (you can expand later)
DATES = [
    "20240101", "20240201", "20240301",
    "20240401", "20240501", "20240601"
]

def fetch_day(date):
    url = BASE_URL.format(date=date)
    try:
        df = pd.read_csv(url, sep='\t', header=None, low_memory=False)
        df = df[[1, 34]]
        df.columns = ["date", "tone"]
        return df
    except:
        print(f"❌ Failed: {date}")
        return pd.DataFrame()

def main():
    all_data = []

    for d in DATES:
        print(f"Fetching {d}")
        df = fetch_day(d)
        if not df.empty:
            all_data.append(df)

    if not all_data:
        print("❌ No data fetched")
        return

    df = pd.concat(all_data, ignore_index=True)

    # Clean
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df.dropna()

    # Monthly aggregation
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    df = df.groupby("month")["tone"].mean().reset_index()
    df = df.rename(columns={"tone": "sentiment_mean"})

    df = df.sort_values("month")

    # Save
    path = Path(__file__).resolve().parents[1] / "data" / "news_sentiment.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)

    print("\n✅ FINAL NEWS DATA READY")
    print("Rows:", len(df))
    print(df.head())


if __name__ == "__main__":
    main()