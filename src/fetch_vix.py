from pathlib import Path
import pandas as pd

URL = "https://raw.githubusercontent.com/datasets/finance-vix/master/data/vix-daily.csv"

def main():
    try:
        df = pd.read_csv(URL)
    except Exception as e:
        print("❌ Failed to fetch VIX:", e)
        return

    print("Columns found:", df.columns.tolist())  # DEBUG

    # Normalize column names (important fix)
    df.columns = [col.lower().strip() for col in df.columns]

    # Handle different possible column names
    if "date" not in df.columns:
        print("❌ date column missing")
        return

    if "vix close" in df.columns:
        value_col = "vix close"
    elif "close" in df.columns:
        value_col = "close"
    else:
        print("❌ VIX value column not found")
        return

    df = df[["date", value_col]]
    df.columns = ["date", "vix"]

    # Convert date
    df["date"] = pd.to_datetime(df["date"])

    # Convert daily → monthly
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df = df.groupby("month")["vix"].mean().reset_index()

    # Save
    path = Path(__file__).resolve().parents[1] / "data" / "vix.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)

    print("✅ VIX SUCCESS")
    print(df.head())


if __name__ == "__main__":
    main()