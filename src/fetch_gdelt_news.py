from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


COUNTRIES = ["USA", "IND", "GBR", "CHN", "BRA"]
START = "2015-01-01"
END = "2024-12-31"


def build_dataframe() -> pd.DataFrame:
    """Generate synthetic monthly news sentiment data."""
    dates = pd.date_range(start=START, end=END, freq="MS")
    records = []

    for country in COUNTRIES:
        scores = np.random.normal(loc=0.0, scale=0.3, size=len(dates))
        scores = np.clip(scores, -1, 1)
        for dt, score in zip(dates, scores):
            records.append(
                {
                    "date": dt,
                    "country": country,
                    "sentiment_score": float(score),
                }
            )

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(["country", "date"]).reset_index(drop=True)
    return df


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    df = build_dataframe()
    project_root = Path(__file__).resolve().parents[1]
    out_path = project_root / "data" / "news_sentiment.csv"
    save_csv(df, out_path)
    print("Generated synthetic news sentiment data successfully")


if __name__ == "__main__":
    main()
