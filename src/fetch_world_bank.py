import time
from pathlib import Path
import pandas as pd
import requests

BASE_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
DATE_RANGE = "2000:2024"

COUNTRIES = [
    "USA","CAN","GBR","DEU","FRA","ESP","ITA","AUS","JPN","KOR",
    "CHN","IND","BRA","MEX","ARG","TUR","ZAF","RUS","IDN","SAU",
    "ARE","SGP","MYS","THA","PHL","VNM","POL","CZE","HUN","ROU"
]

INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "FP.CPI.TOTL.ZG": "inflation",
    "SL.UEM.TOTL.ZS": "unemployment",
    "FR.INR.RINR": "interest_rate",
    "NE.EXP.GNFS.ZS": "exports",
    "NE.IMP.GNFS.ZS": "imports",
}

def fetch_indicator(country, indicator):
    url = BASE_URL.format(country=country, indicator=indicator)
    params = {"date": DATE_RANGE, "format": "json", "per_page": 20000}

    try:
        r = requests.get(url, params=params, timeout=60)
        data = r.json()
    except:
        return []

    if not isinstance(data, list) or len(data) < 2:
        return []

    rows = []
    for item in data[1]:
        if item["value"] is None:
            continue

        rows.append({
            "country": country,
            "year": int(item["date"]),
            "indicator": indicator,
            "value": item["value"]
        })

    return rows


def main():
    all_data = []

    for country in COUNTRIES:
        print(f"Fetching {country}")
        for indicator in INDICATORS:
            all_data.extend(fetch_indicator(country, indicator))
            time.sleep(0.2)

    df = pd.DataFrame(all_data)

    if df.empty:
        print("❌ No data fetched")
        return

    # Pivot
    df = df.pivot_table(
        index=["country", "year"],
        columns="indicator",
        values="value",
        aggfunc="mean"
    ).reset_index()

    df.columns.name = None
    df = df.rename(columns=INDICATORS)

    # Convert year → datetime
    df["year"] = pd.to_datetime(df["year"], format="%Y")

    # SMART monthly conversion
    df_list = []

    for country in df["country"].unique():
        temp = df[df["country"] == country].copy()
        temp = temp.set_index("year")

        idx = pd.date_range(start=temp.index.min(), end=temp.index.max(), freq="MS")
        temp = temp.reindex(idx)

        temp["country"] = country
        temp = temp.interpolate(method="linear")

        temp = temp.reset_index().rename(columns={"index": "month"})
        df_list.append(temp)

    df = pd.concat(df_list, ignore_index=True)

    # Sort
    df = df.sort_values(["country", "month"])

    # Fill missing
    df = df.ffill().bfill()

    # Final columns
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
        ]
    ]

    # Save
    path = Path(__file__).resolve().parents[1] / "data" / "world_bank_data.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)

    print("\n✅ SUCCESS")
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())


if __name__ == "__main__":
    main()