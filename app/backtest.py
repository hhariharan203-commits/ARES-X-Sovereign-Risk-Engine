import pandas as pd
import numpy as np
from data_api import load_dataset, get_country_series
from forecast import forecast_country

def backtest_country(country):
    df = load_dataset()
    series = get_country_series(df, country)

    if len(series) < 6:
        return None

    actuals = []
    preds = []

    for i in range(3, len(series)):
        row = series.iloc[i-1]

        try:
            pred = forecast_country(country)["predicted_gdp"]
        except:
            continue

        actual = series.iloc[i]["gdp_growth"]

        actuals.append(actual)
        preds.append(pred)

    if not actuals:
        return None

    actuals = np.array(actuals)
    preds = np.array(preds)

    rmse = np.sqrt(np.mean((actuals - preds)**2))
    mae = np.mean(np.abs(actuals - preds))

    direction_acc = np.mean(
        np.sign(actuals) == np.sign(preds)
    ) * 100

    return {
        "country": country,
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "direction_accuracy": round(direction_acc, 1)
    }
