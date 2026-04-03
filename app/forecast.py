"""
forecast.py — Model-driven GDP forecasting (global and country-level).
"""

import pandas as pd
import numpy as np
from data_api import load_dataset, load_model, load_feature_cols, get_latest, get_country_series
from utils import safe_mean


def _build_features(row: pd.Series, feature_cols: list) -> pd.DataFrame:
    data = {}

    for col in feature_cols:

        # lag + macro features
        if col in row.index:
            data[col] = row[col]

        # country one-hot (IMPORTANT FIX)
        elif col.startswith("country_"):
            country_name = col.replace("country_", "")
            data[col] = 1.0 if row["country"] == country_name else 0.0

        else:
            data[col] = 0.0

    return pd.DataFrame([data])[feature_cols]


def forecast_country(country: str) -> dict:
    """
    Return forecast dict for a single country:
    {country, latest_gdp, predicted_gdp, delta, confidence, date}
    """
    df           = load_dataset()
    model        = load_model()
    feature_cols = load_feature_cols()

    series = get_country_series(df, country)
    if series.empty:
        return {}
    
    if len(series) < 2:
        return {}

    latest = series.iloc[-2]
    X = _build_features(latest, feature_cols)
    
    print(country, X.iloc[0][:5])

    predicted = float(model.predict(X)[0])
    current   = float(latest.get("gdp_growth", 0.0))
    delta     = predicted - current

    # Confidence: proxy from model R² and prediction magnitude reasonableness
    from data_api import load_metrics
    metrics = load_metrics()
    r2      = metrics.get("r2", 0.5)
    rmse    = metrics.get("rmse", 1.0)

    # Confidence score 0–100
    confidence = round(float(np.clip(r2 * 100, 0, 100)), 1)

    return {
        "country":       country,
        "date":          str(latest["month"].date()) if hasattr(latest["month"], "date") else str(latest["month"]),
        "current_gdp":   round(current, 3),
        "predicted_gdp": round(predicted, 3),
        "delta":         round(delta, 3),
        "confidence":    confidence,
        "rmse":          round(rmse, 4),
    }


def forecast_all() -> pd.DataFrame:
    """Return forecast for all countries as a DataFrame."""
    df       = load_dataset()
    countries = df["country"].unique().tolist()

    records = []
    for c in countries:
        result = forecast_country(c)
        if result:
            records.append(result)

    return pd.DataFrame(records).sort_values("predicted_gdp", ascending=False).reset_index(drop=True)


def forecast_timeseries(country: str) -> pd.DataFrame:
    """
    Return historical GDP growth + model-fitted values for charting.
    """
    df           = load_dataset()
    model        = load_model()
    feature_cols = load_feature_cols()

    series = get_country_series(df, country)
    if series.empty:
        return pd.DataFrame()

    rows = []
    for _, row in series.iterrows():
        try:
            X    = _build_features(row, feature_cols)
            pred = float(model.predict(X)[0])
        except Exception:
            pred = np.nan

        rows.append({
            "month":        row["month"],
            "actual_gdp":   row.get("gdp_growth", np.nan),
            "model_fitted": pred,
        })

    return pd.DataFrame(rows)