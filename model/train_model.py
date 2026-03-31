from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "clean_master_dataset.csv"
FORECAST_OUT = BASE_DIR / "data" / "forecast.csv"
METRICS_OUT = BASE_DIR / "data" / "model_metrics.csv"


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"country", "month"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values(["country", "month"]).reset_index(drop=True)
    return df


def pick_target(df: pd.DataFrame) -> str:
    # Prefer inflation if present; else risk_score; else last numeric as fallback
    if "inflation_cpi_pct" in df.columns:
        return "inflation_cpi_pct"
    if "risk_score" in df.columns:
        return "risk_score"
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns available for target selection")
    return numeric_cols[-1]


def prepare_features(df: pd.DataFrame, target: str) -> (pd.DataFrame, pd.Series, List[str]):
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in {"country", "month", target}]
    X = df[feature_cols]
    y = df[target]
    return X, y, feature_cols


def time_split(df: pd.DataFrame, frac: float = 0.8):
    n = len(df)
    split_idx = int(n * frac)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def train_model(X_train, y_train) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return rmse, mae


def build_forecast(model, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    forecasts = []
    for country, group in df.groupby("country"):
        if group.empty:
            continue
        last_row = group.sort_values("month").iloc[-1]
        base_month = pd.Period(last_row["month"], freq="M")
        feature_values = last_row[feature_cols].values.reshape(1, -1)
        for i in range(1, 7):
            future_month = (base_month + i).to_timestamp("M")
            pred = model.predict(feature_values)[0]
            forecasts.append({"country": country, "month": future_month, "prediction": pred})
    return pd.DataFrame(forecasts)


def main():
    df = load_data(DATA_PATH)
    df = df.sort_values(["country", "month"]).reset_index(drop=True)

    target = pick_target(df)
    # Drop rows where target missing; fill others per country
    df = df.dropna(subset=[target])
    df = df.groupby("country", group_keys=False).apply(lambda x: x.ffill().bfill()).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("Dataset is empty after preprocessing")

    print(f"[INFO] Dataset shape after cleaning: {df.shape}")

    X, y, feature_cols = prepare_features(df, target)
    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")

    train_df, test_df = time_split(pd.concat([df[["country", "month"]], X, y], axis=1))
    X_train = train_df[feature_cols]
    y_train = train_df[target]
    X_test = test_df[feature_cols]
    y_test = test_df[target]

    model = train_model(X_train, y_train)
    rmse, mae = evaluate(model, X_test, y_test)

    forecast_df = build_forecast(model, df[["country", "month"] + feature_cols], feature_cols)

    # Save outputs
    FORECAST_OUT.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(FORECAST_OUT, index=False)

    metrics = {"rmse": rmse, "mae": mae}
    pd.DataFrame([metrics]).to_csv(METRICS_OUT, index=False)

    print(f"Saved forecast to {FORECAST_OUT}")
    print(f"Saved metrics to {METRICS_OUT}")


if __name__ == "__main__":
    main()
