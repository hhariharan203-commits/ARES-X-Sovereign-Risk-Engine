from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


DATA_PATH = (Path(__file__).resolve().parent / "../data/clean_master_dataset.csv").resolve()
MODEL_PATH = (Path(__file__).resolve().parent / "../models/model.pkl").resolve()
FEATIMP_PATH = (Path(__file__).resolve().parent / "../data/feature_importance.csv").resolve()
FEATURE_COLS_PATH = (Path(__file__).resolve().parent / "../models/feature_cols.json").resolve()
THRESHOLDS_PATH = (Path(__file__).resolve().parent / "../models/risk_thresholds.json").resolve()
BASE_DIR_OS = os.path.dirname(os.path.dirname(__file__))
METRICS_PATH = Path(BASE_DIR_OS) / "outputs" / "model_metrics.json"

BASE_FEATURES = [
    "exports_pct_gdp",
    "imports_pct_gdp",
    "gdp_current_usd",
    "gdp_growth_pct",
    "unemployment_rate_pct",
    "interest_rate_pct",
]
LAGS = [1, 3, 6]


def ensure_directories() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_COLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATIMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["month"])
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values(["country", "month"]).reset_index(drop=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for country, g in df.groupby("country"):
        g = g.sort_values("month").copy()

        # Lag features for base economic variables
        for col in BASE_FEATURES:
            if col not in g.columns:
                g[col] = np.nan
            for lag in LAGS:
                g[f"{col}_lag{lag}"] = g[col].shift(lag)

        # Rolling mean (3 months) for base variables
        for col in BASE_FEATURES:
            g[f"{col}_roll3"] = g[col].rolling(window=3, min_periods=1).mean()
            g[f"{col}_vol6"] = g[col].rolling(window=6, min_periods=2).std()

        # Percentage change features for base variables
        for col in BASE_FEATURES:
            g[f"{col}_pct_change"] = g[col].pct_change()

        frames.append(g)

    df_feat = pd.concat(frames, ignore_index=True)

    def compute_crisis(df_in: pd.DataFrame, infl_thr: float, gdp_thr: float, exp_thr: float) -> pd.Series:
        return (
            (df_in["inflation_cpi_pct"] > infl_thr)
            | (df_in["gdp_current_usd_pct_change"] * 100 < gdp_thr)
            | (df_in["exports_pct_gdp_pct_change"] * 100 < exp_thr)
        ).astype(int)

    # Initial crisis definition
    df_feat["crisis_risk"] = compute_crisis(df_feat, 7, -2, -1)
    share = df_feat["crisis_risk"].mean()
    print("Crisis share (initial):", f"{share:.2%}")

    # Relax thresholds if too few crisis cases
    if share < 0.10:
        df_feat["crisis_risk"] = compute_crisis(df_feat, 6, -1, -0.5)
        share = df_feat["crisis_risk"].mean()
        print("Crisis share (relaxed 1):", f"{share:.2%}")
    if share < 0.10:
        df_feat["crisis_risk"] = compute_crisis(df_feat, 5, -0.5, -0.25)
        share = df_feat["crisis_risk"].mean()
        print("Crisis share (relaxed 2):", f"{share:.2%}")

    print(df_feat["crisis_risk"].value_counts(normalize=True))

    # Lead target: crisis risk 3 months ahead
    df_feat["crisis_risk_future"] = (
        df_feat.groupby("country")["crisis_risk"].shift(-3)
    )

    # Drop current-month crisis indicator to avoid leakage
    df_feat = df_feat.drop(columns=["crisis_risk"])

    # Remove all inflation-related columns to eliminate leakage
    infl_cols = [c for c in df_feat.columns if "inflation" in c]
    df_feat = df_feat.drop(columns=infl_cols)

    # Drop rows with missing values produced by lags/pct_change/lead target
    df_feat = df_feat.dropna(subset=["crisis_risk_future"])
    df_feat = df_feat.dropna().reset_index(drop=True)
    return df_feat


def train_model(df: pd.DataFrame) -> XGBClassifier:
    feature_cols = [
        c
        for c in df.columns
        if c not in {"country", "month", "crisis_risk", "crisis_risk_future"}
    ]
    # Persist feature schema for inference alignment
    with open(FEATURE_COLS_PATH, "w") as f:
        json.dump(feature_cols, f)

    df = df.sort_values(["country", "month"])
    df["month"] = pd.to_datetime(df["month"])

    train = df[df["month"] < "2020-01-01"]
    test = df[df["month"] >= "2020-01-01"]

    X_train = train[feature_cols]
    y_train = train["crisis_risk_future"]
    X_test = test[feature_cols]
    y_test = test["crisis_risk_future"]
    print(f"Train size: {len(train)}, Test size: {len(test)}")

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        class_weight={0: 1, 1: 5},
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.3).astype(int)
    acc = accuracy_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_prob))
    r2 = r2_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, zero_division=0)

    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob)

    print("Accuracy (predicting 3-month-ahead crisis):", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print("ROC-AUC:", roc)
    print("RMSE (probabilities):", rmse)
    print("R2 (probabilities):", r2)
    print("Classification Report:\n", report)

    # Risk thresholds based on percentiles to balance LOW/MEDIUM/HIGH
    train_probs = model.predict_proba(X_train)[:, 1]
    low_cut = float(np.percentile(train_probs, 40))
    high_cut = float(np.percentile(train_probs, 80))
    thresholds = {"low": low_cut, "high": high_cut}
    with open(THRESHOLDS_PATH, "w") as f:
        json.dump(thresholds, f)

    # Save artifacts
    joblib.dump(model, MODEL_PATH)
    feat_imp = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    feat_imp.to_csv(FEATIMP_PATH, index=False)

    base_dir = os.path.dirname(os.path.dirname(__file__))
    output_path = os.path.join(base_dir, "outputs", "model_metrics.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc),
    }
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved model metrics to {output_path}")

    return model


def main() -> None:
    ensure_directories()
    df = load_data()
    df_feat = add_features(df)
    train_model(df_feat)


if __name__ == "__main__":
    main()
