"""
train_model.py
--------------
Leakage-safe GDP forecasting model (production-ready)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)


# CONFIG
BASE_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_PATH / "data" / "master_dataset.csv"
MODEL_PATH = BASE_PATH / "models" / "model.pkl"

TRAIN_RATIO = 0.8
CV_SPLITS = 5

XGB_PARAMS = dict(
    n_estimators=100,
    max_depth=2,
    learning_rate=0.07,
    subsample=0.6,
    colsample_bytree=0.6,
    reg_alpha=0.7,
    reg_lambda=3.0,
    random_state=42,
    n_jobs=-1,
)


def log(msg):
    print(msg, flush=True)


def main():

    # LOAD
    df = pd.read_csv(DATA_PATH, parse_dates=["month"])
    df = df.sort_values(["country", "month"]).reset_index(drop=True)

    # TARGET (t+1)
    df["target"] = df.groupby("country")["gdp_growth"].shift(-1)
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    log(f"✅ Data loaded: {df.shape}")

    # BASE FEATURES
    BASE_MACRO = [
        "gdp_growth",
        "inflation",
        "unemployment",
        "interest_rate",
        "exports",
        "imports",
        "vix",
        "sentiment_mean"
    ]

    # SHIFT
    for lag in [1, 2, 3]:
        for col in BASE_MACRO:
            df[f"{col}_s{lag}"] = df.groupby("country")[col].shift(lag)

    # DERIVED
    df["inflation_trend"] = df["inflation_s1"] - df["inflation_s3"]
    df["trade_balance"] = df["exports_s1"] - df["imports_s1"]
    df["vix_change"] = df["vix_s1"] - df["vix_s2"]

    # INTERACTIONS
    df["inflation_x_rate"] = df["inflation_s1"] * df["interest_rate_s1"]
    df["vix_x_inflation"] = df["vix_s1"] * df["inflation_s1"]

    # CLEAN
    df = df.dropna().reset_index(drop=True)

    # REMOVE LEAKAGE
    RAW_COLS = set(BASE_MACRO)
    LEGACY_LAGS = {c for c in df.columns if "_lag" in c}

    EXCLUDE = RAW_COLS | LEGACY_LAGS | {"target", "country", "month"}

    features = [c for c in df.columns if c not in EXCLUDE]
    X = df[features]
    y = df["target"]

    log(f"✅ Features: {len(features)}")

    # SPLIT
    cutoff = df["month"].iloc[int(len(df) * TRAIN_RATIO)]
    train_mask = df["month"] <= cutoff
    test_mask = df["month"] > cutoff

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # CV
    log("\n🔄 Time-series CV")
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

    for i, (tr, val) in enumerate(tscv.split(X_train), 1):
        m = XGBRegressor(**XGB_PARAMS)
        m.fit(X_train.iloc[tr], y_train.iloc[tr])
        preds = m.predict(X_train.iloc[val])
        rmse = root_mean_squared_error(y_train.iloc[val], preds)
        log(f"Fold {i}: RMSE={rmse:.4f}")

    # FINAL TRAIN
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)

    # PREDICT
    preds = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    log("\n📊 TEST RESULTS")
    log(f"RMSE: {rmse:.4f}")
    log(f"MAE : {mae:.4f}")
    log(f"R2  : {r2:.4f}")

    # SHUFFLE TEST
    log("\n🔍 Shuffle Test (Leakage Check)")

    y_shuffled = np.random.permutation(y_train.values)

    m_shuff = XGBRegressor(**XGB_PARAMS)
    m_shuff.fit(X_train, y_shuffled)

    preds_shuff = m_shuff.predict(X_test)
    r2_shuff = r2_score(y_test, preds_shuff)

    log(f"Shuffled R2: {r2_shuff:.4f}")

    if r2_shuff < 0.1:
        log("✅ PASSED — no leakage")
    else:
        log("❌ WARNING — possible leakage")

    # OVERFITTING CHECK
    log("\n🔍 Overfitting Check")

    train_preds = model.predict(X_train)

    train_rmse = root_mean_squared_error(y_train, train_preds)
    gap = (rmse - train_rmse) / train_rmse

    log(f"Train RMSE: {train_rmse:.4f}")
    log(f"Test RMSE : {rmse:.4f}")
    log(f"Gap       : {gap*100:.1f}%")

    if gap < 1.0:
        log("✅ Acceptable generalization")
    else:
        log("⚠️ High overfitting")

    # SAVE MODEL
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    log(f"\n💾 Model saved: {MODEL_PATH}")

    # SAVE METRICS
    OUTPUT_PATH = BASE_PATH / "outputs"
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "train_rmse": float(train_rmse),
    }

    with open(OUTPUT_PATH / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    log("💾 Metrics saved -> outputs/model_metrics.json")

    # SAVE FEATURE COLUMNS
    with open(MODEL_PATH.parent / "feature_cols.json", "w") as f:
        json.dump(features, f, indent=4)

    log("💾 Feature columns saved -> models/feature_cols.json")


if __name__ == "__main__":
    main()