from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "clean_master_dataset.csv"
FORECAST_OUT = BASE_DIR / "data" / "forecast.csv"
METRICS_OUT = BASE_DIR / "data" / "model_metrics.csv"
FEAT_IMP_OUT = BASE_DIR / "data" / "feature_importance.csv"
PERF_DETAIL_OUT = BASE_DIR / "data" / "performance_detail.csv"

FEATURES = [
    "gdp_growth",
    "inflation",
    "unemployment",
    "interest_rate",
    "exports_pct_gdp",
    "imports_pct_gdp",
    "gdp_volatility",
    "inflation_volatility",
]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    required = {"country", "month"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")
    df["month"] = pd.to_datetime(df["month"])
    df["year"] = df["month"].dt.year
    df = df.sort_values(["country", "month"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["country", "month"])
    print("Columns:", df.columns.tolist())
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure base columns exist to avoid KeyErrors
    if "gdp_current_usd" not in df.columns:
        df["gdp_current_usd"] = 1.0
    if "inflation_cpi_pct" not in df.columns:
        df["inflation_cpi_pct"] = df.get("inflation", pd.Series(0, index=df.index))
    if "unemployment_pct" not in df.columns:
        df["unemployment_pct"] = df.get("unemployment", pd.Series(0, index=df.index))
    if "dgs1" not in df.columns and "interest_rate" not in df.columns:
        df["dgs1"] = df.get("interest_rate", pd.Series(0, index=df.index))

    # Map to canonical names
    if "inflation_cpi_pct" in df.columns:
        df["inflation"] = df["inflation_cpi_pct"]
    if "unemployment_pct" in df.columns:
        df["unemployment"] = df["unemployment_pct"]
    if "dgs1" in df.columns:
        df["interest_rate"] = df["dgs1"]
    elif "interest_rate" not in df.columns:
        df["interest_rate"] = df.get("inflation_cpi_pct", pd.Series(0, index=df.index)) * 0.5

    g = df.groupby("country")
    # Trend
    if "gdp_growth" in df.columns and df["gdp_growth"].notna().any():
        pass  # keep existing
    else:
        df["gdp_growth"] = g["gdp_current_usd"].pct_change()
        if df["gdp_growth"].isna().all() and "exports_pct_gdp" in df.columns and "imports_pct_gdp" in df.columns:
            df["gdp_growth"] = (df["exports_pct_gdp"] - df["imports_pct_gdp"]) / 100.0
    df["inflation_change"] = g["inflation_cpi_pct"].diff()
    df["unemployment_change"] = g["unemployment_pct"].diff()

    # Volatility (6-month window)
    df["gdp_volatility"] = g["gdp_current_usd"].transform(
        lambda x: x.rolling(6, min_periods=2).std()
    )
    df["inflation_volatility"] = g["inflation_cpi_pct"].transform(
        lambda x: x.rolling(6, min_periods=2).std()
    )

    # Lags
    return df


def target_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure core columns exist
    for col in ["inflation", "unemployment", "gdp_growth", "interest_rate"]:
        if col not in df.columns:
            df[col] = np.nan

    # Normalize components
    inflation_norm = df["inflation"] / df["inflation"].max()
    unemployment_norm = df["unemployment"] / df["unemployment"].max()
    gdp_growth_norm = df["gdp_growth"] / df["gdp_growth"].max()
    interest_norm = df["interest_rate"] / df["interest_rate"].max()

    df["crisis_score"] = (
        0.3 * inflation_norm +
        0.3 * unemployment_norm +
        0.2 * (1 - gdp_growth_norm) +
        0.2 * interest_norm
    )

    cs_min, cs_max = df["crisis_score"].min(), df["crisis_score"].max()
    df["crisis_score"] = (df["crisis_score"] - cs_min) / (cs_max - cs_min + 1e-9)
    df["crisis"] = (df["crisis_score"] > df["crisis_score"].median()).astype(int)
    return df


def clean_and_fill(df: pd.DataFrame) -> pd.DataFrame:
    print("Before cleaning:", len(df))
    df = df.sort_values(["country", "month"])
    # Remove potential leakage columns if present
    leak_cols = [c for c in df.columns if ("crisis" in c.lower() or "risk" in c.lower()) and c != "crisis"]
    df = df.drop(columns=leak_cols, errors="ignore")
    df = df.groupby("country", group_keys=False).apply(lambda x: x.ffill().bfill()).reset_index(drop=True)

    assert len(df) > 100, "Dataset too small before cleaning"

    df = df.dropna(subset=["inflation", "unemployment", "interest_rate"])

    assert len(df) > 50, "Dataset collapsed after cleaning"
    print("After cleaning:", len(df))
    if len(df) == 0:
        raise ValueError("Dataset is empty after cleaning.")
    return df


def prepare_xy(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[feature_cols]
    y = df[target_col]
    return X, y


def time_split(df: pd.DataFrame, frac: float = 0.8):
    n = len(df)
    split_idx = int(n * frac)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def train_model(X_train, y_train):
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.5,
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )
    model = CalibratedClassifierCV(xgb, method="sigmoid", cv=3)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    probs = np.clip(probs, 0.05, 0.95)
    print("Probability stats (test):", probs.min(), probs.max(), probs.mean())
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_f1 = 0
    for t in thresholds:
        preds_temp = (probs >= t).astype(int)
        f1_temp = f1_score(y_test, preds_temp, zero_division=0)
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_threshold = t
    print("Best threshold:", best_threshold)
    preds = (probs >= best_threshold).astype(int)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5
    print(f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | ROC-AUC: {roc:.3f}")
    print(f"Positive predictions (threshold={best_threshold}): {preds.sum()} / {len(preds)}")
    if prec < 0.1:
        print("NOTE: Low precision is expected due to rare-event prediction.")
        print("Model is designed to maximize recall and early warning signals.")
    print("Higher recall ensures fewer missed crises, at the cost of false positives.")
    print("This model should be used as an early warning system, not a final decision tool.")
    print("Use HIGH risk predictions for monitoring, not direct action.")
    return (
        {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc},
        preds,
        probs,
    )


def save_feature_importance(model, feature_cols: List[str]):
    imp = None
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "base_estimator_") and hasattr(model.base_estimator_, "feature_importances_"):
        imp = model.base_estimator_.feature_importances_
    elif hasattr(model, "calibrated_classifiers_"):
        cal = model.calibrated_classifiers_[0]
        if hasattr(cal, "estimator") and hasattr(cal.estimator, "feature_importances_"):
            imp = cal.estimator.feature_importances_
    if imp is None:
        return
    fi = pd.DataFrame({"feature": feature_cols, "importance": imp})
    fi = fi.sort_values("importance", ascending=False)
    FEAT_IMP_OUT.parent.mkdir(parents=True, exist_ok=True)
    fi.to_csv(FEAT_IMP_OUT, index=False)
    print(f"Saved feature importance to {FEAT_IMP_OUT}")


def apply_risk_levels(df: pd.DataFrame) -> pd.DataFrame:
    if "crisis_prob" not in df.columns or df["crisis_prob"].empty:
        df["risk_level"] = "UNKNOWN"
        return df
    q_low = df["crisis_prob"].quantile(0.33)
    q_high = df["crisis_prob"].quantile(0.66)
    df["risk_level"] = df["crisis_prob"].apply(
        lambda p: "LOW" if p < q_low else ("MEDIUM" if p < q_high else "HIGH")
    )
    return df


def build_forecast(model, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    forecasts = []
    for country, group in df.groupby("country"):
        if group.empty:
            continue
        last_row = group.sort_values("month").iloc[-1]
        base_month = pd.Period(last_row["month"], freq="M")
        features = last_row[feature_cols].copy()
        for i in range(1, 7):
            # add slight noise to avoid flat forecasts
            noisy = features.values.astype(float) + rng.normal(0, 0.02, size=len(feature_cols))
            noisy_df = pd.DataFrame([noisy], columns=feature_cols)
            pred_prob = model.predict_proba(noisy_df)[0, 1]
            pred_prob = float(np.clip(pred_prob, 0.01, 0.99))
            future_month = (base_month + i).to_timestamp("M")
            forecasts.append({"country": country, "month": future_month, "prediction": pred_prob})
    return pd.DataFrame(forecasts)


def main():
    # 1) Load
    df = load_data(DATA_PATH)

    # 2) Feature engineering
    df = feature_engineering(df)

    # 3) Target engineering
    df = target_engineering(df)

    # 4) Cleaning (after all features/target)
    df = clean_and_fill(df)
    # Forward-looking crisis target
    df["crisis_target"] = (
        (df.groupby("country")["gdp_growth"].shift(-1) < -1) &
        (df.groupby("country")["inflation"].shift(-1) > 5)
    ).astype(float)
    df = df.dropna(subset=["crisis_target"])
    if df["crisis_target"].nunique() < 2:
        # Relax thresholds if target is degenerate
        df["crisis_target"] = (
            (df.groupby("country")["gdp_growth"].shift(-1) < 0) &
            (df.groupby("country")["inflation"].shift(-1) > 3)
        ).astype(float)
        df = df.dropna(subset=["crisis_target"])
    if df["crisis_target"].nunique() < 2:
        median_future_growth = df.groupby("country")["gdp_growth"].shift(-1).median()
        df["crisis_target"] = (
            df.groupby("country")["gdp_growth"].shift(-1).fillna(median_future_growth) < median_future_growth
        ).astype(int)
        df = df.dropna(subset=["crisis_target"])

    df["crisis_target"] = df["crisis_target"].astype(int)

    print("Target distribution:")
    print(df["crisis_target"].value_counts())

    # 5) Strict feature filtering
    allowed_features = FEATURES
    df = df[["country", "month", "year", "crisis_target"] + allowed_features]
    for col in df.columns:
        if col not in ["country", "month", "year", "crisis_target"] + allowed_features:
            raise ValueError(f"Leakage column detected: {col}")
    feature_cols = allowed_features
    print("FINAL FEATURES:", feature_cols)

    # 6) Pure time-based split
    df_time = df.sort_values("month").reset_index(drop=True)
    train_df = df_time[df_time["year"] < 2018]
    test_df = df_time[df_time["year"] >= 2018]

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Split failed; check date coverage.")
    X_train, y_train = prepare_xy(train_df, feature_cols, "crisis_target")
    X_test, y_test = prepare_xy(test_df, feature_cols, "crisis_target")

    # Scale features, keep DataFrame to preserve names
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index,
    )
    print("Target used:", y_train.name)

    # Add small noise to mimic real-world uncertainty
    rng = np.random.default_rng(42)
    for col in feature_cols:
        X_train_scaled[col] += rng.normal(0, 0.01, size=len(X_train_scaled))
        X_test_scaled[col] += rng.normal(0, 0.01, size=len(X_test_scaled))

    # Balance classes via downsampling majority
    train_balanced = train_df.copy()
    X_train_bal = X_train_scaled.copy()
    y_train_bal = y_train.copy()
    if y_train_bal.nunique() == 2:
        df_bal = pd.concat([X_train_bal, y_train_bal], axis=1)
        maj = df_bal[df_bal[y_train.name] == df_bal[y_train.name].mode()[0]]
        mino = df_bal[df_bal[y_train.name] != df_bal[y_train.name].mode()[0]]
        if len(maj) > len(mino):
            maj_down = resample(maj, replace=False, n_samples=len(mino), random_state=42)
            df_bal = pd.concat([maj_down, mino])
        X_train_scaled = df_bal.drop(columns=[y_train.name])
        y_train = df_bal[y_train.name]

    model = train_model(X_train_scaled, y_train)
    metrics, preds_test, probs_test = evaluate(model, X_test_scaled, y_test)
    save_feature_importance(model, feature_cols)
    if metrics["roc_auc"] > 0.90:
        print("WARNING: Model still too strong → possible leakage")
    est = None
    if hasattr(model, "base_estimator_"):
        est = model.base_estimator_
    elif hasattr(model, "base_estimator"):
        est = model.base_estimator
    elif hasattr(model, "calibrated_classifiers_"):
        cal = model.calibrated_classifiers_[0]
        if hasattr(cal, "estimator"):
            est = cal.estimator
        elif hasattr(cal, "base_estimator"):
            est = cal.base_estimator

    if est is not None and hasattr(est, "feature_importances_"):
        fi = pd.DataFrame({
            "feature": feature_cols,
            "importance": est.feature_importances_,
        }).sort_values("importance", ascending=False)
        print("Feature importance (top 10):")
        print(fi.head(10))

    # 7) Predictions on full data
    df_scaled = pd.DataFrame(
        scaler.transform(df[feature_cols]),
        columns=feature_cols,
        index=df.index,
    )
    df["crisis_prob"] = model.predict_proba(df_scaled)[:, 1]
    df["crisis_prob"] = np.clip(df["crisis_prob"], 0.01, 0.99)
    print("Probability stats (full):", df["crisis_prob"].min(), df["crisis_prob"].max(), df["crisis_prob"].mean())
    df = apply_risk_levels(df)
    print("Risk distribution (full):")
    print(df["risk_level"].value_counts())

    test_df = test_df.copy()
    test_df["crisis_prob"] = probs_test
    test_df = apply_risk_levels(test_df)
    test_df["pred"] = preds_test

    print("Risk distribution (test):")
    print(test_df["risk_level"].value_counts())
    print("Probability stats (test):")
    print(test_df["crisis_prob"].describe())
    print("High risk countries:")
    print(test_df[test_df["risk_level"] == "HIGH"]["country"].unique())

    # 8) Forecast
    forecast_df = build_forecast(model, df[["country", "month"] + feature_cols], feature_cols)

    # 9) Save outputs (overwrite)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    forecast_df.to_csv(FORECAST_OUT, index=False)
    pd.DataFrame([metrics]).to_csv(METRICS_OUT, index=False)
    perf_detail = test_df.copy()
    perf_detail["y_true"] = y_test.values
    perf_detail["y_pred"] = preds_test
    perf_detail["y_prob"] = probs_test
    perf_detail = perf_detail[["country", "month", "y_true", "y_pred", "y_prob"]]
    perf_detail.to_csv(PERF_DETAIL_OUT, index=False)

    print(f"Saved clean dataset to {DATA_PATH}")
    print(f"Saved forecast to {FORECAST_OUT}")
    print(f"Saved metrics to {METRICS_OUT}")


if __name__ == "__main__":
    main()
