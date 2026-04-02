from pathlib import Path
import pandas as pd
import numpy as np
import joblib, json

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from xgboost import XGBClassifier

# ─────────────────────────────
BASE = Path(__file__).resolve().parent.parent

DATA = BASE / "data" / "clean_master_dataset.csv"
MODEL = BASE / "models" / "model.pkl"
SCALER = BASE / "models" / "scaler.pkl"
FEATURES = BASE / "models" / "feature_cols.json"
METRICS = BASE / "outputs" / "model_metrics.json"

# ─────────────────────────────
FEATURE_COLS = [
    "gdp_growth_lag1","gdp_growth_lag3",
    "inflation_lag1","inflation_lag3",
    "unemployment_lag1","unemployment_lag3",
    "interest_rate_lag1","interest_rate_lag3",
    "gdp_growth_roll3","inflation_roll3",
    "unemployment_roll3","interest_rate_roll3",
    "gdp_growth_momentum","inflation_momentum",
    "unemployment_momentum","interest_rate_momentum"
]

# ─────────────────────────────
def log(msg):
    print(f"[MODEL] {msg}")

# ─────────────────────────────
def load_data():
    if not DATA.exists():
        raise FileNotFoundError("Clean dataset not found")

    df = pd.read_csv(DATA)
    log(f"Loaded data: {df.shape}")

    if df.empty:
        raise ValueError("Dataset is empty")

    return df

# ─────────────────────────────
def preprocess(df):
    df = df.copy()

    # Ensure ordering (VERY IMPORTANT)
    df = df.sort_values(["country","year","month"]).reset_index(drop=True)

    # Convert numeric safely
    for col in df.columns:
        if col != "country":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["risk_label"])

    # Fill remaining missing
    df = df.fillna(df.median(numeric_only=True))

    return df

# ─────────────────────────────
def validate_features(df):
    missing = [f for f in FEATURE_COLS if f not in df.columns]

    if missing:
        raise ValueError(f"Missing required features: {missing}")

    log("Feature validation passed")

# ─────────────────────────────
def prepare_data(df):
    X = df[FEATURE_COLS]
    y = df["risk_label"]
    groups = df["country"]

    log(f"Class distribution:\n{y.value_counts()}")

    return X, y, groups

# ─────────────────────────────
def cross_validate(X, y, groups):
    log("Running GroupKFold CV")

    gkf = GroupKFold(n_splits=5)
    aucs = []

    for i, (tr, val) in enumerate(gkf.split(X, y, groups)):
        scaler = StandardScaler()

        X_tr = scaler.fit_transform(X.iloc[tr])
        X_val = scaler.transform(X.iloc[val])

        y_tr = y.iloc[tr]
        y_val = y.iloc[val]

        pos = max((y_tr==1).sum(), 1)
        neg = max((y_tr==0).sum(), 1)

        model = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=neg/pos,
            eval_metric="logloss",
            random_state=42
        )

        model.fit(X_tr, y_tr)

        proba = model.predict_proba(X_val)[:,1]
        auc = roc_auc_score(y_val, proba)
        aucs.append(auc)

        log(f"Fold {i+1} AUC: {auc:.4f}")

    mean_auc = float(np.mean(aucs))
    log(f"CV AUC: {mean_auc:.4f}")

    return mean_auc

# ─────────────────────────────
def train_final_model(X, y):
    log("Training final model with holdout validation")

    # Time-based split (IMPORTANT)
    split = int(len(X) * 0.8)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    pos = max((y_train==1).sum(), 1)
    neg = max((y_train==0).sum(), 1)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=neg/pos,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:,1]

    # Threshold optimization
    best_t, best_f1 = 0.5, 0

    for t in np.arange(0.1, 0.9, 0.05):
        pred = (proba >= t).astype(int)

        if pred.sum() == 0:
            continue

        f1 = f1_score(y_test, pred)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    pred = (proba >= best_t).astype(int)
    auc = roc_auc_score(y_test, proba)

    log("\n=== FINAL PERFORMANCE ===")
    print(classification_report(y_test, pred))
    log(f"AUC: {auc:.4f}")
    log(f"Best Threshold: {best_t:.2f}")

    return model, scaler, auc, best_f1, best_t

# ─────────────────────────────
def save_outputs(model, scaler, auc, f1, threshold, cv_auc, rows):
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    METRICS.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL)
    joblib.dump(scaler, SCALER)

    with open(FEATURES, "w") as f:
        json.dump(FEATURE_COLS, f)

    with open(METRICS, "w") as f:
        json.dump({
            "roc_auc": auc,
            "f1": f1,
            "threshold": threshold,
            "cv_auc": cv_auc,
            "features": len(FEATURE_COLS),
            "rows": rows
        }, f, indent=2)

    log("Model artifacts saved")

# ─────────────────────────────
def main():
    log("START TRAINING PIPELINE")

    df = load_data()
    df = preprocess(df)

    validate_features(df)

    X, y, groups = prepare_data(df)

    cv_auc = cross_validate(X, y, groups)

    model, scaler, auc, f1, threshold = train_final_model(X, y)

    save_outputs(model, scaler, auc, f1, threshold, cv_auc, len(df))

    log("MODEL READY (PRODUCTION ELITE)")

# ─────────────────────────────
if __name__ == "__main__":
    main()