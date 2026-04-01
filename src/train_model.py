from pathlib import Path
import pandas as pd
import numpy as np
import joblib, json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# PATHS
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH    = BASE_DIR / "data" / "clean_master_dataset.csv"
MODEL_PATH   = BASE_DIR / "models" / "model.pkl"
SCALER_PATH  = BASE_DIR / "models" / "scaler.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_cols.json"

# FEATURES (must match cleaning)
FEATURE_COLS = [
    "gdp_growth_lag1","gdp_growth_lag3","gdp_growth_lag6",
    "inflation_lag1","inflation_lag3","inflation_lag6",
    "unemployment_lag1","unemployment_lag3","unemployment_lag6",
    "interest_rate_lag1","interest_rate_lag3","interest_rate_lag6",
    "gdp_growth_roll3","inflation_roll3","unemployment_roll3","interest_rate_roll3",
    "gdp_growth_roll6","inflation_roll6","unemployment_roll6","interest_rate_roll6",
    "gdp_growth_momentum","inflation_momentum","unemployment_momentum","interest_rate_momentum",
    "stress_index",
]

def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded: {df.shape}")

    # ensure numeric
    for col in df.columns:
        if col != "country":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["risk_label"])
    df = df.fillna(df.median(numeric_only=True))

    available = [f for f in FEATURE_COLS if f in df.columns]
    print(f"Using {len(available)} features")

    X = df[available]
    y = df["risk_label"]

    if y.nunique() < 2:
        raise ValueError("Only one class present!")

    # split
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    scale_pos = (y_train==0).sum() / (y_train==1).sum()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:,1]
    y_pred  = (y_proba >= 0.5).astype(int)

    print("\n=== PERFORMANCE ===")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))

    # save
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    with open(FEATURE_PATH, "w") as f:
        json.dump(available, f)

    print("\n✅ model.pkl saved")
    print("✅ scaler.pkl saved")
    print("✅ feature_cols.json saved")

if __name__ == "__main__":
    main()