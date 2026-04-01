from pathlib import Path
import pandas as pd
import numpy as np
import joblib, json

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from xgboost import XGBClassifier

BASE = Path(__file__).resolve().parent.parent

DATA = BASE / "data" / "clean_master_dataset.csv"
MODEL = BASE / "models" / "model.pkl"
SCALER = BASE / "models" / "scaler.pkl"
FEATURES = BASE / "models" / "feature_cols.json"
METRICS = BASE / "outputs" / "model_metrics.json"

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

def main():
    df = pd.read_csv(DATA)
    print("Loaded:", df.shape)

    for col in df.columns:
        if col != "country":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["risk_label"])
    df = df.fillna(df.median(numeric_only=True))

    feats = [f for f in FEATURE_COLS if f in df.columns]

    X = df[feats]
    y = df["risk_label"]
    groups = df["country"]

    print("Risk dist:\n", y.value_counts())

    # ── CV ─────────────────
    gkf = GroupKFold(5)
    aucs = []

    for i,(tr,val) in enumerate(gkf.split(X,y,groups)):
        sc = StandardScaler()

        X_tr = sc.fit_transform(X.iloc[tr])
        X_val = sc.transform(X.iloc[val])

        y_tr = y.iloc[tr]

        pos = max((y_tr==1).sum(), 1)
        neg = max((y_tr==0).sum(), 1)

        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=neg/pos,
            eval_metric="logloss",
            random_state=42
        )

        model.fit(X_tr,y_tr)

        auc = roc_auc_score(y.iloc[val], model.predict_proba(X_val)[:,1])
        aucs.append(auc)

        print(f"Fold {i+1}: {auc:.4f}")

    print("CV AUC:", np.mean(aucs))

    # ── FINAL TRAIN ─────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pos = max((y==1).sum(), 1)
    neg = max((y==0).sum(), 1)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg/pos,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_scaled,y)

    proba = model.predict_proba(X_scaled)[:,1]

    best_t,best_f1 = 0.5,0
    for t in np.arange(0.1,0.8,0.05):
        pred = (proba>=t).astype(int)
        if pred.sum()==0:
            continue
        f1 = f1_score(y,pred)
        if f1>best_f1:
            best_f1,best_t = f1,t

    pred = (proba>=best_t).astype(int)
    auc = roc_auc_score(y,proba)

    print("\n=== FINAL REAL PERFORMANCE ===")
    print(classification_report(y,pred))
    print("AUC:",auc)
    print("Best threshold:",best_t)

    MODEL.parent.mkdir(parents=True,exist_ok=True)
    METRICS.parent.mkdir(parents=True,exist_ok=True)

    joblib.dump(model,MODEL)
    joblib.dump(scaler,SCALER)

    with open(FEATURES,"w") as f:
        json.dump(feats,f)

    with open(METRICS,"w") as f:
        json.dump({
            "roc_auc":float(auc),
            "best_threshold":float(best_t),
            "f1":float(best_f1),
            "cv_auc":float(np.mean(aucs)),
            "n_features":len(feats),
            "rows":len(df)
        },f,indent=2)

    print("\n✅ MODEL READY (FINAL SAFE)")

if __name__=="__main__":
    main()