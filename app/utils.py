from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import json
import numpy as np
import shap
import joblib
import pandas as pd

# ================= PATHS =================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "clean_master_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
FEATURE_COLS_PATH = BASE_DIR / "models" / "feature_cols.json"
THRESHOLDS_PATH = BASE_DIR / "models" / "risk_thresholds.json"

# ================= FEATURE NAME CLEANING =================
def humanize_feature(feat: str) -> str:
    feat = feat.replace("_pct", "%")
    feat = feat.replace("_usd", " USD")
    feat = feat.replace("_lag1", " (Short-term)")
    feat = feat.replace("_lag3", " (Medium-term)")
    feat = feat.replace("_lag6", " (Long-term)")
    feat = feat.replace("_z", " (Normalized)")
    feat = feat.replace("_", " ").title()
    return feat

# ================= LOADERS =================
@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["month"])
    return df.sort_values(["country", "month"]).reset_index(drop=True)

@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)

@lru_cache(maxsize=1)
def load_feature_cols() -> list[str]:
    with open(FEATURE_COLS_PATH) as f:
        return json.load(f)

@lru_cache(maxsize=1)
def load_thresholds() -> dict:
    if THRESHOLDS_PATH.exists():
        with open(THRESHOLDS_PATH) as f:
            return json.load(f)
    return {"low": 0.33, "high": 0.66}

@lru_cache(maxsize=1)
def load_explainer():
    return shap.TreeExplainer(load_model())

# ================= CORE =================
def align_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = load_feature_cols()
    df_aligned = df.reindex(columns=cols)
    return df_aligned[cols]  # ❌ no fillna here

def risk_label(prob: float) -> str:
    t = load_thresholds()
    if prob < t["low"]:
        return "LOW"
    elif prob < t["high"]:
        return "MEDIUM"
    return "HIGH"

# ================= PREDICTION =================
def predict_with_explanations(df_row: pd.DataFrame):
    model = load_model()
    explainer = load_explainer()

    aligned = align_features(df_row).iloc[[0]]

    # ✅ SMART FILL (avoid identical predictions)
    aligned = aligned.fillna(aligned.mean())

    # ✅ FINAL SAFETY (only if everything missing)
    if aligned.isnull().all(axis=None):
        return 0.25, ["Insufficient data for reliable prediction"]

    prob = float(model.predict_proba(aligned)[0, 1])

    shap_values = explainer.shap_values(aligned)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.array(shap_values).flatten()
    feature_cols = load_feature_cols()

    top_idx = np.argsort(np.abs(shap_values))[::-1][:3]

    insights = []
    for i in top_idx:
        feat = feature_cols[i]
        val = shap_values[i]
        direction = "↑" if val > 0 else "↓"
        insights.append(f"{humanize_feature(feat)} {direction}")

    return prob, insights

# ================= EXECUTIVE INSIGHTS =================
def generate_executive_insights(aligned: pd.DataFrame, shap_values) -> dict:
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.array(shap_values).flatten()
    feature_cols = list(aligned.columns)

    order = np.argsort(np.abs(shap_values))[::-1]

    group_map = {
        "gdp": "Economic Growth",
        "interest": "Interest Rates",
        "import": "Import Dependency",
        "export": "Export Strength",
        "unemployment": "Labor Market",
    }

    seen_groups = set()
    drivers = []
    actions = []

    for idx in order:
        feat = feature_cols[idx]
        delta = shap_values[idx]

        group = None
        for k in group_map:
            if k in feat:
                group = group_map[k]
                break

        if group and group not in seen_groups:
            direction = "increasing" if delta > 0 else "reducing"
            drivers.append(f"- {group} is {direction} financial risk")
            seen_groups.add(group)

            if group == "Interest Rates":
                actions.append("Review central bank rate policy to control inflation")
            elif group == "Export Strength":
                actions.append("Enhance export competitiveness through trade incentives")
            elif group == "Import Dependency":
                actions.append("Reduce import dependency via domestic production")
            elif group == "Economic Growth":
                actions.append("Deploy fiscal stimulus to support economic growth")
            elif group == "Labor Market":
                actions.append("Strengthen labor market through job creation programs")

        if len(drivers) == 3:
            break

    summary = drivers[0].replace("- ", "").capitalize() if drivers else "Overall risk remains stable"

    return {
        "summary": summary,
        "drivers": drivers,
        "actions": list(set(actions)),
    }

# ================= UI HELPERS =================
def apply_dark_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
    )
    return fig

# ================= SHAP LOADER =================
@lru_cache(maxsize=1)
def load_shap():
    shap_path = BASE_DIR / "outputs" / "shap_importance.csv"

    if shap_path.exists():
        return pd.read_csv(shap_path)

    return pd.DataFrame()
