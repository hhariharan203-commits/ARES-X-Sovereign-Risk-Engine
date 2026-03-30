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
SHAP_PATH = BASE_DIR / "outputs" / "shap_importance.csv"
FEATURE_COLS_PATH = BASE_DIR / "models" / "feature_cols.json"
THRESHOLDS_PATH = BASE_DIR / "models" / "risk_thresholds.json"

# ================= FEATURE NAME MAPPING =================
FEATURE_NAME_MAP = {
    "gdp_current_usd": "GDP (Current USD)",
    "gdp_growth_pct": "GDP Growth",
    "exports_pct_gdp": "Exports (% of GDP)",
    "imports_pct_gdp": "Imports (% of GDP)",
    "interest_rate_pct": "Interest Rates",
    "interest_rate_pct_z": "Interest Rates (normalized)",
    "unemployment_rate_pct": "Unemployment Rate",
    "unemployment_rate_pct_lag1": "Short-term unemployment trend",
    "exports_pct_gdp_lag1": "Short-term export trend",
    "exports_pct_gdp_lag3": "Medium-term export trend",
    "imports_pct_gdp_lag1": "Short-term import dependency",
    "imports_pct_gdp_lag3": "Medium-term import dependency",
}

def humanize_feature(feat: str) -> str:
    return FEATURE_NAME_MAP.get(feat, feat.replace("_", " ").title())

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

# ✅ NEW (FIX)
@lru_cache(maxsize=1)
def load_shap():
    if SHAP_PATH.exists():
        return pd.read_csv(SHAP_PATH)
    return pd.DataFrame()

# ================= CORE =================
def align_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = load_feature_cols()
    df_aligned = df.reindex(columns=cols, fill_value=0)
    return df_aligned[cols]

def risk_label(prob: float) -> str:
    t = load_thresholds()
    if prob < t["low"]:
        return "LOW"
    elif prob < t["high"]:
        return "MEDIUM"
    return "HIGH"

# ================= UI THEME (FIX) =================
def apply_dark_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
    )
    return fig

# ================= PREDICTION =================
def predict_with_explanations(df_row: pd.DataFrame):
    model = load_model()
    explainer = load_explainer()

    aligned = align_features(df_row).iloc[[0]]
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

# ================= BUSINESS INSIGHTS =================
def driver_to_business(feat: str, delta: float) -> str:
    label = humanize_feature(feat)
    return f"{label} is {'increasing' if delta > 0 else 'reducing'} financial risk"

def generate_executive_insights(aligned: pd.DataFrame, shap_values) -> dict:
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.array(shap_values).flatten()
    feature_cols = list(aligned.columns)

    order = np.argsort(np.abs(shap_values))[::-1][:3]

    drivers = []
    actions = []

    for idx in order:
        feat = feature_cols[idx]
        delta = shap_values[idx]

        text = driver_to_business(feat, delta)
        drivers.append(f"- {text}")

        if "interest" in feat:
            actions.append("Adjust monetary policy")
        elif "export" in feat:
            actions.append("Boost export performance")
        elif "import" in feat:
            actions.append("Reduce import dependency")
        elif "gdp" in feat:
            actions.append("Stimulate economic growth")

    actions = list(set(actions))
    summary = drivers[0] if drivers else "Risk stable"

    return {
        "summary": summary,
        "drivers": drivers,
        "actions": actions,
    }
