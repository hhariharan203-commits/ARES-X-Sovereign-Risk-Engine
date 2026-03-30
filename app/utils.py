from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import json
import numpy as np
import shap

import joblib
import pandas as pd


# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "clean_master_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
SHAP_PATH = BASE_DIR / "outputs" / "shap_importance.csv"
FEATURE_COLS_PATH = BASE_DIR / "models" / "feature_cols.json"
THRESHOLDS_PATH = BASE_DIR / "models" / "risk_thresholds.json"


# =========================
# HUMAN READABLE FEATURE MAP
# =========================
FEATURE_NAME_MAP = {
    "gdp_current_usd": "GDP level",
    "gdp_growth_pct": "GDP growth",
    "exports_pct_gdp": "Export share of GDP",
    "imports_pct_gdp": "Import share of GDP",
    "interest_rate_pct": "Interest rates",
    "inflation_cpi_pct": "Inflation",
    "unemployment_rate_pct": "Unemployment",

    "exports_pct_gdp_lag1": "Export momentum (short-term)",
    "exports_pct_gdp_lag3": "Export trend (medium-term)",
    "imports_pct_gdp_lag1": "Import pressure (short-term)",
    "imports_pct_gdp_lag3": "Import trend (medium-term)",
    "gdp_current_usd_lag3": "GDP trend",
}


# =========================
# LOADERS
# =========================
@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["month"])
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values(["country", "month"]).reset_index(drop=True)
    return df


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_shap() -> pd.DataFrame:
    return pd.read_csv(SHAP_PATH)


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
    model = load_model()
    return shap.TreeExplainer(model)


# =========================
# CORE FUNCTIONS
# =========================
def align_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = load_feature_cols()
    df_aligned = df.reindex(columns=feature_cols, fill_value=0)
    df_aligned = df_aligned[feature_cols]
    return df_aligned


def risk_label(prob: float) -> str:
    thresholds = load_thresholds()
    if prob < thresholds.get("low", 0.33):
        return "LOW"
    if prob < thresholds.get("high", 0.66):
        return "MEDIUM"
    return "HIGH"


# =========================
# UI THEME
# =========================
def apply_dark_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        hoverlabel=dict(bgcolor="black", font_size=14, font_color="white"),
        xaxis=dict(title_font=dict(color="white"), tickfont=dict(color="white"), gridcolor="gray"),
        yaxis=dict(title_font=dict(color="white"), tickfont=dict(color="white"), gridcolor="gray"),
    )

    # Safe trace updates
    for trace in fig.data:
        if trace.type != "choropleth":
            if hasattr(trace, "opacity"):
                trace.opacity = 0.9
            if hasattr(trace, "line"):
                trace.line.width = 3

    return fig


# =========================
# PREDICTION + SHAP
# =========================
def predict_with_explanations(df_row: pd.DataFrame):
    model = load_model()
    explainer = load_explainer()

    aligned = align_features(df_row).iloc[[0]]
    prob = float(model.predict_proba(aligned)[0, 1])

    shap_values = explainer.shap_values(aligned)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.array(shap_values)

    if len(shap_values.shape) > 1:
        shap_values = shap_values[0]

    shap_values = shap_values.flatten()

    feature_cols = list(load_feature_cols())
    top_idx = np.argsort(np.abs(shap_values))[::-1][:3]

    insights = []
    for i in top_idx:
        idx = int(i)
        if idx >= len(feature_cols) or idx >= len(shap_values):
            continue

        feat_raw = feature_cols[idx]
        feat = FEATURE_NAME_MAP.get(feat_raw, feat_raw.replace("_", " "))

        val = float(shap_values[idx])
        direction = "↑" if val > 0 else "↓"

        insights.append(f"{feat} {direction}")

    if not insights:
        insights.append("No strong risk drivers identified")

    return prob, insights


# =========================
# BUSINESS TRANSLATION
# =========================
def driver_to_business(feat: str, delta: float) -> str:
    label = FEATURE_NAME_MAP.get(feat, feat.replace("_", " "))
    direction = "raising" if delta > 0 else "reducing"
    return f"{label} is {direction} risk"


# =========================
# EXECUTIVE INSIGHTS
# =========================
def generate_executive_insights(aligned: pd.DataFrame, shap_values, expected_value: float) -> dict:
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_array = np.array(shap_values).flatten()
    feature_cols = list(aligned.columns)

    order = np.argsort(np.abs(shap_array))[::-1][:3]

    drivers = []
    actions = []

    for idx in order:
        idx = int(idx)
        if idx >= len(shap_array) or idx >= len(feature_cols):
            continue

        feat = feature_cols[idx]
        delta = float(shap_array[idx])

        drivers.append(f"- {driver_to_business(feat, delta)}")

    # Summary
    if drivers:
        summary = drivers[0].replace("-", "").strip().capitalize()
    else:
        summary = "Risk drivers stable."

    # Actions
    for d in drivers:
        lower = d.lower()

        if "interest rate" in lower:
            actions.append("Consider monetary policy adjustment")

        elif "export" in lower:
            actions.append("Stabilize export growth and trade balance")

        elif "inflation" in lower:
            actions.append("Control inflation through fiscal tightening")

        elif "gdp" in lower:
            actions.append("Stimulate economic growth via policy support")

    if not actions:
        actions.append("Maintain current macroeconomic stability")

    # Remove duplicates
    actions = list(set(actions))

    return {
        "summary": summary,
        "drivers": drivers,
        "actions": actions,
        "insights": drivers,
    }


