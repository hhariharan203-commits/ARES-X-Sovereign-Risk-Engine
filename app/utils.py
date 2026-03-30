from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import json
import numpy as np
import shap

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "clean_master_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
SHAP_PATH = BASE_DIR / "outputs" / "shap_importance.csv"
FEATURE_COLS_PATH = BASE_DIR / "models" / "feature_cols.json"
THRESHOLDS_PATH = BASE_DIR / "models" / "risk_thresholds.json"


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
    imp = pd.read_csv(SHAP_PATH)
    return imp


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


def align_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = load_feature_cols()
    df_aligned = df.reindex(columns=feature_cols, fill_value=0)
    df_aligned = df_aligned[feature_cols]
    assert df_aligned.shape[1] == len(feature_cols)
    return df_aligned


def risk_label(prob: float) -> str:
    thresholds = load_thresholds()
    low_cut = thresholds.get("low", 0.33)
    high_cut = thresholds.get("high", 0.66)
    if prob < low_cut:
        return "LOW"
    if prob < high_cut:
        return "MEDIUM"
    return "HIGH"


@lru_cache(maxsize=1)
def load_explainer():
    model = load_model()
    return shap.TreeExplainer(model)


def apply_dark_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        hoverlabel=dict(
            bgcolor="black",
            font_size=14,
            font_color="white",
        ),
        xaxis=dict(
            title_font=dict(color="white"),
            tickfont=dict(color="white"),
            gridcolor="gray",
        ),
        yaxis=dict(
            title_font=dict(color="white"),
            tickfont=dict(color="white"),
            gridcolor="gray",
        ),
    )

    # Safe trace update (choropleth lacks opacity attr)
    for trace in fig.data:
        if trace.type != "choropleth":
            if hasattr(trace, "opacity"):
                trace.opacity = 0.9
            if hasattr(trace, "line"):
                trace.line.width = 3

    return fig


def predict_with_explanations(df_row: pd.DataFrame):
    """
    Predict crisis probability for a single-row DataFrame and return SHAP insights.
    """
    model = load_model()
    explainer = load_explainer()

    aligned = align_features(df_row).iloc[[0]]  # enforce single row, correct schema
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
        feat = feature_cols[idx]
        val = float(shap_values[idx])
        direction = "↑" if val > 0 else "↓"
        insights.append(f"{feat}: {direction}")
    if not insights:
        insights.append("No strong risk drivers identified")

    return prob, insights


def top_shap_insights(aligned: pd.DataFrame, shap_values, expected_value: float, top_n: int = 3):
    # Deprecated in favor of direct logic in predict_with_explanations
    return []


def driver_to_business(feat: str, delta: float) -> str:
    mapping = {
        "gdp_current_usd": "economic strength",
        "gdp_growth_pct": "GDP growth momentum",
        "exports_pct_gdp": "export share of GDP",
        "imports_pct_gdp": "import dependency",
        "interest_rate_pct": "interest rate levels",
        "unemployment_rate_pct": "labor market stress",
        "exports_pct_gdp_lag1": "short-term export performance",
        "exports_pct_gdp_lag3": "medium-term export trends",
        "imports_pct_gdp_lag1": "short-term import dependency",
        "imports_pct_gdp_lag3": "medium-term import dependency",
    }

    # heuristic humanization for lags/rolls
    if feat.endswith("_lag1") and feat not in mapping:
        base = feat.replace("_lag1", "")
        mapping[feat] = f"short-term {base.replace('_', ' ')}"
    if feat.endswith("_lag3") and feat not in mapping:
        base = feat.replace("_lag3", "")
        mapping[feat] = f"medium-term {base.replace('_', ' ')}"
    if "_roll" in feat and feat not in mapping:
        base = feat.split("_roll")[0].replace("_", " ")
        mapping[feat] = f"smoothed {base}"
    if "_vol" in feat and feat not in mapping:
        base = feat.split("_vol")[0].replace("_", " ")
        mapping[feat] = f"volatility in {base}"

    label = mapping.get(feat, feat.replace("_", " "))
    direction = "raising" if delta > 0 else "reducing"
    return f"{label} is {direction} risk"


def generate_executive_insights(aligned: pd.DataFrame, shap_values, expected_value: float) -> dict:
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_array = np.array(shap_values).flatten()

    feature_cols = list(aligned.columns)
    order = np.argsort(np.abs(shap_array))[::-1][:3]

    key_drivers = []
    actions = []

    for idx in order:
        idx = int(idx)
        if idx >= len(shap_array) or idx >= len(feature_cols):
            continue
        feat = feature_cols[idx]
        delta = float(shap_array[idx])
        key_drivers.append(f"- {driver_to_business(feat, delta)}")

    # Build summary line
    summary = "Risk drivers stable."
    if key_drivers:
        phrases = []
        for idx in order[:2]:
            idx = int(idx)
            if idx >= len(shap_array) or idx >= len(feature_cols):
                continue
            feat = feature_cols[idx]
            delta = float(shap_array[idx])
            phrases.append(driver_to_business(feat, delta))
        if phrases:
            summary = ", while ".join([p.capitalize() for p in phrases])

    actions = []
    for insight in key_drivers:
        lower = insight.lower()
        if "interest_rate" in lower:
            actions.append("Consider monetary policy adjustment")
        elif "exports" in lower:
            actions.append("Stabilize export growth and trade balance")
        elif "inflation" in lower:
            actions.append("Control inflation through fiscal tightening")
        elif "gdp" in lower:
            actions.append("Stimulate economic growth via policy support")
    if not actions:
        actions.append("Maintain current macroeconomic stability")

    actions = list(set(actions))

    return {
        "summary": summary,
        "drivers": key_drivers,
        "actions": actions,
        "insights": key_drivers,
    }


