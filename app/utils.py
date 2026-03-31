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


# ================= HELPERS =================
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower()
    return df


def humanize_feature(feat: str) -> str:
    return feat.replace("_", " ").title()


# ================= LOADERS =================
@lru_cache(maxsize=1)
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = standardize_columns(df)

    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")

    return df


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_feature_cols():
    with open(FEATURE_COLS_PATH) as f:
        cols = json.load(f)
    return [c.strip().lower() for c in cols]


@lru_cache(maxsize=1)
def load_explainer():
    try:
        return shap.TreeExplainer(load_model())
    except Exception:
        return None


# ================= CORE =================
def align_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = load_feature_cols()
    df = standardize_columns(df.copy())

    df_aligned = df.reindex(columns=cols)
    df_aligned = df_aligned.apply(pd.to_numeric, errors="coerce")

    return df_aligned


def fill_missing_values(row_df: pd.DataFrame) -> pd.DataFrame:
    """
    Smart imputation:
    1. Country last known
    2. Country mean
    3. Global mean
    4. Zero fallback
    """
    full_df = load_data()

    country = row_df["country"].iloc[0] if "country" in row_df.columns else None

    aligned_row = align_features(row_df)
    aligned_full = align_features(full_df)

    if country:
        country_hist = full_df[full_df["country"] == country]
        aligned_country = align_features(country_hist)

        if not aligned_country.empty:
            aligned_row = aligned_row.fillna(aligned_country.ffill().tail(1))
            aligned_row = aligned_row.fillna(aligned_country.mean())

    aligned_row = aligned_row.fillna(aligned_full.mean())
    aligned_row = aligned_row.fillna(0)

    return aligned_row


def get_thresholds():
    # More realistic spread
    return {"low": 0.25, "high": 0.65}


def assign_risk(prob: float) -> str:
    t = get_thresholds()

    if prob < t["low"]:
        return "LOW"
    elif prob < t["high"]:
        return "MEDIUM"
    return "HIGH"


def add_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    model = load_model()

    X = align_features(df).fillna(0)

    if hasattr(model, "feature_names_in_"):
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)

    try:
        probs = model.predict_proba(X.astype(float))[:, 1]
    except Exception:
        probs = np.zeros(len(df))

    # Clip for realism (avoid 0 or 1)
    probs = np.clip(probs, 0.01, 0.99)

    df["crisis_prob"] = probs
    df["risk_level"] = [assign_risk(p) for p in probs]

    return df


# ================= PREDICTION =================
def predict_with_explanations(row_df: pd.DataFrame):
    model = load_model()

    X = fill_missing_values(row_df)

    if hasattr(model, "feature_names_in_"):
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)

    X = X.astype(float)

    try:
        prob = float(model.predict_proba(X)[0][1])
        prob = np.clip(prob, 0.01, 0.99)
    except Exception:
        return 0.25, ["Prediction fallback"]

    insights = []

    explainer = load_explainer()

    if explainer is not None:
        try:
            shap_values = explainer.shap_values(X)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            shap_values = np.array(shap_values).flatten()

            feature_cols = list(X.columns)
            top_idx = np.argsort(np.abs(shap_values))[::-1][:3]

            for i in top_idx:
                if i >= len(feature_cols):
                    continue

                feat = feature_cols[i]
                val = shap_values[i]
                direction = "↑" if val > 0 else "↓"

                insights.append(f"{humanize_feature(feat)} {direction}")

        except Exception:
            insights = ["Explainability unavailable"]
    else:
        insights = ["Explainability unavailable"]

    if not insights:
        insights = ["No strong drivers"]

    return prob, insights


# ================= EXECUTIVE INSIGHTS =================
def generate_executive_insights(aligned, shap_values):

    try:
        shap_values = np.array(shap_values).flatten()
        feature_cols = list(aligned.columns)

        order = np.argsort(np.abs(shap_values))[::-1]

        drivers = []
        for i in order[:3]:
            if i >= len(feature_cols):
                continue
            drivers.append(humanize_feature(feature_cols[i]))

        return {
            "summary": "Macroeconomic conditions influencing sovereign risk",
            "drivers": drivers if drivers else ["No strong drivers"],
            "actions": [
                "Monitor inflation and interest rate dynamics",
                "Assess external trade exposure",
                "Evaluate fiscal stability"
            ],
        }

    except Exception:
        return {
            "summary": "Stable risk outlook",
            "drivers": ["No strong signals"],
            "actions": ["Maintain monitoring"],
        }
# ================= EXTRA FUNCTIONS =================

def load_shap():
    try:
        path = BASE_DIR / "data" / "shap_importance.csv"
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        return df
    except Exception:
        return pd.DataFrame()


def apply_dark_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
    )
    return fig


def risk_label(prob):
    return assign_risk(prob)
