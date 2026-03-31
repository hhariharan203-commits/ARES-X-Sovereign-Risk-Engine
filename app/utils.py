from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import json
import numpy as np
import shap
import joblib
import pandas as pd
import streamlit as st

# ================= PATHS =================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "clean_master_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
FEATURE_COLS_PATH = BASE_DIR / "models" / "feature_cols.json"


# ================= HELPERS =================
def standardize_columns(df):
    df.columns = df.columns.str.strip().str.lower()
    return df


def humanize_feature(feat: str) -> str:
    return feat.replace("_", " ").title()


# ================= LOADERS =================
@lru_cache(maxsize=1)
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = standardize_columns(df)
    return df


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_feature_cols():
    with open(FEATURE_COLS_PATH) as f:
        return json.load(f)


# ================= CORE =================
def align_features(df):
    cols = load_feature_cols()
    df = standardize_columns(df.copy())
    df = df.reindex(columns=cols)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.fillna(0)


def get_thresholds():
    # Stable thresholds (NOT dynamic per page)
    return {"low": 0.3, "high": 0.6}


def assign_risk(prob):
    t = get_thresholds()
    if prob < t["low"]:
        return "LOW"
    elif prob < t["high"]:
        return "MEDIUM"
    return "HIGH"


def add_probabilities(df):
    model = load_model()
    df = df.copy()

    X = align_features(df)

    try:
        probs = model.predict_proba(X)[:, 1]
    except Exception:
        probs = np.zeros(len(df))

    df["crisis_prob"] = probs
    df["risk_level"] = [assign_risk(p) for p in probs]

    return df


# ================= PREDICTION =================
def predict_with_explanations(row_df):
    model = load_model()

    X = align_features(row_df)

    try:
        prob = float(model.predict_proba(X)[0][1])
    except Exception:
        return 0.25, ["Prediction fallback"]

    insights = []

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_values = np.array(shap_values)[0]

        feature_cols = load_feature_cols()
        top_idx = np.argsort(np.abs(shap_values))[::-1][:3]

        for i in top_idx:
            feat = feature_cols[i]
            val = shap_values[i]
            direction = "↑" if val > 0 else "↓"
            insights.append(f"{humanize_feature(feat)} {direction}")

    except Exception:
        insights = ["Explainability unavailable"]

    return prob, insights


# ================= EXECUTIVE INSIGHTS =================
def generate_executive_insights(aligned, shap_values):
    try:
        shap_values = np.array(shap_values)
        if shap_values.ndim > 1:
            shap_values = shap_values[0]

        feature_cols = list(aligned.columns)
        order = np.argsort(np.abs(shap_values))[::-1]

        drivers = []
        for i in order[:3]:
            feat = feature_cols[i]
            drivers.append(humanize_feature(feat))

        return {
            "summary": "Key macroeconomic drivers identified",
            "drivers": drivers,
            "actions": ["Monitor economic indicators closely"]
        }

    except Exception:
        return {
            "summary": "Stable risk outlook",
            "drivers": ["No strong signals"],
            "actions": ["Maintain monitoring"]
        }


# ================= UI =================
def safe_metric(label, value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        st.metric(label, "N/A")
    else:
        st.metric(label, value)
