"""
utils.py — Production-grade data + prediction engine
All model inference flows through this module.
"""

import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ─────────────────────────────────────────────────────────
# PATHS (FIXED FOR STREAMLIT CLOUD)
# ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR.parent

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"


def _check(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


# ─────────────────────────────────────────────────────────
# LOADERS (CACHED)
# ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    path = DATA_DIR / "clean_master_dataset.csv"
    df = pd.read_csv(_check(path))
    df.columns = [c.strip().lower() for c in df.columns]
    return df


@st.cache_resource(show_spinner=False)
def load_model():
    path = MODEL_DIR / "model.pkl"
    with open(_check(path), "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_scaler():
    path = MODEL_DIR / "scaler.pkl"
    with open(_check(path), "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_feature_cols() -> list:
    path = MODEL_DIR / "feature_cols.json"
    with open(_check(path)) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_model_metrics() -> dict:
    path = OUTPUT_DIR / "model_metrics.json"
    with open(_check(path)) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────
# CORE FEATURE ENGINE
# ─────────────────────────────────────────────────────────
def build_feature_matrix(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing features: {missing}")
    return df[feature_cols].copy()


def scale_features(X: pd.DataFrame, scaler) -> np.ndarray:
    try:
        return scaler.transform(X)
    except Exception as e:
        raise RuntimeError(f"Scaling failed: {e}")


# ─────────────────────────────────────────────────────────
# CORE PREDICTION PIPELINE
# ─────────────────────────────────────────────────────────
def predict_risk(df_row: pd.DataFrame) -> tuple[float, int]:
    model = load_model()
    scaler = load_scaler()
    features = load_feature_cols()

    X = build_feature_matrix(df_row, features)
    X_scaled = scale_features(X, scaler)

    proba = model.predict_proba(X_scaled)[:, 1][0]
    pred = int(proba >= 0.5)

    return float(proba), pred


# ─────────────────────────────────────────────────────────
# COUNTRY UTILITIES
# ─────────────────────────────────────────────────────────
def get_country_list(df: pd.DataFrame) -> list:
    return sorted(df["country"].dropna().unique())


def filter_country(df: pd.DataFrame, country: str) -> pd.DataFrame:
    return df[df["country"] == country].sort_values(["year", "month"])


def latest_row(df: pd.DataFrame, country: str) -> pd.DataFrame:
    d = filter_country(df, country)
    return d.tail(1)


def latest_all(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["year", "month"]).groupby("country").tail(1)


# ─────────────────────────────────────────────────────────
# FORECAST HELPERS
# ─────────────────────────────────────────────────────────
def rolling_trend(series: pd.Series, window: int = 3) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return 0.0
    y = s.iloc[-window:]
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])


def lag_project(series: pd.Series, steps: int = 3) -> list:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return [float(s.iloc[-1])] * steps if len(s) else [0.0] * steps

    delta = s.diff().dropna().iloc[-3:].mean()
    last = float(s.iloc[-1])

    return [last + delta * (i + 1) for i in range(steps)]


# ─────────────────────────────────────────────────────────
# GLOBAL RISK ENGINE
# ─────────────────────────────────────────────────────────
def compute_global_risk(df: pd.DataFrame) -> pd.DataFrame:
    latest = latest_all(df)

    scores = []
    for _, row in latest.iterrows():
        p, _ = predict_risk(row.to_frame().T)
        scores.append(p)

    latest["risk_score"] = scores
    return latest.sort_values("risk_score", ascending=False)


# ─────────────────────────────────────────────────────────
# PORTFOLIO ENGINE
# ─────────────────────────────────────────────────────────
def portfolio_risk(df: pd.DataFrame, weights: dict) -> dict:
    latest = latest_all(df)

    total_risk = 0
    for _, row in latest.iterrows():
        country = row["country"]
        w = weights.get(country, 0)

        p, _ = predict_risk(row.to_frame().T)
        total_risk += p * w

    return {
        "portfolio_risk": round(total_risk, 3),
        "status": (
            "High Risk" if total_risk > 0.6 else
            "Moderate Risk" if total_risk > 0.3 else
            "Low Risk"
        )
    }
