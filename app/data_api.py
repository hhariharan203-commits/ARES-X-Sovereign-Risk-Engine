"""
data_api.py — Load dataset, trained model, feature columns, and metrics.
"""

from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parents[1]

DATA_PATH    = BASE / "data"    / "master_dataset.csv"
MODEL_PATH   = BASE / "models"  / "model.pkl"
FEATURES_PATH = BASE / "models" / "feature_cols.json"
METRICS_PATH = BASE / "outputs" / "model_metrics.json"


@st.cache_data(ttl=0)
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["month"])
    df = df.sort_values(["country", "month"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_resource(ttl=0)
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_feature_cols() -> list:
    with open(FEATURES_PATH) as f:
        return json.load(f)


def get_countries(df: pd.DataFrame) -> list:
    return sorted(df["country"].unique().tolist())


def get_latest(df: pd.DataFrame) -> pd.DataFrame:
    """Return the latest observation row per country (correct selection)."""
    df = df.copy()

    # FIX: Proper sorting by country + month
    df = df.sort_values(["country", "month"])

    # FIX: Correctly pick latest row per country
    latest_df = df.groupby("country", as_index=False).tail(1)

    return latest_df.reset_index(drop=True)


def get_country_series(df: pd.DataFrame, country: str) -> pd.DataFrame:
    return df[df["country"] == country].sort_values("month").reset_index(drop=True)
