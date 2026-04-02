import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"


def _check(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path


# ─────────────────────────────
@st.cache_data
def load_dataset():
    df = pd.read_csv(_check(DATA_DIR / "clean_master_dataset.csv"))
    df.columns = [c.strip().lower() for c in df.columns]
    return df


@st.cache_resource
def load_model():
    return joblib.load(_check(MODEL_DIR / "model.pkl"))


@st.cache_resource
def load_scaler():
    return joblib.load(_check(MODEL_DIR / "scaler.pkl"))


@st.cache_data
def load_feature_cols():
    with open(_check(MODEL_DIR / "feature_cols.json")) as f:
        return json.load(f)


@st.cache_data
def load_model_metrics():
    with open(_check(OUTPUT_DIR / "model_metrics.json")) as f:
        return json.load(f)


# ─────────────────────────────
def predict_risk(df_row):
    model = load_model()
    scaler = load_scaler()
    features = load_feature_cols()

    X = df_row[features]
    X_scaled = scaler.transform(X)

    proba = model.predict_proba(X_scaled)[:, 1][0]
    pred = int(proba >= 0.5)

    return float(proba), pred
