import pandas as pd
import numpy as np
import joblib, json, requests
from pathlib import Path
import streamlit as st
from hmmlearn.hmm import GaussianHMM

BASE = Path(__file__).resolve().parent.parent

DATA = BASE / "data" / "clean_master_dataset.csv"
MODEL = BASE / "models" / "model.pkl"
SCALER = BASE / "models" / "scaler.pkl"
FEATURES = BASE / "models" / "feature_cols.json"
METRICS = BASE / "outputs" / "model_metrics.json"

@st.cache_data
def load_data():
    return pd.read_csv(DATA)

@st.cache_resource
def load_all():
    model = joblib.load(MODEL)
    scaler = joblib.load(SCALER)
    feats = json.load(open(FEATURES))
    metrics = json.load(open(METRICS))
    return model, scaler, feats, metrics

def latest(df, country):
    return df[df["country"] == country].sort_values(["year","month"]).iloc[-1:]

def predict(df):
    model, scaler, feats, _ = load_all()
    X = df[feats].fillna(0)
    X = scaler.transform(X)
    prob = model.predict_proba(X)[0][1]
    return prob

def predict_full(df):
    p = predict(df)
    if p > 0.75: tier = "HIGH RISK"
    elif p > 0.5: tier = "MODERATE RISK"
    else: tier = "LOW RISK"
    return p, tier

def global_risk(df):
    model, scaler, feats, _ = load_all()
    latest_df = df.sort_values(["year","month"]).groupby("country").tail(1)
    X = latest_df[feats].fillna(0)
    X = scaler.transform(X)
    latest_df["risk_score"] = model.predict_proba(X)[:,1]
    return latest_df.sort_values("risk_score", ascending=False)

def monte_carlo(row, n=200):
    sims = []
    for _ in range(n):
        r = row.copy()
        r["inflation"] += np.random.normal(0,1)
        r["gdp_growth"] += np.random.normal(0,1)
        sims.append(predict(r))
    return np.array(sims)

def detect_regime(df, country):
    d = df[df["country"]==country].sort_values(["year","month"])
    X = d[["gdp_growth","inflation","unemployment"]].fillna(0)
    model = GaussianHMM(n_components=3, n_iter=100)
    model.fit(X)
    d["regime"] = model.predict(X)
    return d

def metrics():
    _,_,_,m = load_all()
    return m
