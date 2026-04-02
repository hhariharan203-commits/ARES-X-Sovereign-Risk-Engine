import pandas as pd
import numpy as np
import joblib, json
import streamlit as st

DATA = "data/clean_master_dataset.csv"
MODEL = "models/model.pkl"
SCALER = "models/scaler.pkl"
FEATS = "models/feature_cols.json"
METRICS = "outputs/model_metrics.json"

@st.cache_data
def load_data():
    return pd.read_csv(DATA)

@st.cache_resource
def load_model():
    model = joblib.load(MODEL)
    scaler = joblib.load(SCALER)
    feats = json.load(open(FEATS))
    return model, scaler, feats

def predict(row):
    model, scaler, feats = load_model()
    X = row[feats].fillna(0)
    X = scaler.transform(X)
    return model.predict_proba(X)[0][1]

def latest(df, country):
    return df[df["country"]==country].sort_values(["year","month"]).tail(1)

def add_risk(df):
    df = df.copy()
    df["risk_score"] = df.apply(lambda x: predict(x.to_frame().T), axis=1)
    return df

def metrics():
    return json.load(open(METRICS))
