"""
Sovereign Risk Decision Intelligence Engine
Main Application Entry Point
BlackRock / Goldman Sachs — Institutional Grade
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ─── Path resolution ─────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(BASE_DIR)
DATA_PATH  = os.path.join(ROOT_DIR, "data",   "clean_master_dataset.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "model.pkl")
SCALER_PATH= os.path.join(ROOT_DIR, "models", "scaler.pkl")
FEATS_PATH = os.path.join(ROOT_DIR, "models", "feature_cols.json")

sys.path.insert(0, BASE_DIR)

from intelligence import enrich_dataframe
from ui import (
    inject_css,
    render_sidebar,
    render_overview,
    render_global_risk,
    render_country_intelligence,
    render_scenario_lab,
)

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sovereign Risk Intelligence Engine",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Resource Loaders (cached) ────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_feature_cols() -> list:
    with open(FEATS_PATH, "r") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def load_enriched_data(_model, _scaler, feature_cols: list) -> pd.DataFrame:
    """Load and enrich dataset once — cached for session lifetime."""
    df = load_raw_data()

    # Normalize country column name
    col_map = {c: c.lower().strip() for c in df.columns}
    df.rename(columns=col_map, inplace=True)

    # Ensure a country column exists
    for candidate in ["country", "nation", "name", "sovereign", "iso_name", "economy"]:
        if candidate in df.columns:
            df.rename(columns={candidate: "country"}, inplace=True)
            break
    else:
        # Fallback: use first string column
        str_cols = df.select_dtypes(include="object").columns
        if len(str_cols) > 0:
            df.rename(columns={str_cols[0]: "country"}, inplace=True)
        else:
            df["country"] = [f"Sovereign_{i}" for i in range(len(df))]

    df = enrich_dataframe(df, _model, _scaler, feature_cols)
    return df


# ─── Session State Init ───────────────────────────────────────────────────────

def init_session():
    defaults = {
        "nav":            "Overview",
        "last_scenario":  None,
        "country_select": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    inject_css()
    init_session()

    # Load all assets
    try:
        model       = load_model()
        scaler      = load_scaler()
        feature_cols = load_feature_cols()
        df          = load_enriched_data(model, scaler, feature_cols)
    except FileNotFoundError as e:
        st.error(f"**Asset not found:** {e}")
        st.markdown("""
        Ensure the following files exist:
        - `data/clean_master_dataset.csv`
        - `models/model.pkl`
        - `models/scaler.pkl`
        - `models/feature_cols.json`
        """)
        st.stop()
    except Exception as e:
        st.error(f"**Initialization error:** {e}")
        st.stop()

    # Sidebar + navigation
    nav = render_sidebar(df)
    st.session_state["nav"] = nav

    # Route
    if nav == "Overview":
        render_overview(df)

    elif nav == "Global Risk":
        render_global_risk(df)

    elif nav == "Country Intelligence":
        render_country_intelligence(df, model, scaler, feature_cols)

    elif nav == "Scenario Lab":
        render_scenario_lab(df, model, scaler, feature_cols)


if __name__ == "__main__":
    main()
