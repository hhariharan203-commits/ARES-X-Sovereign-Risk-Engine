"""
scenario_lab.py — Allow users to simulate macro shocks and re-run model predictions.
"""

import pandas as pd
import numpy as np
from data_api import load_dataset, load_model, load_feature_cols, get_country_series
from risk_engine import _risk_from_signals
from utils import regime_label, fmt_risk_label, clamp


def _build_scenario_row(base_row: pd.Series, overrides: dict, feature_cols: list) -> pd.DataFrame:
    """
    Construct a feature vector from a dataset row, applying user overrides
    to the lag-1 slots (most proximate influence on the forecast).
    """
    row = base_row.copy()

    # Apply overrides to lag-1 positions
    for col, val in overrides.items():
        lag1 = f"{col}_s1"
        if lag1 in row.index:
            row[lag1] = val
        # Also update lag-2 slightly toward override (tapering)
        lag2 = f"{col}_s2"
        if lag2 in row.index:
            row[lag2] = (row[lag2] + val) / 2

    # Recompute derived features that may depend on overrides
    if "inflation_s1" in row.index and "inflation_s3" in row.index:
        row["inflation_trend"] = row["inflation_s1"] - row["inflation_s3"]
    if "exports_s1" in row.index and "imports_s1" in row.index:
        row["trade_balance"] = row["exports_s1"] - row["imports_s1"]
    if "vix_s1" in row.index and "vix_s2" in row.index:
        row["vix_change"] = row["vix_s1"] - row["vix_s2"]
    if "inflation_s1" in row.index and "interest_rate_s1" in row.index:
        row["inflation_x_rate"] = row["inflation_s1"] * row["interest_rate_s1"]
    if "vix_s1" in row.index and "inflation_s1" in row.index:
        row["vix_x_inflation"] = row["vix_s1"] * row["inflation_s1"]

    available = {c: row[c] for c in feature_cols if c in row.index}
    missing   = {c: 0.0    for c in feature_cols if c not in row.index}
    return pd.DataFrame([{**available, **missing}])[feature_cols]


def run_scenario(country: str, overrides: dict) -> dict:
    """
    Run a macro scenario simulation for a country.

    overrides: dict mapping base signal names to new values
    e.g. {"inflation": 7.0, "interest_rate": 5.5}

    Returns baseline and scenario forecasts + derived risk metrics.
    """
    df           = load_dataset()
    model        = load_model()
    feature_cols = load_feature_cols()

    series = get_country_series(df, country)
    if series.empty:
        return {"error": f"No data for {country}"}

    latest = series.iloc[-1]

    # Baseline prediction
    base_feats   = {c: latest.get(c, 0.0) for c in feature_cols if c in latest.index}
    missing_base = {c: 0.0 for c in feature_cols if c not in latest.index}
    X_base       = pd.DataFrame([{**base_feats, **missing_base}])[feature_cols]
    baseline_gdp = float(model.predict(X_base)[0])

    # Scenario prediction
    X_scenario   = _build_scenario_row(latest, overrides, feature_cols)
    scenario_gdp = float(model.predict(X_scenario)[0])

    # Risk in baseline
    base_inflation    = float(latest.get("inflation",    3.0))
    base_unemp        = float(latest.get("unemployment", 5.0))
    base_vix          = float(latest.get("vix",         20.0))
    base_trade        = float(latest.get("exports", 0.0)) - float(latest.get("imports", 0.0))
    base_risk         = _risk_from_signals(baseline_gdp, base_inflation, base_unemp, base_vix, base_trade)

    # Risk in scenario (apply overrides to risk inputs too)
    sc_inflation  = overrides.get("inflation",    base_inflation)
    sc_unemp      = overrides.get("unemployment", base_unemp)
    sc_vix        = overrides.get("vix",          base_vix)
    sc_exports    = overrides.get("exports",       float(latest.get("exports", 0.0)))
    sc_imports    = overrides.get("imports",       float(latest.get("imports", 0.0)))
    sc_trade      = sc_exports - sc_imports
    sc_risk       = _risk_from_signals(scenario_gdp, sc_inflation, sc_unemp, sc_vix, sc_trade)

    delta_gdp  = round(scenario_gdp  - baseline_gdp, 3)
    delta_risk = round(sc_risk       - base_risk,    1)

    sc_regime  = regime_label(scenario_gdp, sc_inflation)

    return {
        "country":        country,
        "overrides":      overrides,
        "baseline_gdp":   round(baseline_gdp, 3),
        "scenario_gdp":   round(scenario_gdp, 3),
        "delta_gdp":      delta_gdp,
        "baseline_risk":  round(base_risk, 1),
        "scenario_risk":  round(sc_risk, 1),
        "delta_risk":     delta_risk,
        "scenario_regime": sc_regime,
        "risk_label":     fmt_risk_label(sc_risk),
    }