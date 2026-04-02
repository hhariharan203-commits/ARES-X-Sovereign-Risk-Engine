"""
intelligence.py — Elite Decision Intelligence Engine
Transforms model outputs into executive-grade decisions.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from app.utils import scale_features

# ─────────────────────────────────────────────
# RISK TAXONOMY
# ─────────────────────────────────────────────
RISK_LEVELS = {
    "Low": (0.0, 0.35),
    "Moderate": (0.35, 0.55),
    "Elevated": (0.55, 0.72),
    "High": (0.72, 0.88),
    "Critical": (0.88, 1.01),
}

DECISIONS = {
    "Low":      ("Accumulate / Overweight", "green"),
    "Moderate": ("Neutral — Monitor", "gold"),
    "Elevated": ("Reduce Exposure", "orange"),
    "High":     ("De-risk Aggressively", "red"),
    "Critical": ("Exit / Full Hedge", "darkred"),
}

# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────
@dataclass
class RiskIntelligence:
    country: str
    year: Optional[int]
    risk_score: float
    risk_level: str
    confidence: float
    decision: str
    decision_color: str
    reasoning: str
    action: str
    regime: str


# ─────────────────────────────────────────────
# CORE PREDICTION
# ─────────────────────────────────────────────
def _predict(X_scaled, model):
    proba = model.predict_proba(X_scaled)[:, 1][0]
    confidence = max(proba, 1 - proba)
    return float(proba), float(confidence)


def _classify(score):
    for k, (lo, hi) in RISK_LEVELS.items():
        if lo <= score < hi:
            return k
    return "Critical"


# ─────────────────────────────────────────────
# REGIME DETECTION (STABLE VERSION)
# ─────────────────────────────────────────────
def detect_regime(row: dict) -> str:
    gdp = float(row.get("gdp_growth", 2))
    inf = float(row.get("inflation", 3))

    if gdp < 0 and inf > 6:
        return "Crisis"
    elif gdp < 1 and inf > 5:
        return "Stagflation"
    elif inf > 5:
        return "Inflation Stress"
    elif gdp > 2:
        return "Growth"
    return "Neutral"


# ─────────────────────────────────────────────
# BUSINESS REASONING
# ─────────────────────────────────────────────
def build_reasoning(level, regime, row):
    return (
        f"The economy is classified as {level} risk under a {regime} regime. "
        f"GDP growth is {round(row.get('gdp_growth',0),2)}% "
        f"and inflation is {round(row.get('inflation',0),2)}%. "
        f"This combination indicates macro instability affecting capital flows."
    )


def build_action(level):
    actions = {
        "Low": "Increase allocation to local assets.",
        "Moderate": "Maintain exposure with hedging.",
        "Elevated": "Reduce risk positions gradually.",
        "High": "Exit vulnerable positions and hedge.",
        "Critical": "Immediate capital protection required.",
    }
    return actions[level]


# ─────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────
def compute_risk_intelligence(row, model, scaler, feature_cols, country, year=None):

    # Safe feature extraction
    data = {}
    for f in feature_cols:
        val = row.get(f, 0)
        try:
            data[f] = float(val)
        except:
            data[f] = 0.0

    X = pd.DataFrame([data])
    X_scaled = scale_features(X, scaler)

    score, confidence = _predict(X_scaled, model)
    level = _classify(score)
    decision, color = DECISIONS[level]

    regime = detect_regime(data)
    reasoning = build_reasoning(level, regime, data)
    action = build_action(level)

    return RiskIntelligence(
        country=country,
        year=year,
        risk_score=score,
        risk_level=level,
        confidence=confidence,
        decision=decision,
        decision_color=color,
        reasoning=reasoning,
        action=action,
        regime=regime
    )


# ─────────────────────────────────────────────
# GLOBAL RISK TABLE
# ─────────────────────────────────────────────
def compute_global(df, model, scaler, feature_cols):
    latest = df.sort_values(["year", "month"]).groupby("country").tail(1)

    rows = []
    for _, r in latest.iterrows():
        intel = compute_risk_intelligence(
            r, model, scaler, feature_cols,
            r["country"], r.get("year")
        )
        rows.append({
            "Country": intel.country,
            "Risk Score": intel.risk_score,
            "Risk Level": intel.risk_level,
            "Decision": intel.decision
        })

    return pd.DataFrame(rows).sort_values("Risk Score", ascending=False)


# ─────────────────────────────────────────────
# PORTFOLIO ENGINE
# ─────────────────────────────────────────────
def compute_portfolio(exposure, global_df):
    total = sum(exposure.values())

    risk = 0
    for c, w in exposure.items():
        score = global_df.set_index("Country").loc[c]["Risk Score"]
        risk += (w / total) * score

    return {
        "portfolio_risk": round(risk, 3),
        "decision": "Reduce risk" if risk > 0.6 else "Maintain"
    }


# ─────────────────────────────────────────────
# SCENARIO ENGINE
# ─────────────────────────────────────────────
def simulate_scenario(row, shocks, model, scaler, features, country):

    base = compute_risk_intelligence(row, model, scaler, features, country)

    shocked = row.copy()
    for k, v in shocks.items():
        if k in shocked:
            shocked[k] = v

    new = compute_risk_intelligence(shocked, model, scaler, features, country)

    delta = new.risk_score - base.risk_score

    return {
        "before": base.risk_score,
        "after": new.risk_score,
        "change": delta,
        "decision_shift": base.risk_level != new.risk_level
    }
