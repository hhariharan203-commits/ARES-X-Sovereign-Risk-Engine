"""
Sovereign Risk Decision Intelligence Engine
Core Analytics, Regime Detection, Decision Logic
Institutional Grade — BlackRock / Goldman Sachs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


# ─── Decision Thresholds ────────────────────────────────────────────────────

DECISION_MATRIX = {
    "Invest":  (0.00, 0.30),
    "Hold":    (0.30, 0.50),
    "Reduce":  (0.50, 0.70),
    "Exit":    (0.70, 1.01),
}

REGIME_THRESHOLDS = {
    "Stable":           (0.00, 0.35),
    "Inflation Stress": (0.35, 0.60),
    "Crisis":           (0.60, 1.01),
}

DECISION_COLORS = {
    "Invest": "#00C896",
    "Hold":   "#F5C518",
    "Reduce": "#FF8C42",
    "Exit":   "#FF3B5C",
}

REGIME_COLORS = {
    "Stable":           "#00C896",
    "Inflation Stress": "#F5C518",
    "Crisis":           "#FF3B5C",
}


# ─── Core Classifiers ────────────────────────────────────────────────────────

def risk_to_decision(risk_score: float) -> str:
    for decision, (lo, hi) in DECISION_MATRIX.items():
        if lo <= risk_score < hi:
            return decision
    return "Exit"


def risk_to_regime(risk_score: float) -> str:
    for regime, (lo, hi) in REGIME_THRESHOLDS.items():
        if lo <= risk_score < hi:
            return regime
    return "Crisis"


def get_confidence(risk_score: float) -> Tuple[str, float]:
    boundaries = [0.30, 0.50, 0.70]
    min_dist = min(abs(risk_score - b) for b in boundaries)

    if min_dist > 0.15:
        return "High", round(min(0.85 + min_dist * 0.30, 0.97), 2)
    elif min_dist > 0.08:
        return "Moderate", round(0.65 + min_dist * 0.50, 2)
    else:
        return "Low", round(max(0.45 + min_dist * 0.80, 0.40), 2)


# ─── Narrative Engine ────────────────────────────────────────────────────────

def _format_feature_name(name: str) -> str:
    return (
        name
        .replace("_lag1", " (1M Lag)")
        .replace("_lag3", " (3M Lag)")
        .replace("_roll3", " (3M Trend)")
        .replace("_momentum", " Momentum")
        .replace("_", " ")
        .title()
    )


def build_reasoning(
    country: str,
    risk_score: float,
    decision: str,
    regime: str,
    top_drivers: List[Tuple[str, float]],
) -> Dict:

    def fmt(d):
        name = _format_feature_name(d[0])
        sign = "+" if d[1] > 0 else ""
        return f"{name} ({sign}{d[1]:.3f})"

    risk_pct = f"{risk_score:.1%}"
    pos = [d for d in top_drivers if d[1] > 0]
    neg = [d for d in top_drivers if d[1] <= 0]

    if decision == "Invest":
        drivers = ", ".join(fmt(d) for d in neg[:2]) or "contained macro stress"
        narrative = (
            f"{country} presents a compelling entry point at {risk_pct} risk. "
            f"Macro conditions remain stable with limited downside exposure. "
            f"Key stabilizing factors: {drivers}."
        )
        impact = "Favorable environment for capital deployment and yield capture."
        action = "Initiate or scale exposure. Allocate 3–5% with medium-term horizon."

    elif decision == "Hold":
        drivers = ", ".join(fmt(d) for d in top_drivers[:2])
        narrative = (
            f"{country} shows mixed signals at {risk_pct}. "
            f"Risk trajectory remains uncertain. Key variables: {drivers}."
        )
        impact = "Neutral positioning; wait for directional confirmation."
        action = "Maintain exposure. Monitor closely for macro shifts."

    elif decision == "Reduce":
        drivers = ", ".join(fmt(d) for d in pos[:2])
        narrative = (
            f"{country} risk is rising under {regime} regime at {risk_pct}. "
            f"Stress factors: {drivers}."
        )
        impact = "Downside risk outweighs yield potential."
        action = "Reduce exposure. Hedge or rotate capital."

    else:
        drivers = ", ".join(fmt(d) for d in top_drivers[:3])
        narrative = (
            f"{country} has breached risk thresholds at {risk_pct}. "
            f"Systemic risk present. Drivers: {drivers}."
        )
        impact = "Capital preservation required."
        action = "Exit immediately and reallocate to safe assets."

    return {
        "narrative": narrative,
        "impact": impact,
        "action": action,
        "drivers": top_drivers[:5],
    }


# ─── Feature Attribution ─────────────────────────────────────────────────────

def compute_drivers_batch(model, scaler, feature_cols, df, base_probs):
    X = df[feature_cols].values.astype(float)
    epsilon = 0.05
    results = []

    for i in range(len(df)):
        base_prob = base_probs[i]
        base = X[i:i+1].copy()
        impacts = []

        for j, feat in enumerate(feature_cols):
            perturbed = base.copy()
            perturbed[0, j] += epsilon * (abs(base[0, j]) + 1)

            try:
                scaled = scaler.transform(perturbed)
                prob = model.predict_proba(scaled)[0][1]
                impact = (prob - base_prob) / epsilon
            except:
                impact = 0.0

            impacts.append((feat, round(impact, 4)))

        impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        results.append(impacts[:5])

    return results


# ─── Global Summary ─────────────────────────────────────────────────────────

def global_risk_summary(df: pd.DataFrame) -> Dict:
    avg = float(df["risk_score"].mean())

    if avg > 0.65:
        status = "CRITICAL"
        color = "#FF3B5C"
    elif avg > 0.45:
        status = "ELEVATED"
        color = "#FF8C42"
    else:
        status = "STABLE"
        color = "#00C896"

    return {
        "avg_risk": round(avg, 4),
        "status": status,
        "status_color": color,
        "decision": risk_to_decision(avg),
        "total_countries": len(df),
    }


# ─── Scenario Engine ─────────────────────────────────────────────────────────

def apply_scenario_shock(row, feature_cols, shocks):
    shocked = row.copy()

    for k, delta in shocks.items():
        for feat in feature_cols:
            if k.lower() in feat.lower():
                try:
                    shocked[feat] = float(shocked[feat]) + delta
                except:
                    pass

    return shocked


def run_scenario(model, scaler, feature_cols, row, shocks, country):

    before = float(row["risk_score"])
    shocked_row = apply_scenario_shock(row, feature_cols, shocks)

    try:
        X = shocked_row[feature_cols].values.astype(float).reshape(1, -1)
        Xs = scaler.transform(X)
        after = float(model.predict_proba(Xs)[0][1])
    except:
        after = before

    delta = after - before

    return {
        "before": before,
        "after": after,
        "delta": delta,
        "decision_before": risk_to_decision(before),
        "decision_after": risk_to_decision(after),
    }


# ─── Main Enrichment ─────────────────────────────────────────────────────────

def enrich_dataframe(df, model, scaler, feature_cols):

    df = df.copy()

    X = df[feature_cols].values.astype(float)
    Xs = scaler.transform(X)
    probs = model.predict_proba(Xs)[:, 1]

    df["risk_score"] = np.clip(probs, 0, 1)

    df["decision"] = df["risk_score"].apply(risk_to_decision)
    df["regime"] = df["risk_score"].apply(risk_to_regime)

    conf = df["risk_score"].apply(get_confidence)
    df["confidence_label"] = conf.apply(lambda x: x[0])
    df["confidence_value"] = conf.apply(lambda x: x[1])

    drivers = compute_drivers_batch(model, scaler, feature_cols, df, probs)
    df["drivers"] = drivers

    country_col = "country" if "country" in df.columns else df.columns[0]

    reasoning = []
    impact = []
    action = []

    countries = df[country_col].astype(str).values
    scores = df["risk_score"].values
    decisions = df["decision"].values
    regimes = df["regime"].values

    for i in range(len(df)):
        r = build_reasoning(
            countries[i],
            scores[i],
            decisions[i],
            regimes[i],
            drivers[i]
        )
        reasoning.append(r["narrative"])
        impact.append(r["impact"])
        action.append(r["action"])

    df["reasoning"] = reasoning
    df["impact"] = impact
    df["action"] = action

    return df
