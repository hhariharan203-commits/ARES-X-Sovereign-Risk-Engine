"""
Sovereign Risk Decision Intelligence Engine
Core Analytics, Regime Detection, Decision Logic
Institutional Grade — BlackRock / Goldman Sachs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


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
    """
    Confidence derived from distance to nearest decision boundary.
    Fully deterministic — zero randomness.
    """
    boundaries = [0.30, 0.50, 0.70]
    min_dist   = min(abs(risk_score - b) for b in boundaries)
    if min_dist > 0.15:
        label = "High"
        value = round(min(0.85 + min_dist * 0.30, 0.97), 2)
    elif min_dist > 0.08:
        label = "Moderate"
        value = round(0.65 + min_dist * 0.50, 2)
    else:
        label = "Low"
        value = round(max(0.45 + min_dist * 0.80, 0.40), 2)
    return label, value


# ─── Narrative Engine ────────────────────────────────────────────────────────

def build_reasoning(
    country: str,
    risk_score: float,
    decision: str,
    regime: str,
    top_drivers: List[Tuple[str, float]],
) -> Dict:
    """
    Deterministic executive-grade reasoning and action narrative.
    All output driven by model inputs — no placeholders, no random text.
    """
    pos_drivers = [d for d in top_drivers[:5] if d[1] > 0]
    neg_drivers = [d for d in top_drivers[:5] if d[1] <= 0]

    def fmt(d: Tuple[str, float]) -> str:
        name = d[0].replace("_", " ").title()
        sign = "+" if d[1] > 0 else ""
        return f"{name} ({sign}{d[1]:.3f})"

    risk_pct = f"{risk_score:.1%}"

    if decision == "Invest":
        primary = (
            ", ".join(fmt(d) for d in neg_drivers[:2])
            if neg_drivers else "contained macro stress indicators"
        )
        narrative = (
            f"{country} presents a risk-adjusted entry opportunity at a composite risk index of {risk_pct}. "
            f"Fundamental indicators remain broadly constructive with downside risk contained below institutional thresholds. "
            f"Key supporting factors: {primary}. "
            f"Sovereign spread dynamics and stable fiscal trajectory support capital deployment at current levels."
        )
        action = (
            "Initiate or scale sovereign exposure with conviction. "
            "Target 3–5% strategic portfolio allocation with an 18-month investment horizon. "
            "Layer duration via 5Y and 10Y sovereign bonds; engage local prime brokers for execution efficiency."
        )

    elif decision == "Hold":
        primary = (
            ", ".join(fmt(d) for d in top_drivers[:2])
            if top_drivers else "mixed macro signals"
        )
        narrative = (
            f"{country} is exhibiting non-directional risk dynamics at {risk_pct}, consistent with a transitional macro environment. "
            f"Deterioration has not crossed institutional re-rating thresholds, but positive momentum is absent. "
            f"Key variables under surveillance: {primary}. "
            f"Position extension is premature; capital preservation remains the near-term priority."
        )
        action = (
            "Maintain current position sizing without adding duration or notional exposure. "
            "Re-evaluate at next macro data release, credit rating event, or central bank decision. "
            "Set automated risk alerts at the 0.50 threshold to trigger formal review protocol."
        )

    elif decision == "Reduce":
        primary = (
            ", ".join(fmt(d) for d in pos_drivers[:2])
            if pos_drivers else "deteriorating fundamental indicators"
        )
        narrative = (
            f"{country} is exhibiting fundamental deterioration consistent with the {regime} regime at {risk_pct}. "
            f"Elevated sovereign exposure is unjustified at prevailing risk premium levels — carry is no longer compensating for downside risk. "
            f"Escalating stress factors: {primary}. "
            f"Negative macro momentum and elevated contagion risk to regional peers demand immediate tactical repositioning."
        )
        action = (
            "Reduce to minimum strategic weight within 5 trading sessions. "
            "Hedge residual exposure via sovereign CDS or reduce portfolio duration by 40–60%. "
            "Rotate proceeds into investment-grade safe-haven instruments (USD, CHF, or JPY denominated)."
        )

    else:  # Exit
        primary = (
            ", ".join(fmt(d) for d in top_drivers[:3])
            if top_drivers else "systemic risk threshold breach"
        )
        narrative = (
            f"{country} has breached institutional risk tolerance thresholds at {risk_pct} — {regime} regime confirmed. "
            f"Systemic stress indicators have triggered mandatory capital preservation protocols; structural impairment risk is material. "
            f"Critical risk drivers: {primary}. "
            f"Fundamental deterioration is no longer transient; sovereign debt sustainability is under active market scrutiny."
        )
        action = (
            "Execute full exit within 2 trading sessions. "
            "Transfer capital to liquid safe-haven instruments immediately upon exit execution. "
            "Activate capital preservation protocols and notify the risk committee of exposure breach. "
            "Reinitiation is prohibited until risk score registers below 0.50 across two consecutive review cycles."
        )

    return {
        "narrative": narrative,
        "action":    action,
        "drivers":   top_drivers[:5],
    }


# ─── Feature Attribution — Single Row ───────────────────────────────────────

def compute_top_drivers(
    model,
    scaler,
    feature_cols: List[str],
    row: pd.Series,
) -> List[Tuple[str, float]]:
    """
    Marginal perturbation attribution for a single row.
    Used in Scenario Lab (post-shock re-attribution).
    No randomness — returns zero-impact list only when model is unavailable.
    """
    try:
        base_arr    = row[feature_cols].values.astype(float).reshape(1, -1)
        base_scaled = scaler.transform(base_arr)
        base_prob   = float(model.predict_proba(base_scaled)[0][1])
    except Exception:
        return [(f, 0.0) for f in feature_cols[:5]]

    impacts = []
    epsilon = 0.05

    for i, feat in enumerate(feature_cols):
        perturbed       = base_arr.copy()
        perturbed[0, i] += epsilon * (abs(base_arr[0, i]) + 1.0)
        try:
            p_scaled = scaler.transform(perturbed)
            p_prob   = float(model.predict_proba(p_scaled)[0][1])
            impact   = (p_prob - base_prob) / epsilon
        except Exception:
            impact = 0.0
        impacts.append((feat, round(impact, 4)))

    impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    return impacts[:5]


# ─── Feature Attribution — Batch ─────────────────────────────────────────────

def compute_drivers_batch(
    model,
    scaler,
    feature_cols: List[str],
    df: pd.DataFrame,
    base_probs: np.ndarray,
) -> List[List[Tuple[str, float]]]:
    """
    Batch perturbation attribution across entire dataset.
    Accepts pre-computed base_probs to avoid redundant model calls.
    Single model-call loop — never called again at render time.
    """
    X       = df[feature_cols].values.astype(float)
    epsilon = 0.05
    results = []

    for row_idx in range(len(df)):
        base_prob = float(base_probs[row_idx])
        base_arr  = X[row_idx : row_idx + 1].copy()
        impacts   = []

        for i, feat in enumerate(feature_cols):
            perturbed       = base_arr.copy()
            perturbed[0, i] += epsilon * (abs(base_arr[0, i]) + 1.0)
            try:
                p_scaled = scaler.transform(perturbed)
                p_prob   = float(model.predict_proba(p_scaled)[0][1])
                impact   = (p_prob - base_prob) / epsilon
            except Exception:
                impact = 0.0
            impacts.append((feat, round(impact, 4)))

        impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        results.append(impacts[:5])

    return results


# ─── Global Risk Summary ─────────────────────────────────────────────────────

def global_risk_summary(df: pd.DataFrame) -> Dict:
    avg_risk = float(df["risk_score"].mean())
    decision = risk_to_decision(avg_risk)

    if avg_risk > 0.65:
        status       = "CRITICAL"
        status_color = "#FF3B5C"
        headline     = (
            "Global sovereign stress is systemic. "
            "Capital protection protocols apply across all emerging and frontier exposure."
        )
    elif avg_risk > 0.45:
        status       = "ELEVATED"
        status_color = "#FF8C42"
        headline     = (
            "Risk dispersion is widening across the sovereign universe. "
            "Selective de-risking and active hedging are warranted."
        )
    else:
        status       = "STABLE"
        status_color = "#00C896"
        headline     = (
            "Macro environment is broadly constructive. "
            "Selective deployment opportunities exist in investment-grade sovereigns."
        )

    top10   = df.nlargest(10, "risk_score")[["country", "risk_score", "decision", "regime"]].copy()
    bottom5 = df.nsmallest(5, "risk_score")[["country", "risk_score", "decision"]].copy()

    return {
        "avg_risk":        round(avg_risk, 4),
        "status":          status,
        "status_color":    status_color,
        "headline":        headline,
        "decision":        decision,
        "top10":           top10,
        "bottom5":         bottom5,
        "decision_dist": {
            "Invest": int((df["decision"] == "Invest").sum()),
            "Hold":   int((df["decision"] == "Hold").sum()),
            "Reduce": int((df["decision"] == "Reduce").sum()),
            "Exit":   int((df["decision"] == "Exit").sum()),
        },
        "total_countries": len(df),
    }


# ─── Scenario Engine ─────────────────────────────────────────────────────────

def apply_scenario_shock(
    row: pd.Series,
    feature_cols: List[str],
    shocks: Dict[str, float],
) -> pd.Series:
    """Apply macro shocks to a country feature vector. Deterministic, zero randomness."""
    shocked = row.copy()

    shock_map = {
        "inflation": ["inflation", "cpi", "inflation_rate", "consumer_price", "price_level"],
        "gdp":       ["gdp", "gdp_growth", "real_gdp", "growth", "output"],
        "interest":  ["interest", "rate", "policy_rate", "lending_rate", "base_rate"],
    }

    for shock_key, delta in shocks.items():
        if delta == 0.0:
            continue
        keywords = shock_map.get(shock_key, [])
        for feat in feature_cols:
            if any(kw in feat.lower() for kw in keywords):
                if feat in shocked.index:
                    try:
                        shocked[feat] = float(shocked[feat]) + delta
                    except (ValueError, TypeError):
                        pass

    return shocked


def run_scenario(
    model,
    scaler,
    feature_cols: List[str],
    row: pd.Series,
    shocks: Dict[str, float],
    country: str,
) -> Dict:
    """
    Full before/after scenario comparison.
    Before score sourced from pre-enriched row — no redundant baseline model call.
    After score computed from shocked feature vector via model — no synthetic fallbacks.
    """
    before_score = float(np.clip(row["risk_score"], 0.0, 1.0))
    shocked_row  = apply_scenario_shock(row, feature_cols, shocks)

    try:
        shock_feats  = shocked_row[feature_cols].values.astype(float).reshape(1, -1)
        shock_scaled = scaler.transform(shock_feats)
        after_score  = float(np.clip(model.predict_proba(shock_scaled)[0][1], 0.0, 1.0))
    except Exception:
        # If model call fails on shocked vector, hold baseline — do not fabricate
        after_score = before_score

    delta            = round(after_score - before_score, 4)
    before_decision  = risk_to_decision(before_score)
    after_decision   = risk_to_decision(after_score)
    before_regime    = risk_to_regime(before_score)
    after_regime     = risk_to_regime(after_score)
    decision_changed = before_decision != after_decision

    if delta > 0.05:
        impact_label = "RISK INCREASE"
        impact_color = "#FF3B5C"
    elif delta < -0.05:
        impact_label = "RISK REDUCTION"
        impact_color = "#00C896"
    else:
        impact_label = "NEUTRAL"
        impact_color = "#8B8FA8"

    return {
        "country":          country,
        "before_score":     round(before_score, 4),
        "after_score":      round(after_score, 4),
        "delta":            delta,
        "before_decision":  before_decision,
        "after_decision":   after_decision,
        "before_regime":    before_regime,
        "after_regime":     after_regime,
        "decision_changed": decision_changed,
        "impact_label":     impact_label,
        "impact_color":     impact_color,
    }


# ─── Dataset Enrichment — Full Pipeline ──────────────────────────────────────

def enrich_dataframe(
    df: pd.DataFrame,
    model,
    scaler,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Complete enrichment pass — invoked ONCE at startup, result cached by app.py.

    Columns added:
        risk_score        float  — model probability output, clipped [0, 1]
        decision          str    — Invest / Hold / Reduce / Exit
        regime            str    — Stable / Inflation Stress / Crisis
        confidence_label  str    — High / Moderate / Low
        confidence_value  float  — [0.40, 0.97]
        drivers           list   — [(feature, impact), ...] top 5
        reasoning         str    — executive narrative (what is happening + why it matters)
        action            str    — institutional recommendation (what to do)

    Raises on model/scaler failure so app.py can surface a clean error message.
    No random values used anywhere in this pipeline.
    """
    df = df.copy()

    # ── Vectorised risk scoring ──────────────────────────────────────────────
    X        = df[feature_cols].values.astype(float)
    X_scaled = scaler.transform(X)
    probs    = model.predict_proba(X_scaled)[:, 1]
    df["risk_score"] = np.clip(probs, 0.0, 1.0).round(4)

    # ── Decision + regime ────────────────────────────────────────────────────
    df["decision"] = df["risk_score"].apply(risk_to_decision)
    df["regime"]   = df["risk_score"].apply(risk_to_regime)

    # ── Confidence (deterministic, boundary-distance-based) ─────────────────
    conf_results          = df["risk_score"].apply(get_confidence)
    df["confidence_label"] = conf_results.apply(lambda x: x[0])
    df["confidence_value"] = conf_results.apply(lambda x: x[1])

    # ── Batch driver attribution (single perturbation pass) ──────────────────
    all_drivers  = compute_drivers_batch(model, scaler, feature_cols, df, probs)
    df["drivers"] = all_drivers

    # ── Reasoning + action (deterministic, driver-integrated per row) ────────
    country_col = "country" if "country" in df.columns else df.columns[0]

    reasoning_list: List[str] = []
    action_list:    List[str] = []

    for idx, row in df.iterrows():
        country = str(row.get(country_col, f"Sovereign_{idx}"))
        r = build_reasoning(
            country     = country,
            risk_score  = float(row["risk_score"]),
            decision    = str(row["decision"]),
            regime      = str(row["regime"]),
            top_drivers = list(row["drivers"]),
        )
        reasoning_list.append(r["narrative"])
        action_list.append(r["action"])

    df["reasoning"] = reasoning_list
    df["action"]    = action_list

    return df
