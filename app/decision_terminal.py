"""
decision_terminal.py — Final BUY / HOLD / DEFENSIVE decision engine.
"""

from forecast import forecast_country
from risk_engine import country_risk, macro_score
from utils import regime_label


def make_decision(country: str) -> dict:
    """
    Produce a final investment decision for a country.

    Decision logic:
    - BUY:       High macro score + positive predicted GDP + manageable risk
    - HOLD:      Mixed signals — moderate scores, neither compelling nor alarming
    - DEFENSIVE: High risk score + falling/negative GDP + macro deterioration
    """
    fc     = forecast_country(country)
    rk     = country_risk(country)
    ms     = macro_score(country)

    pred_gdp   = fc.get("predicted_gdp", 0.0)
    curr_gdp   = fc.get("current_gdp",   0.0)
    risk_score = rk["risk_score"]
    inflation  = rk["inflation"]
    regime     = regime_label(pred_gdp, inflation)
    delta      = fc.get("delta", 0.0)
    confidence = fc.get("confidence", 50.0)

    # Scoring system
    score = 0

    # GDP signal
    if pred_gdp > 3.0:      score += 3
    elif pred_gdp > 1.5:    score += 1
    elif pred_gdp <= 0:     score -= 3
    else:                   score -= 1

    # GDP momentum
    if delta > 0.5:          score += 1
    elif delta < -0.5:       score -= 1

    # Risk
    if risk_score < 35:      score += 2
    elif risk_score < 55:    score += 0
    elif risk_score < 70:    score -= 1
    else:                    score -= 3

    # Macro score
    if ms > 65:              score += 2
    elif ms > 45:            score += 0
    else:                    score -= 2

    # Regime bonus/penalty
    if regime == "Expansion":    score += 1
    elif regime == "Recession":  score -= 2
    elif regime == "Stagflation": score -= 1

    # Decision thresholds
    if score >= 4:
        decision = "BUY"
    elif score >= 0:
        decision = "HOLD"
    else:
        decision = "DEFENSIVE"

    # Build rationale
    rationale = _build_rationale(decision, pred_gdp, curr_gdp, delta, risk_score, ms, regime, inflation, confidence)

    # Supporting factors
    supporting = _build_supporting_factors(pred_gdp, delta, risk_score, inflation, regime, ms)

    return {
        "country":    country,
        "decision":   decision,
        "score":      score,
        "rationale":  rationale,
        "supporting": supporting,
        "macro_score":  ms,
        "risk_score":   risk_score,
        "pred_gdp":     pred_gdp,
        "regime":       regime,
        "confidence":   confidence,
    }


def _build_rationale(decision, pred_gdp, curr_gdp, delta, risk_score, ms, regime, inflation, confidence) -> str:
    if decision == "BUY":
        return (
            f"Model forecasts GDP growth of {pred_gdp:.1f}% (Δ {delta:+.1f}% vs current), "
            f"with a macro health score of {ms:.0f}/100 and contained risk score of {risk_score:.0f}. "
            f"The {regime} regime with {confidence:.0f}% model confidence supports a risk-on stance. "
            f"Fundamentals are constructive and entry conditions are favorable."
        )
    elif decision == "HOLD":
        return (
            f"GDP forecast at {pred_gdp:.1f}% (Δ {delta:+.1f}%) presents mixed signals. "
            f"Macro score of {ms:.0f}/100 and risk score of {risk_score:.0f} do not strongly favor "
            f"aggressive positioning. In a {regime} regime with inflation at {inflation:.1f}%, "
            f"maintaining current allocations with selective rebalancing is appropriate."
        )
    else:
        return (
            f"Risk score of {risk_score:.0f}/100 and GDP forecast of {pred_gdp:.1f}% "
            f"(Δ {delta:+.1f}%) signal macro deterioration. The {regime} regime with "
            f"inflation at {inflation:.1f}% reduces the margin of safety. "
            f"A defensive posture — reducing beta, increasing duration, and building liquidity — "
            f"is the model-recommended course of action."
        )


def _build_supporting_factors(pred_gdp, delta, risk_score, inflation, regime, ms) -> list:
    factors = []

    if pred_gdp > 2.5:
        factors.append(f"✅ Strong growth forecast: {pred_gdp:.1f}% GDP")
    elif pred_gdp < 0:
        factors.append(f"⚠️ Negative GDP forecast: {pred_gdp:.1f}% — contraction risk")
    else:
        factors.append(f"⚡ Moderate growth: {pred_gdp:.1f}% GDP — selective opportunity")

    if delta > 0.3:
        factors.append(f"✅ Positive GDP momentum: Δ {delta:+.1f}% forecast vs current")
    elif delta < -0.3:
        factors.append(f"⚠️ Decelerating momentum: Δ {delta:+.1f}% decline expected")

    if risk_score < 40:
        factors.append(f"✅ Low macro risk environment ({risk_score:.0f}/100)")
    elif risk_score > 65:
        factors.append(f"⚠️ Elevated macro risk ({risk_score:.0f}/100) — reduce exposure")

    if inflation > 5.0:
        factors.append(f"⚠️ Inflation pressure ({inflation:.1f}%) compresses real returns")
    elif inflation < 3.0:
        factors.append(f"✅ Inflation contained ({inflation:.1f}%) — supportive for bonds & equities")

    if regime in ["Expansion"]:
        factors.append(f"✅ Expansion regime — cyclical assets historically outperform")
    elif regime in ["Recession", "Stagflation"]:
        factors.append(f"⚠️ {regime} regime — defensive assets and quality bias recommended")

    factors.append(f"📊 Composite macro health score: {ms:.0f}/100")

    return factors[:6]


def bulk_decisions() -> list:
    """Return decisions for all countries."""
    from data_api import load_dataset
    df        = load_dataset()
    countries = df["country"].unique().tolist()

    results = []
    for c in countries:
        try:
            results.append(make_decision(c))
        except Exception:
            pass
    return results