"""
report.py — Executive Intelligence Reporting Layer

Produces decision-grade reports for investors, strategy teams,
and risk committees.

Output is NOT descriptive — it is actionable.
"""

from __future__ import annotations
import io
from datetime import datetime
import pandas as pd

from intelligence import RiskIntelligence, PortfolioIntelligence, ScenarioIntelligence


# ─────────────────────────────────────────────
# COUNTRY REPORT (TOP-TIER VERSION)
# ─────────────────────────────────────────────
def build_country_report(intel: RiskIntelligence) -> str:

    ts = datetime.utcnow().strftime("%d %b %Y, %H:%M UTC")

    drivers = "\n".join(
        f"- **{f}** → {'Risk Driver' if v > 0 else 'Risk Stabiliser'}"
        for f, v in intel.top_drivers[:5]
    )

    return f"""
## ARES-X Sovereign Intelligence Brief

**Country:** {intel.country}  
**Period:** {intel.year or "Latest"}  
**Generated:** {ts}

---

### 1. Decision Signal

**Risk Level:** {intel.risk_level}  
**Risk Score:** {intel.risk_score:.3f}  
**Confidence:** {intel.confidence:.1%}  

**Recommended Action:**  
➡ **{intel.decision}**

---

### 2. What is happening

The model identifies the economy as operating under a **{intel.regime} regime**.

{intel.reasoning}

---

### 3. Why it matters

Current macro conditions imply elevated uncertainty in:
- Capital flows  
- Sovereign stability  
- Currency positioning  

Sustained deterioration may trigger **portfolio drawdowns or allocation shifts**.

---

### 4. What to do

{intel.action}

---

### 5. Key Risk Drivers

{drivers}

---

### 6. Snapshot (Model Inputs)

""" + "\n".join(
        f"- {k}: {round(v,3) if isinstance(v,float) else v}"
        for k, v in list(intel.raw_features.items())[:10]
    ) + f"""

---

*ARES-X Decision Engine | Model: XGBoost | Real-time inference pipeline*
""".strip()


# ─────────────────────────────────────────────
# PORTFOLIO REPORT (ELITE)
# ─────────────────────────────────────────────
def build_portfolio_report(p: PortfolioIntelligence) -> str:

    ts = datetime.utcnow().strftime("%d %b %Y, %H:%M UTC")

    top = p.country_contributions.head(5).to_markdown(index=False)

    return f"""
## ARES-X Portfolio Risk Assessment

**Generated:** {ts}

---

### 1. Portfolio Risk Summary

- **Weighted Risk:** {p.weighted_risk:.3f}  
- **Stress Level:** {p.stress_level}  
- **Estimated VaR (95%):** ${p.var_estimate:.2f}

---

### 2. Risk Concentration

{top}

---

### 3. Interpretation

Portfolio risk is driven primarily by high-exposure countries
with elevated macro instability.

---

### 4. Recommended Action

{p.recommendation}

---

*ARES-X Portfolio Engine — Exposure-weighted sovereign risk model*
""".strip()


# ─────────────────────────────────────────────
# SCENARIO REPORT (ELITE)
# ─────────────────────────────────────────────
def build_scenario_report(s: ScenarioIntelligence) -> str:

    ts = datetime.utcnow().strftime("%d %b %Y, %H:%M UTC")
    sign = "+" if s.delta_risk > 0 else ""

    return f"""
## ARES-X Scenario Impact Analysis

**Country:** {s.baseline.country}  
**Generated:** {ts}

---

### 1. Impact Summary

| Metric | Before | After | Change |
|---|---|---|---|
| Risk Score | {s.baseline.risk_score:.3f} | {s.shocked.risk_score:.3f} | {sign}{s.delta_risk:.3f} |
| Risk Level | {s.baseline.risk_level} | {s.shocked.risk_level} | {"Changed" if s.decision_shifted else "Stable"} |
| Regime | {s.baseline.regime} | {s.shocked.regime} | {"Shifted" if s.baseline.regime != s.shocked.regime else "Same"} |

---

### 2. Interpretation

{s.narrative}

---

### 3. Strategic Implication

- {"Material deterioration — immediate action required." if s.delta_risk > 0.05 else "Moderate impact — monitor closely."}

---

### 4. Recommended Action

➡ {s.shocked.action}

---

*ARES-X Scenario Engine — real-time re-evaluation under macro shocks*
""".strip()


# ─────────────────────────────────────────────
# EXPORTS
# ─────────────────────────────────────────────
def global_risk_to_csv(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


def portfolio_to_csv(p: PortfolioIntelligence) -> bytes:
    buffer = io.BytesIO()
    p.country_contributions.to_csv(buffer, index=False)
    return buffer.getvalue()
