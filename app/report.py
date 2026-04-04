"""
report.py — Full structured macro intelligence report (text, download-ready).
"""

from datetime import datetime
from forecast import forecast_country
from intelligence import generate_country_intelligence, generate_global_intelligence
from risk_engine import country_risk
from decision_terminal import make_decision
from portfolio import get_allocation
from data_api import load_metrics, load_dataset


def _divider(char="─", width=70):
    return char * width


def _section(title):
    return f"\n{_divider()}\n  {title.upper()}\n{_divider()}\n"


def generate_country_report(country: str) -> str:
    """Generate a full text report for a single country."""
    ts   = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    fc   = forecast_country(country)
    intel = generate_country_intelligence(country)
    rk   = country_risk(country)
    dec  = make_decision(country)
    alloc = get_allocation(country)
    metrics = load_metrics()

    lines = []
    lines.append("=" * 70)
    lines.append(f"  ARES-X MACRO INTELLIGENCE REPORT")
    lines.append(f"  Country: {country.upper()}")
    lines.append(f"  Generated: {ts}")
    lines.append("=" * 70)

    # Model confidence
    lines.append(_section("Model Performance"))
    lines.append(f"  R²   : {metrics.get('r2', 0):.4f}")
    lines.append(f"  RMSE : {metrics.get('rmse', 0):.4f}")
    lines.append(f"  MAE  : {metrics.get('mae', 0):.4f}")
    lines.append(f"  Model Confidence: {fc.get('confidence', 0):.1f}%")

    # Forecast
    lines.append(_section("GDP Forecast"))
    lines.append(f"  Current GDP Growth  : {fc.get('current_gdp', 0):.2f}%")
    lines.append(f"  Forecast GDP Growth : {fc.get('predicted_gdp', 0):.2f}%")
    lines.append(f"  Delta               : {fc.get('delta', 0):+.2f}%")
    lines.append(f"  As of               : {fc.get('date', 'N/A')}")

    # Risk
    lines.append(_section("Risk Assessment"))
    lines.append(f"  Risk Score  : {rk['risk_score']:.1f} / 100")
    lines.append(f"  Risk Level  : {rk['risk_label']}")
    lines.append(f"  Regime      : {intel.get('regime', 'N/A')}")
    lines.append("")
    lines.append("  Risk Components:")
    for k, v in rk.get("components", {}).items():
        lines.append(f"    {k:<22}: {v:.1f}")

    # Executive Intelligence
    lines.append(_section("Executive Intelligence"))
    lines.append("  SUMMARY:")
    summary = intel.get("summary", "").replace("**", "")
    for para in summary.split(". "):
        if para.strip():
            lines.append(f"  {para.strip()}.")

    lines.append("")
    lines.append("  KEY DRIVERS:")
    for i, d in enumerate(intel.get("drivers", []), 1):
        lines.append(f"  {i}. {d}")

    lines.append("")
    lines.append("  SUGGESTED ACTIONS:")
    for i, a in enumerate(intel.get("actions", []), 1):
        lines.append(f"  {i}. {a}")

    # Portfolio
    lines.append(_section("Portfolio Allocation"))
    lines.append(f"  Recommended Regime: {alloc['regime']}")
    lines.append("")
    for asset, pct in alloc["allocation"].items():
        bar = "█" * int(pct / 5)
        lines.append(f"  {asset:<22}: {pct:>5.1f}%  {bar}")

    # Decision
    lines.append(_section("Investment Decision"))
    lines.append(f"  ► DECISION : {dec['decision']}")
    lines.append(f"  ► Score    : {dec['score']}")
    lines.append("")
    lines.append("  RATIONALE:")
    lines.append(f"  {dec['rationale']}")
    lines.append("")
    lines.append("  SUPPORTING FACTORS:")
    for f_ in dec.get("supporting", []):
        lines.append(f"  {f_}")

    lines.append("\n" + "=" * 70)
    lines.append("  ARES-X | Macro Intelligence Terminal | Confidential")
    lines.append("=" * 70)

    return "\n".join(lines)


def generate_global_report() -> str:
    """Generate a global macro intelligence summary report."""
    ts    = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    intel = generate_global_intelligence()
    metrics = load_metrics()

    lines = []
    lines.append("=" * 70)
    lines.append("  ARES-X GLOBAL MACRO INTELLIGENCE REPORT")
    lines.append(f"  Generated: {ts}")
    lines.append("=" * 70)

    lines.append(_section("Global Macro Snapshot"))
    lines.append(f"  Global Regime        : {intel.get('regime', 'N/A')}")
    lines.append(f"  Avg Forecast GDP     : {intel.get('avg_gdp', 0):.2f}%")
    lines.append(f"  Avg Inflation        : {intel.get('avg_inflation', 0):.2f}%")
    lines.append(f"  VIX Level            : {intel.get('global_vix', 0):.2f}")

    lines.append(_section("Global Executive Summary"))
    summary = intel.get("summary", "").replace("**", "")
    lines.append(f"  {summary}")

    lines.append(_section("Top Growth Markets"))
    for c in intel.get("top_growth", []):
        lines.append(f"  ▲ {c}")

    lines.append(_section("Risk-Elevated Markets"))
    for c in intel.get("bottom_growth", []):
        lines.append(f"  ▼ {c}")

    lines.append(_section("Strategic Actions"))
    for i, a in enumerate(intel.get("actions", []), 1):
        lines.append(f"  {i}. {a}")

    lines.append(_section("Model Validation"))
    lines.append(f"  R²   : {metrics.get('r2', 0):.4f}")
    lines.append(f"  RMSE : {metrics.get('rmse', 0):.4f}")
    lines.append(f"  MAE  : {metrics.get('mae', 0):.4f}")

    lines.append("\n" + "=" * 70)
    lines.append("  ARES-X | Macro Intelligence Terminal | Confidential")
    lines.append("=" * 70)

    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────
# PDF EXPORT (ADD ONLY THIS — NO CHANGES ABOVE)
# ─────────────────────────────────────────────────────────────

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from io import BytesIO


def generate_pdf_report(report_text: str):
    """
    Convert report text into PDF (returns BytesIO buffer)
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    content = []

    for line in report_text.split("\n"):
        content.append(Paragraph(line, styles["Normal"]))
        content.append(Spacer(1, 10))

    doc.build(content)
    buffer.seek(0)

    return buffer
