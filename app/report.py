"""
report.py — Full structured macro intelligence report (text + premium PDF).
"""

from datetime import datetime
from forecast import forecast_country
from intelligence import generate_country_intelligence, generate_global_intelligence
from risk_engine import country_risk
from decision_terminal import make_decision
from portfolio import get_allocation
from data_api import load_metrics


# ─────────────────────────────────────────
# TEXT REPORT
# ─────────────────────────────────────────

def _divider(char="─", width=70):
    return char * width


def _section(title):
    return f"\n{_divider()}\n  {title.upper()}\n{_divider()}\n"


def generate_country_report(country: str) -> str:
    ts   = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    fc   = forecast_country(country)
    intel = generate_country_intelligence(country)
    rk   = country_risk(country)
    dec  = make_decision(country)
    alloc = get_allocation(country)
    metrics = load_metrics()

    lines = []
    lines.append("=" * 70)
    lines.append("  ARES-X MACRO INTELLIGENCE REPORT")
    lines.append(f"  Country: {country.upper()}")
    lines.append(f"  Generated: {ts}")
    lines.append("=" * 70)

    # ───────── EXECUTIVE SUMMARY ─────────
    lines.append(_section("Executive Summary"))
    lines.append(
        f"  {country.upper()} is rated {dec['decision']} with GDP forecast at "
        f"{fc.get('predicted_gdp', 0):.1f}% and risk score {rk['risk_score']:.0f}."
    )
    lines.append(
        f"  Macro score of {dec['macro_score']:.0f}/100 under a {intel.get('regime', 'N/A')} regime "
        f"supports this positioning with {fc.get('confidence', 0):.0f}% model confidence."
    )

    # ───────── INVESTMENT THESIS ─────────
    lines.append(_section("Investment Thesis"))

    lines.append(
        f"  {country.upper()} presents a {dec['decision']} opportunity driven by:"
    )

    if fc.get("predicted_gdp", 0) > 2:
        lines.append("  • Strong GDP growth momentum")

    if rk["inflation"] < 5:
        lines.append("  • Controlled inflation environment")

    if rk["unemployment"] < 6:
        lines.append("  • Stable labor market conditions")

    if rk["risk_score"] < 60:
        lines.append("  • Moderate macro risk profile")

    lines.append("")
    lines.append(
        f"  The current {intel.get('regime', 'N/A')} regime supports this positioning "
        f"with favorable macro dynamics."
    )

    # Model
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

    # Risk
    lines.append(_section("Risk Assessment"))
    lines.append(f"  Risk Score  : {rk['risk_score']:.1f} / 100")
    lines.append(f"  Risk Level  : {rk['risk_label']}")

    # Decision
    lines.append(_section("Investment Decision"))
    lines.append(f"  ► DECISION : {dec['decision']}")
    lines.append(f"  ► Score    : {dec['score']}")
    lines.append("")
    lines.append("  RATIONALE:")
    lines.append(f"  {dec['rationale']}")

    lines.append("\n" + "=" * 70)
    lines.append("  ARES-X | Macro Intelligence Terminal | Confidential")
    lines.append("=" * 70)

    return "\n".join(lines)


def generate_global_report() -> str:
    ts    = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    intel = generate_global_intelligence()
    metrics = load_metrics()

    lines = []
    lines.append("=" * 70)
    lines.append("  ARES-X GLOBAL MACRO INTELLIGENCE REPORT")
    lines.append(f"  Generated: {ts}")
    lines.append("=" * 70)

    lines.append(_section("Global Snapshot"))
    lines.append(f"  Global Regime : {intel.get('regime', 'N/A')}")

    lines.append("\n" + "=" * 70)
    lines.append("  ARES-X | Macro Intelligence Terminal | Confidential")
    lines.append("=" * 70)

    return "\n".join(lines)


# ─────────────────────────────────────────
# PREMIUM PDF (NO EXTRA DEPENDENCIES)
# ─────────────────────────────────────────

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from io import BytesIO


def generate_pdf_report(report_text: str):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    # Premium styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor("#00E5FF"),
        spaceAfter=14
    )

    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor("#000000"),
        spaceAfter=10
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor("#333333"),
        spaceAfter=6
    )

    content = []
    lines = report_text.split("\n")

    for line in lines:
        line = line.strip()

        if not line:
            content.append(Spacer(1, 6))
            continue

        if "ARES-X" in line:
            content.append(Paragraph(line, title_style))

        elif "──" in line or "===" in line:
            continue

        elif line.isupper():
            content.append(Paragraph(line, section_style))

        elif line.startswith("►"):
            content.append(Paragraph(f"<b>{line}</b>", body_style))

        else:
            content.append(Paragraph(line, body_style))

    doc.build(content)
    buffer.seek(0)

    return buffer
