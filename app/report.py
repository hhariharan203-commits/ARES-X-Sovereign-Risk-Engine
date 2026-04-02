from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def generate_report(country, score, intelligence):

    file_path = f"{country}_risk_report.pdf"

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = []

    # Title
    content.append(Paragraph(f"{country} Sovereign Risk Report", styles["Title"]))
    content.append(Spacer(1, 12))

    # Score
    content.append(Paragraph(f"<b>Risk Score:</b> {round(score,3)}", styles["Normal"]))
    content.append(Spacer(1, 12))

    # Regime
    content.append(Paragraph(f"<b>Regime:</b> {intelligence['regime']}", styles["Normal"]))
    content.append(Spacer(1, 12))

    # Drivers
    content.append(Paragraph("<b>Key Drivers:</b>", styles["Normal"]))
    for d in intelligence["drivers"]:
        content.append(Paragraph(f"- {d}", styles["Normal"]))
    content.append(Spacer(1, 12))

    # Action
    content.append(Paragraph(f"<b>Recommended Action:</b> {intelligence['action']}", styles["Normal"]))

    doc.build(content)

    return file_path
