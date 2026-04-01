from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime

def generate_report(country, score, summary, drivers, output_path="report.pdf"):
    doc = SimpleDocTemplate(output_path)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph(f"<b>{country} Sovereign Risk Report</b>", styles["Title"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"Generated: {datetime.now()}", styles["Normal"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"<b>Risk Score:</b> {round(score,3)}", styles["Heading2"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph("<b>Executive Summary:</b>", styles["Heading2"]))
    content.append(Paragraph(summary, styles["Normal"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph("<b>Key Drivers:</b>", styles["Heading2"]))
    for k, v in drivers.items():
        content.append(Paragraph(f"{k}: {round(float(v),3)}", styles["Normal"]))

    doc.build(content)
    return output_path
