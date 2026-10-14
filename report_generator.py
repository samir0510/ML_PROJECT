"""CropCare AI dynamic PDF report generator."""


import os
import io
import re
from datetime import datetime
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

from chatbot import generate_report_content_with_chatbot


def get_master_guide_weather_matrix(disease_name: str, crop: str) -> dict:
    """
    Returns the official Master Guide weather, environmental triggers,
    and scouting schedules for the diagnosed condition.
    """
    d_lower = disease_name.lower()
    if "late blight" in d_lower:
        return {
            "title": "⛈️ Weather & Microclimate Epidemiological Matrix (Late Blight)",
            "risk_level": "CRITICAL EPIDEMIC THREAT (Can collapse entire field within 7–10 days)",
            "risk_color": "#DC2626",
            "rows": [
                ("🌡️ Critical Temperature", "15°C – 20°C (Cool, overcast weather)", "Optimal sporangia formation; spores survive prolonged cloudy weather"),
                ("💧 Humidity & Leaf Wetness", "RH > 90%, free leaf moisture 4–8+ hrs", "Dew, dense fog, or overhead sprinklers trigger explosive spore germination"),
                ("🌧️ Rain & Overcast Spells", "Extended cloud cover & rain spells", "Lack of UV sunlight preserves spores; rain splash spreads spores up canopy"),
                ("💨 Spore Travel / Wind", "Long-range windborne travel (kilometers)", "Airborne spores blow across fields; neighboring farms must be alerted"),
                ("📅 Master Guide Scouting", "Days 0, 3, 6, 9, 12, 18, 24, 30 + Weekly", "MANDATORY: Extra photo check within 2–3 days after rain or cool cloudy spells"),
                ("🚫 Strict Disposal Protocol", "STRICT NO COMPOSTING (Bag or deep bury)", "Compost piles cannot reach heat needed to destroy oospores; discard suspect tubers")
            ]
        }
    elif "early blight" in d_lower:
        return {
            "title": "🌦️ Weather & Microclimate Epidemiological Matrix (Early Blight)",
            "risk_level": "ELEVATED PROGRESSIVE RISK (Attacks lower leaves first under warm humidity)",
            "risk_color": "#D97706",
            "rows": [
                ("🌡️ Critical Temperature", "24°C – 29°C (Warm, humid conditions)", "Optimal mycelial growth and conidia spore production"),
                ("💧 Humidity & Morning Dew", "Alternating wet/dry cycles & heavy dew", "Prolonged leaf moisture allows germ tubes to penetrate leaf stomata"),
                ("🌧️ Rain & Soil-Splash", "Rain splashing soil onto lower foliage", "Primary inoculum in soil; mulch beds & prune lower foliage up to 30 cm"),
                ("💨 Spore Travel / Wind", "Air currents & farm equipment movement", "Spores easily dispersed between crop rows by wind gusts during dry intervals"),
                ("📅 Master Guide Scouting", "Days 0, 5, 10, 15, 20, 30 + Weekly", "Inspect oldest lower foliage first; take extra photo 2–3 days after heavy rains"),
                ("🌿 Crop Rotation & Health", "2-year non-solanaceous rotation", "Rotate away from tomatoes/potatoes/peppers; ensure balanced potassium nutrition")
            ]
        }
    elif "mosaic" in d_lower:
        return {
            "title": "🦠 Environmental & Biosecurity Vector Matrix (Tomato Mosaic Virus)",
            "risk_level": "HIGH SYSTEMIC CONTAGION (Mechanical & Seedborne Vector)",
            "risk_color": "#7C3AED",
            "rows": [
                ("🌡️ Temperature Dynamics", "Warm conditions (25°C – 30°C)", "Accelerates systemic viral replication and vascular movement inside plant"),
                ("💧 Moisture Relationship", "Working in wet foliage worsens spread", "Avoid handling wet plants; plant sap transfers virus on physical contact"),
                ("✂️ Primary Vector Mode", "Pruning shears, hands, staking twine", "Direct mechanical transmission; sanitize tools in 1:4 bleach or skim milk"),
                ("🌱 Seed & Host Reservoir", "Infected seed lots & solanaceous weeds", "Never save seeds from infected plants; eliminate nightshade and weed hosts"),
                ("📅 Master Guide Scouting", "Days 0, 3, 7, 10, 14 + Weekly", "Flag suspect plants immediately; inspect adjacent plants in the row"),
                ("🚫 Treatment Rule", "NO CHEMICAL CURE SPRAY EXISTS", "Immediately rogue out and destroy infected plants; do NOT prune or tie infected vines")
            ]
        }
    else:
        return {
            "title": "🌤️ Preventive Weather & Field Scouting Protocol (Healthy Crop)",
            "risk_level": "BASELINE FIELD MONITORING & PREVENTATIVE CARE",
            "risk_color": "#16A34A",
            "rows": [
                ("🌡️ Favorable Growing Range", "20°C – 28°C (Balanced solanaceous climate)", "Optimal photosynthetic activity and vigorous vegetative/flowering growth"),
                ("💧 Canopy Moisture Management", "Keep foliage dry; soil drip irrigation", "Avoid overhead sprinkler wetting; water in morning so canopy dries rapidly"),
                ("🌧️ Weather Alert Triggers", "Rain spells, dense fog, or cool spells", "Watch for latent late-blight spore clouds blowing in from neighboring fields"),
                ("💨 Air Circulation", "Proper staking and row spacing", "Allows uniform wind movement through canopy to shorten leaf wetness duration"),
                ("📅 Master Guide Scouting", "Every 7 days + Post-Rain Check", "Inspect whole plant weekly; mandatory extra photo within 2–3 days after rain"),
                ("🛡️ Preventative Action", "No chemical spray needed from photo", "Maintain weed-free borders, sanitize tools, and scout lower leaf surfaces")
            ]
        }


def safe_create_paragraph(text: str, style) -> Paragraph:
    """Safely creates a ReportLab Paragraph, falling back to stripped plain text if XML parsing fails."""
    try:
        return Paragraph(text, style)
    except Exception:
        clean = re.sub(r"<[^>]+>", "", text)
        clean = clean.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return Paragraph(clean, style)


def markdown_to_reportlab_elements(md_text: str, styles: dict) -> list:
    """Converts structured Markdown text from the Chatbot into ReportLab Flowables."""
    elements = []
    lines = md_text.split("\n")
    
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            elements.append(Spacer(1, 3))
            continue

        # Escape XML entities before formatting
        line = re.sub(r"&(?!(amp|lt|gt|quot|apos);)", "&amp;", line)
        
        # Handle bold-italic first, then bold, then italic, then code
        line = re.sub(r"\*\*\*(.*?)\*\*\*", r"<b><i>\1</i></b>", line)
        line = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line)
        line = re.sub(r"\*(.*?)\*", r"<i>\1</i>", line)
        line = re.sub(r"`(.*?)`", r"<font name='Courier'>\1</font>", line)

        if line.startswith("### ") or line.startswith("## ") or line.startswith("# "):
            title = re.sub(r"^#+\s*", "", line)
            elements.append(Spacer(1, 5))
            elements.append(safe_create_paragraph(f"<b>{title}</b>", styles["SectionHeading"]))
            elements.append(Spacer(1, 2))
        elif line.startswith("* ") or line.startswith("- ") or line.startswith("• "):
            bullet_body = re.sub(r"^[\*\-•]\s*", "", line)
            elements.append(safe_create_paragraph(f"• {bullet_body}", styles["Bullet"]))
        elif re.match(r"^\d+\.\s+", line):
            num_body = re.sub(r"^\d+\.\s+", "", line)
            elements.append(safe_create_paragraph(f"• {num_body}", styles["Bullet"]))
        else:
            elements.append(safe_create_paragraph(line, styles["Body"]))

    return elements


def generate_pdf_report(
    diag: dict,
    leaf_image: PILImage.Image,
    chatbot_advisory: str = None,
    leaf_confidence: float = None
) -> bytes:
    """
    Builds a professional agronomic diagnostic PDF report.
    The report's agronomic advice is generated dynamically by the Gemini RAG Chatbot
    using the uploaded Master Guide PDF.
    """
    # If no chatbot text provided, generate live from the Chatbot
    if not chatbot_advisory:
        chatbot_advisory = generate_report_content_with_chatbot(diag)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    base_styles = getSampleStyleSheet()

    styles = {
        "HeaderTitle": ParagraphStyle(
            "HTitle",
            parent=base_styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=23,
            textColor=colors.HexColor("#1B5E20")
        ),
        "HeaderSub": ParagraphStyle(
            "HSub",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#4B5563")
        ),
        "SectionHeading": ParagraphStyle(
            "SecHead",
            parent=base_styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=15,
            textColor=colors.HexColor("#111827"),
            spaceBefore=6,
            spaceAfter=4
        ),
        "Body": ParagraphStyle(
            "BodyD",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8.8,
            leading=12.5,
            textColor=colors.HexColor("#1F2937"),
            spaceAfter=4
        ),
        "Bullet": ParagraphStyle(
            "BullD",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#374151"),
            leftIndent=12,
            spaceAfter=3
        ),
        "AiBanner": ParagraphStyle(
            "AiBan",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#2E7D32")
        )
    }

    story = []

    # Header
    report_id = f"CC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    report_date = datetime.now().strftime("%B %d, %Y - %I:%M %p")
    header_data = [
        [
            Paragraph("🌱 <b>CropCare AI — Diagnostic Health Report</b>", styles["HeaderTitle"]),
            Paragraph(f"<para align='right'><b>Report ID:</b> {report_id}<br/><b>Date:</b> {report_date}</para>", styles["HeaderSub"])
        ]
    ]
    header_table = Table(header_data, colWidths=[4.2 * inch, 3.1 * inch])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2E7D32"), spaceAfter=10))

    # Diagnosis summary and leaf photo
    img_buffer = io.BytesIO()
    rgb_thumb = leaf_image.convert("RGB")
    rgb_thumb.thumbnail((240, 240))
    rgb_thumb.save(img_buffer, format="JPEG", quality=90)
    img_buffer.seek(0)
    pdf_leaf_img = RLImage(img_buffer, width=2.3 * inch, height=2.3 * inch)

    crop_name = diag.get("crop", "Solanaceae")
    disease_name = diag.get("disease_name", "Unknown Condition")
    scientific_name = diag.get("scientific_name", "")
    confidence = diag.get("confidence", 0.0)
    severity = diag.get("severity", "Unknown")
    severity_color = diag.get("severity_color", "#C62828")
    leaf_conf_str = f"{leaf_confidence:.1f}%" if leaf_confidence else "Verified"

    diag_info_html = f"""
    <b>Target Crop Species:</b> {crop_name}<br/><br/>
    <b>Identified Condition:</b> <font color='{severity_color}' size='11.5'><b>{disease_name}</b></font><br/>
    <b>Pathogen / Scientific:</b> <i>{scientific_name}</i><br/><br/>
    <b>Stage 2 Diagnostic Confidence:</b> <font color='#1B5E20' size='11'><b>{confidence}%</b></font><br/>
    <b>Stage 1 Leaf Verification:</b> {leaf_conf_str}<br/><br/>
    <b>Risk Severity Rating:</b> <font color='{severity_color}'><b>{severity.upper()}</b></font><br/>
    <b>Initial AI Observation:</b> {diag.get('summary', '')}
    """
    diag_paragraph = Paragraph(diag_info_html, styles["Body"])

    two_col_data = [[pdf_leaf_img, diag_paragraph]]
    two_col_table = Table(two_col_data, colWidths=[2.5 * inch, 4.8 * inch])
    two_col_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#F9FAFB")),
        ("BOX", (1, 0), (1, 0), 1, colors.HexColor("#E5E7EB")),
        ("PADDING", (1, 0), (1, 0), 8),
    ]))
    story.append(two_col_table)
    story.append(Spacer(1, 8))

    # 7-Class confidence breakdown
    story.append(Paragraph("🔬 <b>AI Deep Learning 7-Class Confidence Distribution</b>", styles["SectionHeading"]))
    breakdown = diag.get("breakdown", [])
    
    table_headers = ["Disease Category", "Confidence", "Category Assessment"]
    prob_rows = [table_headers]
    for item in breakdown:
        c_name = item.get("class_name", "")
        p_pct = f"{item.get('percentage', 0.0)}%"
        is_top = (c_name == diag.get("class_name"))
        tag = "<b>PRIMARY DIAGNOSIS</b>" if is_top else "Secondary / Excluded"
        prob_rows.append([
            Paragraph(f"<b>{c_name}</b>" if is_top else c_name, styles["Body"]),
            Paragraph(f"<b>{p_pct}</b>" if is_top else p_pct, styles["Body"]),
            Paragraph(tag, styles["Body"])
        ])

    prob_table = Table(prob_rows, colWidths=[3.2 * inch, 1.3 * inch, 2.8 * inch])
    t_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
    ]
    for r_idx in range(1, len(prob_rows)):
        if r_idx % 2 == 0:
            t_style.append(("BACKGROUND", (0, r_idx), (-1, r_idx), colors.HexColor("#F9FAFB")))
        if breakdown and r_idx == 1:
            t_style.append(("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#E8F5E9")))
    prob_table.setStyle(TableStyle(t_style))
    story.append(prob_table)
    story.append(Spacer(1, 8))

    # Weather and epidemiological risk matrix
    weather_info = get_master_guide_weather_matrix(disease_name, crop_name)
    story.append(Paragraph(f"<b>{weather_info['title']}</b>", styles["SectionHeading"]))
    story.append(Paragraph(
        f"<font color='{weather_info['risk_color']}'><b>EPIDEMIOLOGICAL RISK LEVEL:</b> {weather_info['risk_level']}</font>",
        styles["Body"]
    ))
    story.append(Spacer(1, 4))

    weather_table_headers = ["Environmental Factor", "Trigger / Threshold (Master Guide)", "Field Impact & Protocol"]
    weather_rows = [weather_table_headers]
    for row in weather_info["rows"]:
        weather_rows.append([
            Paragraph(f"<b>{row[0]}</b>", styles["Body"]),
            Paragraph(row[1], styles["Body"]),
            Paragraph(row[2], styles["Body"])
        ])

    w_table = Table(weather_rows, colWidths=[1.8 * inch, 2.5 * inch, 3.0 * inch])
    w_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
    ]
    for r_idx in range(1, len(weather_rows)):
        if r_idx % 2 == 0:
            w_style.append(("BACKGROUND", (0, r_idx), (-1, r_idx), colors.HexColor("#F0F9FF")))
    w_table.setStyle(TableStyle(w_style))
    story.append(w_table)
    story.append(Spacer(1, 10))

    # Dynamic chatbot agronomic advisory
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E5E7EB"), spaceAfter=8))
    story.append(Paragraph("💬 <b>Grounded Agronomic Analysis (Generated Live by Chatbot via Master Guide)</b>", styles["AiBanner"]))
    story.append(Spacer(1, 4))

    # Parse and append dynamic chatbot advisory elements
    advisory_elements = markdown_to_reportlab_elements(chatbot_advisory, styles)
    story.extend(advisory_elements)
    story.append(Spacer(1, 8))

    # Safety disclaimer
    disclaimer_html = """
    <b>⚠️ Important Regulatory & Safety Disclaimer:</b><br/>
    This report was dynamically generated by the CropCare AI Chatbot grounded in the official Tomato & Potato Farmer Master Guide. Always consult local registered pesticide labels for exact dosage rates, PPE requirements, and Pre-Harvest Intervals (PHI) before application. Consult your local agricultural extension service for on-site verification.
    """
    disclaimer_table = Table([[Paragraph(disclaimer_html, styles["Body"])]], colWidths=[7.3 * inch])
    disclaimer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFBEB")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#FCD34D")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(disclaimer_table)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
