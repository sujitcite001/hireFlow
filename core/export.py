"""Export candidate evaluation results to CSV and PDF formats."""

import csv
from datetime import datetime
import io
from typing import Any

from core.models import Candidate, CandidateEvaluation, JobDescription


def generate_csv_report(results: list[dict[str, Any]]) -> str:
    """Export ranked candidate evaluations to a CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Rank",
        "Candidate ID",
        "Candidate Name",
        "Fit Score (0-10)",
        "Search Score",
        "Retrieval Mode",
        "Identified Skills",
        "Experience (Years)",
        "Location",
        "Summary",
        "Strengths",
        "Gaps",
        "Risks",
        "Key Evidence",
    ])

    for rank, item in enumerate(results, 1):
        candidate: Candidate = item["candidate"]
        evaluation: CandidateEvaluation = item["evaluation"]
        writer.writerow([
            rank,
            candidate.candidate_id,
            candidate.name,
            f"{evaluation.fit_score:.1f}",
            f"{float(item.get('score', 0.0)):.2f}",
            item.get("retrieval_mode", "BM25"),
            "; ".join(candidate.skills),
            f"{candidate.experience_years:g}" if candidate.experience_years is not None else "Not specified",
            candidate.location or "Not specified",
            evaluation.summary,
            " | ".join(evaluation.strengths),
            " | ".join(evaluation.gaps),
            " | ".join(evaluation.risks),
            " | ".join(evaluation.evidence),
        ])

    return output.getvalue()


def generate_pdf_report(job: JobDescription, results: list[dict[str, Any]]) -> bytes:
    """Generate a recruiter-ready PDF evaluation packet using ReportLab."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        # Fallback minimal plain text/PDF representation if reportlab is unavailable
        return b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 0/Kids[]>>endobj\nxref\n0 3\n0000000000 65535 f\n0000000009 00000 n\n0000000052 00000 n\ntrailer<</Size 3/Root 1 0 R>>\nstartxref\n101\n%%EOF"

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        alignment=0,
    )
    subtitle_style = ParagraphStyle(
        "SubTitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4A5568"),
    )
    h2_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#2D3748"),
    )
    bold_style = ParagraphStyle(
        "BodyDarkBold",
        parent=body_style,
        fontName="Helvetica-Bold",
    )
    quote_style = ParagraphStyle(
        "QuoteText",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#4A5568"),
        leftIndent=12,
        fontName="Helvetica-Oblique",
    )

    story = []
    # Title & Metadata
    story.append(Paragraph("HireFlow Candidate Evaluation Report", title_style))
    story.append(Paragraph(
        f"<b>Role:</b> {job.title} | <b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
        f"<b>Evaluated Candidates:</b> {len(results)}",
        subtitle_style,
    ))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E0"), spaceAfter=12))

    if not results:
        story.append(Paragraph("No candidates met the search and filter criteria.", body_style))
        doc.build(story)
        return buffer.getvalue()

    # Executive Summary Table
    story.append(Paragraph("Summary of Top Candidates", h2_style))
    table_data = [[
        Paragraph("<b>Rank</b>", bold_style),
        Paragraph("<b>Candidate</b>", bold_style),
        Paragraph("<b>Fit Score</b>", bold_style),
        Paragraph("<b>Experience</b>", bold_style),
        Paragraph("<b>Location</b>", bold_style),
        Paragraph("<b>Top Skills</b>", bold_style),
    ]]
    for rank, item in enumerate(results[:10], 1):
        cand: Candidate = item["candidate"]
        evaln: CandidateEvaluation = item["evaluation"]
        exp = f"{cand.experience_years:g} yrs" if cand.experience_years is not None else "N/A"
        skills_str = ", ".join(cand.skills[:4]) + ("..." if len(cand.skills) > 4 else "")
        table_data.append([
            Paragraph(str(rank), body_style),
            Paragraph(f"<b>{cand.name}</b>", body_style),
            Paragraph(f"<b>{evaln.fit_score:.1f}/10</b>", bold_style),
            Paragraph(exp, body_style),
            Paragraph(cand.location or "N/A", body_style),
            Paragraph(skills_str or "None identified", body_style),
        ])

    table = Table(table_data, colWidths=[35, 120, 65, 65, 85, 170])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1A202C")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(table)
    story.append(Spacer(1, 14))

    # Detailed Candidate Breakdowns
    story.append(Paragraph("Detailed Candidate Assessments", h2_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0"), spaceAfter=8))

    for rank, item in enumerate(results, 1):
        cand: Candidate = item["candidate"]
        evaln: CandidateEvaluation = item["evaluation"]
        cand_header = f"#{rank}. {cand.name} — Fit Score: {evaln.fit_score:.1f}/10 (Evaluator: {evaln.evaluator})"
        story.append(Paragraph(f"<b>{cand_header}</b>", bold_style))
        story.append(Paragraph(evaln.summary, body_style))
        story.append(Spacer(1, 4))

        if evaln.strengths:
            story.append(Paragraph("<b>Strengths:</b> " + "; ".join(evaln.strengths), body_style))
        if evaln.gaps:
            story.append(Paragraph("<b>Identified Gaps:</b> " + "; ".join(evaln.gaps), body_style))
        if evaln.risks:
            story.append(Paragraph("<b>Risks / Missing Info:</b> " + "; ".join(evaln.risks), body_style))

        if evaln.evidence:
            story.append(Spacer(1, 2))
            story.append(Paragraph("<b>Resume Evidence:</b>", bold_style))
            for quote in evaln.evidence[:3]:
                story.append(Paragraph(f'"{quote}"', quote_style))

        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#EDF2F7"), spaceAfter=8))

    doc.build(story)
    return buffer.getvalue()
