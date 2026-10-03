"""Create the beginner-friendly HireFlow code-flow PDF in this folder."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


OUTPUT_FILE = Path(__file__).resolve().parent.parent / "ProjectFolder" / "HireFlow_Code_Flow.pdf"

styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="GuideTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=29,
        textColor=colors.HexColor("#17324D"),
        alignment=TA_CENTER,
        spaceAfter=8,
    )
)
styles.add(
    ParagraphStyle(
        name="SectionTitle",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#176B67"),
        spaceBefore=10,
        spaceAfter=5,
    )
)
styles.add(
    ParagraphStyle(
        name="FlowBox",
        parent=styles["BodyText"],
        alignment=TA_CENTER,
        leading=17,
    )
)
styles["BodyText"].leading = 15
styles["BodyText"].spaceAfter = 6


def paragraph(text: str, style: str = "BodyText") -> Paragraph:
    return Paragraph(text, styles[style])


def section(title: str, body: str) -> list[object]:
    return [paragraph(title, "SectionTitle"), paragraph(body)]


def make_flow_table() -> Table:
    steps = [
        "1. Recruiter pastes a job description",
        "2. Streamlit starts RecruitmentService",
        "3. LangGraph parses the job description",
        "4. LangGraph searches resumes",
        "5. LangGraph applies selected filters",
        "6. LangGraph evaluates shortlisted candidates",
        "7. LangGraph prepares results and saves the search",
        "8. Streamlit shows ranked candidate explanations",
    ]
    rows = []
    for index, step in enumerate(steps):
        rows.append([paragraph(step, "FlowBox")])
        if index < len(steps) - 1:
            rows.append([paragraph("v", "FlowBox")])
    table = Table(rows, colWidths=[150 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF3F2")),
                ("BOX", (0, 0), (0, -1), 0.5, colors.HexColor("#B7D2CF")),
                ("INNERGRID", (0, 0), (0, -1), 0.4, colors.white),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def add_footer(canvas: object, document: object) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#52616B"))
    canvas.drawString(18 * mm, 12 * mm, "HireFlow - Beginner Code Flow Guide")
    canvas.drawRightString(192 * mm, 12 * mm, f"Page {document.page}")
    canvas.restoreState()


def build_pdf() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT_FILE),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title="HireFlow Beginner Code Flow Guide",
    )
    story = [
        paragraph("HireFlow Code Flow", "GuideTitle"),
        paragraph("A step-by-step guide to how a job description becomes a ranked candidate list.", "BodyText"),
        Spacer(1, 5 * mm),
        make_flow_table(),
        Spacer(1, 6 * mm),
        paragraph("The short version", "SectionTitle"),
        paragraph(
            "HireFlow reads resume files saved on this computer. When a recruiter searches, "
            "the program looks for matching words, applies only the filters the recruiter selected, "
            "and explains each result using information found in the resume."
        ),
        paragraph(
            "By default, search works without an internet connection or an API key. "
            "Gemini is optional and only used after the recruiter turns it on."
        ),
        PageBreak(),
        paragraph("Step by Step", "GuideTitle"),
        *section(
            "1. Start the application",
            "Run <b>streamlit run streamlit/app.py</b>. The app creates the local data folders "
            "and shows the resume library and job-description form."
        ),
        *section(
            "2. Add resumes",
            "In the Resume library section, choose PDF, TXT, or Markdown files and click "
            "Add selected resumes. <b>core/ingestion.py</b> saves them in <b>data/resumes/</b>. "
            "It changes unsafe filenames and avoids overwriting a file with the same name."
        ),
        *section(
            "3. Read each resume",
            "<b>core/parsing.py</b> extracts PDF or text content, cleans extra spaces, and looks "
            "for common skills, email, phone, location, and years of experience. "
            "<b>core/models.py</b> stores these details in a Candidate record. Long text is split "
            "into overlapping pieces so nearby details are not separated."
        ),
        *section(
            "4. Read the job description",
            "The first non-empty line is used as the job title. HireFlow looks for common skill "
            "names, a written experience requirement, and a line such as Location: Toronto. "
            "This is simple text matching, so extracted details should be checked."
        ),
        PageBreak(),
        paragraph("Search and Results", "GuideTitle"),
        *section(
            "5. Find resumes",
            "<b>core/hybrid_indexer.py</b> compares words in the job description with words in "
            "each resume. Rare words matter more than very common words. The result is a search "
            "score between 0 and 1. If Gemini is enabled, the program can also compare text "
            "embeddings and combine the scores."
        ),
        *section(
            "6. Apply filters",
            "<b>core/filters.py</b> applies the selected must-have skills, location, and minimum "
            "experience. Leave a filter empty to keep candidates who do not meet that condition "
            "in the ranked list."
        ),
        *section(
            "7. Score and explain",
            "<b>core/re_ranker.py</b> compares candidate skills and experience with the job. "
            "It returns a score from 0 to 10, strengths, gaps, risks, a short summary, and exact "
            "lines from the resume as evidence. If Gemini is unavailable, these simple rules are used."
        ),
        *section(
            "8. Show results and save activity",
            "The Streamlit page shows the ranked candidates. If no results appear, it explains "
            "whether there are no resumes, no matches, or filters removed all candidates. "
            "Searches and review clicks are saved under <b>data/memory/</b>."
        ),
        PageBreak(),
        paragraph("Files and Key Ideas", "GuideTitle"),
        paragraph("Where to look", "SectionTitle"),
        paragraph(
            "<b>streamlit/app.py</b> - screen and button actions<br/>"
            "<b>core/recruitment.py</b> - connects the steps in order<br/>"
            "<b>core/ingestion.py</b> - reads and saves resumes<br/>"
            "<b>core/parsing.py</b> - extracts text and simple fields<br/>"
            "<b>core/hybrid_indexer.py</b> - finds and ranks likely resumes<br/>"
            "<b>core/filters.py</b> - applies recruiter-selected filters<br/>"
            "<b>core/re_ranker.py</b> - creates candidate explanations<br/>"
            "<b>core/models.py</b> - defines Candidate, JobDescription, and CandidateEvaluation<br/>"
            "<b>core/memory_rag.py</b> - stores search and review history"
        ),
        paragraph("Fallbacks and limits", "SectionTitle"),
        paragraph(
            "A damaged resume is skipped and reported; the other resumes are still searched. "
            "A scanned PDF needs OCR, which is not included. SQLite stores optional vectors locally; "
            "this starter project does not connect to Pinecone. Evaluation metrics are simple "
            "word-overlap estimates, not the full RAGAS framework."
        ),
        paragraph("Run checks", "SectionTitle"),
        paragraph(
            "Run <b>python -m unittest discover -s tests -v</b> to run tests, or "
            "<b>python -m compileall -q .</b> to check Python syntax."
        ),
    ]
    document.build(story, onFirstPage=add_footer, onLaterPages=add_footer)


if __name__ == "__main__":
    build_pdf()
    print(f"Created: {OUTPUT_FILE}")