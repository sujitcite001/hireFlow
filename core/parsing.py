"""Text extraction and normalization for supported resume formats."""

from datetime import datetime
from pathlib import Path
import re

from core.models import Candidate, JobDescription

KNOWN_SKILLS = (
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "c", "go", "golang",
    "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "bash", "shell", "sql", "nosql",
    # Frontend
    "html", "css", "sass", "react", "next.js", "vue", "angular", "svelte", "tailwind",
    "bootstrap", "redux", "node.js",
    # Backend & Architecture
    "fastapi", "django", "flask", "express", "spring boot", "graphql", "grpc",
    "rest api", "microservices", "system design",
    # Data & ML / AI
    "pandas", "numpy", "scikit-learn", "pytorch", "tensorflow", "keras", "xgboost",
    "lightgbm", "machine learning", "deep learning", "natural language processing",
    "nlp", "data science", "data engineering", "data analysis", "llm", "genai", "rag",
    "langchain", "langgraph", "llamaindex", "hugging face", "opencv",
    # Cloud & DevOps
    "aws", "azure", "google cloud", "gcp", "docker", "kubernetes", "k8s", "helm",
    "terraform", "git", "linux", "ci/cd", "github actions", "gitlab", "jenkins",
    "ansible", "prometheus", "grafana",
    # Big Data & Databases
    "spark", "hadoop", "airflow", "dbt", "tableau", "power bi", "excel", "postgresql",
    "postgres", "mysql", "mongodb", "redis", "kafka", "rabbitmq", "elasticsearch",
    "snowflake", "bigquery", "databricks", "redshift", "cassandra", "dynamodb", "sqlite",
    "vector databases", "pinecone", "chromadb", "faiss", "milvus",
    # Management & Practices
    "communication", "project management", "agile", "scrum", "leadership",
    "salesforce", "healthcare", "finance", "pytest", "unit testing", "selenium",
)

SKILL_LABEL = re.compile(
    r"^\s*(?P<label>(?:(?:required|preferred|optional|nice[- ]to[- ]have)\s+)?"
    r"(?:technical\s+)?(?:skills?|technologies|tools|frameworks|platforms|"
    r"programming languages|tech(?:nology)? stack|core competencies|expertise)|"
    r"required|preferred|nice[- ]to[- ]have)\s*:?\s*(?P<values>.*)$",
    flags=re.IGNORECASE,
)

HEADER_BLACKLIST = {
    "resume", "curriculum vitae", "cv", "summary", "profile", "contact",
    "contact information", "experience", "education", "skills", "about me",
    "professional summary", "career objective", "objective", "work experience"
}


def extract_text(path: str | Path) -> str:
    """Extract text from PDF, TXT, or Markdown files."""
    file_path = Path(path)
    suffix = file_path.suffix.casefold()
    if suffix == ".pdf":
        from pypdf import PdfReader

        text = "\n".join(page.extract_text() or "" for page in PdfReader(file_path).pages)
    elif suffix in {".txt", ".md", ".markdown"}:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported resume type: {suffix or 'no extension'}")
    return clean_text(text)


def clean_text(text: str) -> str:
    """Normalize extracted text while retaining paragraph boundaries."""
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split long text into overlapping pieces without losing the source text."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk size must be positive and overlap must be smaller than chunk size")
    cleaned = clean_text(text)
    chunks = []
    start = 0
    while start < len(cleaned):
        end = start + chunk_size
        chunks.append(cleaned[start:end])
        if end >= len(cleaned):
            break
        start = end - overlap
    return chunks


def extract_skills(text: str) -> list[str]:
    """Find known skills and arbitrary terms in explicitly labeled skill lists."""
    found: list[str] = []
    for skill in KNOWN_SKILLS:
        pattern = r"(?<![\w])" + re.escape(skill) + r"(?![\w])"
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(skill)
    for line in text.splitlines():
        match = SKILL_LABEL.match(line)
        if not match:
            continue
        values = re.sub(r"\s+(?:and|or)\s+", ",", match.group("values"), flags=re.IGNORECASE)
        for value in re.split(r"[,;|]+", values):
            term = re.sub(r"\([^)]*\)", " ", value).strip(" \t-*:.•")
            term = re.sub(r"\s+", " ", term).casefold()
            if (
                term
                and len(term) <= 60
                and len(term.split()) <= 6
                and term not in found
            ):
                found.append(term)
    return found


def candidate_has_skill(candidate: Candidate, skill: str) -> bool:
    """Match an explicit skill term without requiring a global skill taxonomy."""
    normalized = skill.casefold().strip()
    if not normalized:
        return False
    if normalized in {item.casefold() for item in candidate.skills}:
        return True
    pattern = r"(?<![\w])" + re.escape(normalized) + r"(?![\w])"
    return bool(re.search(pattern, candidate.text, flags=re.IGNORECASE))


def _extract_candidate_name(lines: list[str], fallback_name: str) -> str:
    """Identify candidate name from the top header lines, skipping CV titles."""
    for line in lines[:8]:
        clean = line.strip()
        if not clean:
            continue
        words = clean.split()
        if clean.casefold() in HEADER_BLACKLIST:
            continue
        if re.search(r"@|\d|https?:|www\.|linkedin|github|[|:;=/\\]", clean):
            continue
        if 1 <= len(words) <= 4 and len(clean) <= 40:
            if clean[0].isupper() or clean.isupper():
                return clean.title() if clean.isupper() else clean
    return fallback_name


def _extract_experience_from_dates(text: str) -> float | None:
    """Estimate years of experience from employment date ranges (e.g., 2018 - 2023)."""
    current_year = datetime.now().year
    year_pattern = re.compile(
        r"(?<!\d)(?:(?:19|20)\d{2})\s*(?:-|–|—|to)\s*(?:(?:19|20)\d{2}|present|current|now)(?!\d)",
        flags=re.IGNORECASE,
    )
    matches = year_pattern.findall(text)
    if not matches:
        return None

    intervals: list[tuple[int, int]] = []
    for match in matches:
        parts = re.split(r"[-–—]|to", match, flags=re.IGNORECASE)
        if len(parts) == 2:
            start_m = re.search(r"(19\d{2}|20\d{2})", parts[0])
            end_m = re.search(r"(19\d{2}|20\d{2})", parts[1])
            is_present = bool(re.search(r"present|current|now", parts[1], flags=re.IGNORECASE))
            if start_m:
                start_yr = int(start_m.group(1))
                end_yr = current_year if is_present else (int(end_m.group(1)) if end_m else start_yr)
                if 1970 <= start_yr <= current_year and start_yr <= end_yr <= current_year + 1:
                    intervals.append((start_yr, end_yr))

    if not intervals:
        return None

    intervals.sort()
    merged: list[list[int]] = []
    for start, end in intervals:
        if not merged:
            merged.append([start, end])
        else:
            prev = merged[-1]
            if start <= prev[1]:
                prev[1] = max(prev[1], end)
            else:
                merged.append([start, end])

    total_years = sum(end - start for start, end in merged)
    total_years = max(total_years, 1.0)
    return min(float(total_years), 45.0)


def _extract_location(text: str) -> str | None:
    match = re.search(r"(?:location|based in|address)\s*:\s*([^\n,;]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    top_lines = [line.strip() for line in text.splitlines()[:8] if line.strip()]
    city_state_pattern = re.compile(
        r"\b([A-Z][a-zA-Z\s]{2,20},\s*(?:[A-Z]{2}|[A-Z][a-zA-Z\s]{2,20}))\b"
    )
    for line in top_lines:
        city_match = city_state_pattern.search(line)
        if city_match and not re.search(r"@|https?:", line):
            loc = city_match.group(1).strip()
            if loc.casefold() not in {"curriculum vitae", "united states", "remote work"}:
                return loc
    return None


def parse_candidate_text(text: str, candidate_id: str, fallback_name: str) -> Candidate:
    """Turn resume text into a validated candidate record with enhanced parsing."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    name = _extract_candidate_name(lines, fallback_name)
    email_match = re.search(r"[\w.+-]+@[\w.-]+\.\w+", text)
    phone_match = re.search(r"(?:\+?\d[\d ()-]{7,}\d)", text)
    experience_match = re.search(
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)(?:\s+of)?\s+experience",
        text,
        flags=re.IGNORECASE,
    )
    experience_years = float(experience_match.group(1)) if experience_match else None
    if experience_years is None:
        experience_years = _extract_experience_from_dates(text)

    return Candidate(
        candidate_id=candidate_id,
        name=name,
        email=email_match.group(0) if email_match else None,
        phone=phone_match.group(0).strip() if phone_match else None,
        location=_extract_location(text),
        text=text,
        skills=extract_skills(text),
        experience_years=experience_years,
    )


def parse_job_description(text: str, jd_id: str = "job-description") -> JobDescription:
    """Read a job description and identify common skills and stated constraints."""
    cleaned = clean_text(text)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    title = lines[0][:120] if lines else "Job description"
    experience_match = re.search(
        r"(?:at least\s+|minimum\s+of\s+)?(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)(?:\s+of)?\s+experience",
        cleaned,
        flags=re.IGNORECASE,
    )
    skills = extract_skills(cleaned)
    sentences = re.split(r"[\n.;]+", cleaned)
    optional_skills = []
    required_skills = []
    for skill in skills:
        skill_sentence = ""
        for sentence in sentences:
            if skill.casefold() in sentence.casefold():
                skill_sentence = sentence
                break
        if re.search(r"nice to have|preferred|bonus|optional", skill_sentence, flags=re.IGNORECASE):
            optional_skills.append(skill)
        else:
            required_skills.append(skill)
    return JobDescription(
        jd_id=jd_id,
        title=title,
        text=cleaned,
        required_skills=required_skills,
        optional_skills=optional_skills,
        min_experience=float(experience_match.group(1)) if experience_match else None,
        location=_extract_location(cleaned),
    )
