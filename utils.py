"""Small shared helpers."""

import re
from pathlib import Path


def tokenize(text: str) -> list[str]:
    """Normalize text into searchable alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.casefold())


def safe_filename(name: str) -> str:
    """Return a basename safe to store inside the resumes directory."""
    basename = Path(name).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("._")
    return cleaned or "resume"
