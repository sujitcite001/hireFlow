"""Resume file discovery, storage, and parsing."""

from pathlib import Path
from typing import Any

from config import CHUNK_OVERLAP, CHUNK_SIZE, RESUMES_DIR
from core.local_index import LocalResumeIndex
from core.models import Candidate
from core.parsing import extract_text, parse_candidate_text, split_into_chunks
from utils import safe_filename, tokenize

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}


def list_resume_files(directory: str | Path = RESUMES_DIR) -> list[Path]:
    """Return supported resume files in stable name order."""
    root = Path(directory)
    if not root.exists():
        return []
    return sorted(
        (path for path in root.iterdir() if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES),
        key=lambda path: path.name.casefold(),
    )


def save_resume(name: str, content: bytes, directory: str | Path = RESUMES_DIR) -> Path:
    """Store uploaded content without allowing path traversal or overwrites."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    filename = safe_filename(name)
    destination = root / filename
    stem, suffix = destination.stem, destination.suffix
    counter = 1
    while destination.exists():
        destination = root / f"{stem}_{counter}{suffix}"
        counter += 1
    destination.write_bytes(content)
    return destination


def ingest_resumes(
    directory: str | Path = RESUMES_DIR,
    errors: list[str] | None = None,
    index: LocalResumeIndex | None = None,
    force_refresh: bool = False,
) -> list[dict[str, object]]:
    """Load cached resumes or parse changed files, continuing after per-file errors."""
    documents = []
    paths = list_resume_files(directory)
    active_paths = {str(path.resolve()) for path in paths}
    if index:
        index.prune_missing(directory, active_paths)
    for path in paths:
        try:
            fingerprint = LocalResumeIndex.fingerprint(path.read_bytes())
            if index and not force_refresh:
                cached = index.get(path, fingerprint)
                if cached is not None:
                    documents.append(cached)
                    continue
            text = extract_text(path)
            candidate = parse_candidate_text(text, path.name, path.stem)
            chunks = split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
            document: dict[str, Any] = {
                "name": path.name,
                "path": str(path.resolve()),
                "text": text,
                "candidate": candidate,
                "chunks": chunks,
                "tokenized_chunks": [tokenize(chunk) for chunk in chunks],
            }
            if index:
                index.put(path, fingerprint, document)
            documents.append(document)
        except Exception as error:
            if errors is not None:
                errors.append(f"{path.name}: {error}")
            continue
    return documents
