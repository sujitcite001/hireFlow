"""Local interaction history and lightweight query-history retrieval."""

import json
import hashlib
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config import HISTORY_RETENTION_DAYS, MEMORY_DIR, STORE_QUERY_TERMS
from utils import tokenize

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
PHONE_PATTERN = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")


def _prune_expired(history: Path, now: datetime) -> None:
    """Atomically remove expired or malformed records from the local log."""
    if not history.exists():
        return
    cutoff = now - timedelta(days=HISTORY_RETENTION_DAYS)
    retained = []
    for line in history.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            timestamp = datetime.fromisoformat(str(record["timestamp"]).replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        if timestamp >= cutoff:
            legacy_query = record.pop("query", None)
            if legacy_query is not None:
                legacy_query = str(legacy_query)
                record["query_hash"] = hashlib.sha256(legacy_query.encode("utf-8")).hexdigest()
                if STORE_QUERY_TERMS:
                    redacted_query = EMAIL_PATTERN.sub(" ", legacy_query)
                    redacted_query = PHONE_PATTERN.sub(" ", redacted_query)
                    record["query_terms"] = sorted(set(tokenize(redacted_query)))[:128]
            legacy_selection = record.pop("selected_resume", None)
            if legacy_selection:
                record["selected_resume_hash"] = hashlib.sha256(
                    str(legacy_selection).encode("utf-8")
                ).hexdigest()
            retained.append(json.dumps(record, ensure_ascii=True))

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=history.parent, delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write("\n".join(retained) + ("\n" if retained else ""))
        os.replace(temporary_path, history)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


def record_interaction(
    query: str,
    selected_resume: str | None = None,
    path: str | Path | None = None,
) -> None:
    """Append a privacy-minimized event and enforce the configured retention window."""
    history = Path(path) if path else MEMORY_DIR / "interactions.jsonl"
    history.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    _prune_expired(history, now)
    record = {
        "timestamp": now.isoformat(),
        "query_hash": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "selected_resume_hash": (
            hashlib.sha256(selected_resume.encode("utf-8")).hexdigest()
            if selected_resume
            else None
        ),
    }
    if STORE_QUERY_TERMS:
        redacted_query = EMAIL_PATTERN.sub(" ", query)
        redacted_query = PHONE_PATTERN.sub(" ", redacted_query)
        record["query_terms"] = sorted(set(tokenize(redacted_query)))[:128]
    with history.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=True) + "\n")


def retrieve_related(query: str, limit: int = 5, path: str | Path | None = None) -> list[dict[str, Any]]:
    """Return prior interactions ranked by shared query terms."""
    history = Path(path) if path else MEMORY_DIR / "interactions.jsonl"
    query_terms = set(tokenize(query))
    if not history.exists() or not query_terms or limit <= 0:
        return []
    _prune_expired(history, datetime.now(timezone.utc))
    matches = []
    for line in history.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        prior_terms = item.get("query_terms") or []
        shared = query_terms & {str(term).casefold() for term in prior_terms}
        if shared:
            matches.append((len(shared) / len(query_terms), item))
    matches.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in matches[:limit]]
