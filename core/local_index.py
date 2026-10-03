"""SQLite cache for parsed local resumes and their retrieval chunks."""

import json
import hashlib
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from config import HYBRID_INDEX_DIR
from core.models import Candidate

CACHE_VERSION = 2


class LocalResumeIndex:
    """Cache parsed resume data while checking source content on each refresh."""

    def __init__(self, database: str | Path | None = None) -> None:
        self.database = Path(database) if database else HYBRID_INDEX_DIR / "resume_cache.sqlite3"
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS resumes ("
                "path TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, document_json TEXT NOT NULL)"
            )

    @staticmethod
    def fingerprint(content: bytes) -> str:
        """Include the parser/cache version so code changes can invalidate old parses."""
        versioned_content = f"hireflow-cache-v{CACHE_VERSION}\0".encode("ascii") + content
        return hashlib.sha256(versioned_content).hexdigest()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get(self, path: str | Path, fingerprint: str) -> dict[str, Any] | None:
        key = str(Path(path).resolve())
        with self._connect() as connection:
            row = connection.execute(
                "SELECT fingerprint, document_json FROM resumes WHERE path=?", (key,)
            ).fetchone()
        if row is None or row[0] != fingerprint:
            return None
        document = json.loads(row[1])
        if "tokenized_chunks" not in document:
            return None
        document["candidate"] = Candidate.model_validate(
            {**document["candidate"], "text": document["text"]}
        )
        return document

    def put(
        self, path: str | Path, fingerprint: str, document: dict[str, Any]
    ) -> None:
        key = str(Path(path).resolve())
        serializable = {
            **document,
            "path": key,
            "candidate": document["candidate"].model_dump(exclude={"text"}),
        }
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO resumes(path, fingerprint, document_json) VALUES (?, ?, ?) "
                "ON CONFLICT(path) DO UPDATE SET fingerprint=excluded.fingerprint, "
                "document_json=excluded.document_json",
                (key, fingerprint, json.dumps(serializable, ensure_ascii=True)),
            )

    def prune_missing(self, directory: str | Path, active_paths: set[str]) -> None:
        root = Path(directory).resolve()
        with self._connect() as connection:
            rows = connection.execute("SELECT path FROM resumes").fetchall()
            stale_paths = [
                row[0]
                for row in rows
                if Path(row[0]).is_relative_to(root) and row[0] not in active_paths
            ]
            connection.executemany("DELETE FROM resumes WHERE path=?", [(path,) for path in stale_paths])