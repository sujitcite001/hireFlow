"""SQLite storage for optional, caller-generated resume embeddings."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from config import EMBEDDING_DIMENSIONS, HYBRID_INDEX_DIR


class VectorStore:
    """Persist document vectors without coupling HireFlow to an embedding provider."""

    def __init__(
        self,
        database: str | Path | None = None,
        dimensions: int = EMBEDDING_DIMENSIONS,
    ) -> None:
        self.database = Path(database) if database else HYBRID_INDEX_DIR / "vectors.sqlite3"
        self.dimensions = dimensions
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS vectors ("
                "document_id TEXT PRIMARY KEY, vector_json TEXT NOT NULL, metadata_json TEXT NOT NULL)"
            )

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

    def upsert(self, document_id: str, vector: list[float], metadata: dict[str, Any] | None = None) -> None:
        self.upsert_batch([(document_id, vector, metadata)])

    def upsert_batch(self, items: list[tuple[str, list[float], dict[str, Any] | None]]) -> None:
        if not items:
            return
        records = []
        for document_id, vector, metadata in items:
            if len(vector) != self.dimensions:
                raise ValueError(f"Expected a vector with {self.dimensions} values")
            records.append((document_id, json.dumps(vector), json.dumps(metadata or {})))
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO vectors(document_id, vector_json, metadata_json) VALUES (?, ?, ?) "
                "ON CONFLICT(document_id) DO UPDATE SET vector_json=excluded.vector_json, "
                "metadata_json=excluded.metadata_json",
                records,
            )

    def get(self, document_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT vector_json, metadata_json FROM vectors WHERE document_id=?", (document_id,)
            ).fetchone()
        if row is None:
            return None
        vector = json.loads(row[0])
        if len(vector) != self.dimensions:
            return None
        return {"vector": vector, "metadata": json.loads(row[1])}

    def retain_valid_vectors(self, root_key: str, valid_chunks: dict[str, str]) -> None:
        """Remove stale chunk embeddings belonging to one indexed resume directory."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT document_id, metadata_json FROM vectors"
            ).fetchall()
            stale_ids = []
            for document_id, metadata_json in rows:
                metadata = json.loads(metadata_json)
                if metadata.get("text_hash") and not metadata.get("root_key"):
                    stale_ids.append((document_id,))
                    continue
                if metadata.get("root_key") != root_key:
                    continue
                if valid_chunks.get(document_id) != metadata.get("text_hash"):
                    stale_ids.append((document_id,))
            connection.executemany(
                "DELETE FROM vectors WHERE document_id=?", stale_ids
            )
