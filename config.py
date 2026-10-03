"""Central paths and environment-backed settings for HireFlow."""

from pathlib import Path
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = Path(os.getenv("HIREFLOW_DATA_DIR", PROJECT_ROOT / "data")).expanduser()
RESUMES_DIR = DATA_DIR / "resumes"
HYBRID_INDEX_DIR = DATA_DIR / "hybrid_index"
MEMORY_DIR = DATA_DIR / "memory"
MAX_RESULTS = int(os.getenv("HIREFLOW_MAX_RESULTS", "20"))
CHUNK_SIZE = int(os.getenv("HIREFLOW_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("HIREFLOW_CHUNK_OVERLAP", "200"))
BM25_WEIGHT = float(os.getenv("HIREFLOW_BM25_WEIGHT", "0.6"))
VECTOR_WEIGHT = float(os.getenv("HIREFLOW_VECTOR_WEIGHT", "0.4"))
def _get_secret(key: str, default: str = "") -> str:
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


GEMINI_API_KEY = _get_secret("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
EMBEDDING_DIMENSIONS = int(os.getenv("HIREFLOW_EMBEDDING_DIMENSIONS", "768"))
HISTORY_RETENTION_DAYS = max(1, int(os.getenv("HIREFLOW_HISTORY_RETENTION_DAYS", "30")))
STORE_QUERY_TERMS = os.getenv("HIREFLOW_STORE_QUERY_TERMS", "false").casefold() in {
    "1",
    "true",
    "yes",
}


def ensure_data_directories() -> None:
    """Create the local data directories when the app starts."""
    for directory in (RESUMES_DIR, HYBRID_INDEX_DIR, MEMORY_DIR):
        directory.mkdir(parents=True, exist_ok=True)
