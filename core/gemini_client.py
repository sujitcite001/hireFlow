"""Optional Gemini features built with LangChain."""

from functools import lru_cache
import os
import re

from config import EMBEDDING_DIMENSIONS, GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL, GEMINI_MODEL
from core.models import Candidate, CandidateEvaluation, JobDescription


def get_effective_api_key(api_key: str | None = None) -> str:
    """Return an explicitly provided key or the environment/config/secrets key."""
    if api_key and api_key.strip():
        return api_key.strip()
    if GEMINI_API_KEY and GEMINI_API_KEY.strip():
        return GEMINI_API_KEY.strip()
    env_key = os.getenv("GEMINI_API_KEY", "")
    if env_key.strip():
        return env_key.strip()
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass
    return ""


@lru_cache(maxsize=4)
def _get_embeddings(api_key: str) -> "GoogleGenerativeAIEmbeddings":
    """Create the embedding client for a given API key and reuse it."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBEDDING_MODEL,
        google_api_key=api_key,
        output_dimensionality=EMBEDDING_DIMENSIONS,
    )


@lru_cache(maxsize=4)
def _get_evaluation_chain(api_key: str):
    """Build a prompt and a model that returns a checked CandidateEvaluation."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Compare a candidate resume with a job description. Use only facts in the text. "
                "Do not guess missing details or use protected characteristics. Evidence must be "
                "exact quotes copied from the resume. Return a fit score from 0 to 10.",
            ),
            (
                "human",
                "Candidate ID: {candidate_id}\n"
                "Resume:\n{resume_text}\n\n"
                "Job title: {job_title}\n"
                "Job description:\n{job_text}",
            ),
        ]
    )
    model = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0,
        max_retries=1,
    )
    return prompt | model.with_structured_output(CandidateEvaluation)


def embed_text(text: str, api_key: str | None = None) -> list[float] | None:
    """Return an embedding, or None if Gemini is off or the request fails."""
    key = get_effective_api_key(api_key)
    if not key or not text.strip():
        return None
    try:
        return _get_embeddings(key).embed_query(text)
    except Exception:
        return None


def embed_texts(texts: list[str], api_key: str | None = None) -> list[list[float]] | None:
    """Batch embed multiple texts in a single request."""
    key = get_effective_api_key(api_key)
    if not key or not texts:
        return None
    try:
        return _get_embeddings(key).embed_documents(texts)
    except Exception:
        return None


def _normalize_text_for_comparison(text: str) -> str:
    """Strip punctuation and normalize whitespace for quote verification."""
    cleaned = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", cleaned).strip().casefold()


def _is_quote_supported(quote: str, document_text: str) -> bool:
    """Check if an evidence quote is genuinely grounded in the resume."""
    clean_quote = _normalize_text_for_comparison(quote)
    if not clean_quote:
        return False
    clean_doc = _normalize_text_for_comparison(document_text)
    if clean_quote in clean_doc:
        return True

    quote_tokens = clean_quote.split()
    if len(quote_tokens) >= 4:
        prefix = " ".join(quote_tokens[:min(5, len(quote_tokens))])
        if prefix in clean_doc:
            return True
        doc_tokens = set(clean_doc.split())
        overlap = len(set(quote_tokens) & doc_tokens)
        if overlap / len(set(quote_tokens)) >= 0.8:
            return True
    return False


def evaluate_with_gemini(
    candidate: Candidate, job: JobDescription, api_key: str | None = None
) -> CandidateEvaluation | None:
    """Ask Gemini for a structured evaluation and verify every evidence quote."""
    key = get_effective_api_key(api_key)
    if not key:
        return None

    try:
        chain = _get_evaluation_chain(key)
        evaluation = chain.invoke(
            {
                "candidate_id": candidate.candidate_id,
                "resume_text": candidate.text[:12000],
                "job_title": job.title,
                "job_text": job.text[:6000],
            }
        )
        if not isinstance(evaluation, CandidateEvaluation):
            evaluation = CandidateEvaluation.model_validate(evaluation)
        if evaluation.candidate_id != candidate.candidate_id:
            return None

        supported_evidence = [
            quote.strip()
            for quote in evaluation.evidence
            if _is_quote_supported(quote, candidate.text)
        ]
        if evaluation.evidence and len(supported_evidence) == 0:
            return None

        evaluation.evidence = supported_evidence
        evaluation.evaluator = "gemini"
        return evaluation
    except Exception:
        return None


def test_gemini_connection(api_key: str | None = None) -> tuple[bool, str]:
    """Verify whether the configured Gemini API key and model work."""
    key = get_effective_api_key(api_key)
    if not key:
        return False, "No Gemini API key provided."
    try:
        vec = embed_text("HireFlow connection test", api_key=key)
        if vec and len(vec) == EMBEDDING_DIMENSIONS:
            return True, "Gemini connection successful."
        return False, f"Embedding returned unexpected format or dimension (expected {EMBEDDING_DIMENSIONS})."
    except Exception as exc:
        return False, f"Connection test failed: {exc}"
