"""Lexical search with optional blending of externally generated vectors."""

import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Any

from config import BM25_WEIGHT, MAX_RESULTS, RESUMES_DIR, VECTOR_WEIGHT
from core.gemini_client import embed_text, embed_texts
from core.ingestion import ingest_resumes
from core.local_index import LocalResumeIndex
from core.models import Candidate
from core.re_ranker import rerank
from core.vector_store import VectorStore
from utils import tokenize


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(sum(value * value for value in right))
    return sum(a * b for a, b in zip(left, right)) / denominator if denominator else 0.0


def _bm25_scores(query_terms: list[str], documents: list[dict[str, Any]]) -> list[float]:
    """Calculate simple BM25 keyword scores for the current resume set."""
    if not query_terms or not documents:
        return [0.0] * len(documents)

    resume_words = [
        document.get("tokens") or tokenize(str(document["text"]))
        for document in documents
    ]
    number_of_resumes = len(resume_words)
    average_resume_length = sum(len(words) for words in resume_words) / number_of_resumes or 1
    resumes_with_word = Counter(word for words in resume_words for word in set(words))

    scores = []
    for words in resume_words:
        word_counts = Counter(words)
        score = 0.0
        for query_word in set(query_terms):
            word_count = word_counts[query_word]
            if not word_count:
                continue

            resumes_with_this_word = resumes_with_word[query_word]
            inverse_document_frequency = math.log(
                1 + (number_of_resumes - resumes_with_this_word + 0.5)
                / (resumes_with_this_word + 0.5)
            )
            length_adjustment = 1.5 * (
                1 - 0.75 + 0.75 * len(words) / average_resume_length
            )
            score += inverse_document_frequency * (word_count * 2.5) / (word_count + length_adjustment)
        scores.append(score)

    maximum = max(scores, default=0.0)
    return [score / maximum if maximum else 0.0 for score in scores]


def search_resumes(
    query: str,
    directory: str | Path = RESUMES_DIR,
    limit: int = MAX_RESULTS,
    vector_store: VectorStore | None = None,
    query_vector: list[float] | None = None,
    use_semantic: bool = False,
    api_key: str | None = None,
    ingestion_errors: list[str] | None = None,
    resume_index: LocalResumeIndex | None = None,
    refresh_index: bool = False,
) -> list[dict[str, Any]]:
    """Search resume chunks with BM25 and optional Gemini vectors with batch embeddings."""
    query_terms = tokenize(query)
    if not query_terms or limit <= 0:
        return []
    if BM25_WEIGHT < 0 or VECTOR_WEIGHT < 0 or BM25_WEIGHT + VECTOR_WEIGHT <= 0:
        raise ValueError("BM25 and vector weights must be non-negative with a positive total")
    weight_total = BM25_WEIGHT + VECTOR_WEIGHT
    bm25_weight = BM25_WEIGHT / weight_total
    vector_weight = VECTOR_WEIGHT / weight_total
    root = Path(directory).resolve()
    active_resume_index = resume_index or LocalResumeIndex()
    documents = ingest_resumes(
        root,
        errors=ingestion_errors,
        index=active_resume_index,
        force_refresh=refresh_index,
    )
    chunks = [
        (document_index, chunk_index, str(chunk))
        for document_index, document in enumerate(documents)
        for chunk_index, chunk in enumerate(document.get("chunks") or [document["text"]])
    ]
    chunk_documents = [
        {
            "text": chunk,
            "tokens": documents[document_index]["tokenized_chunks"][chunk_index],
        }
        for document_index, chunk_index, chunk in chunks
    ]
    lexical_scores = _bm25_scores(query_terms, chunk_documents)
    active_vector_store = vector_store or VectorStore()
    if use_semantic and query_vector is None:
        query_vector = embed_text(query, api_key=api_key)

    root_key = hashlib.sha256(str(root).encode("utf-8")).hexdigest()
    source_keys = {
        index: hashlib.sha256(str(Path(document["path"]).resolve()).encode("utf-8")).hexdigest()
        for index, document in enumerate(documents)
    }
    valid_vector_hashes = {
        f"{source_keys[document_index]}:{chunk_index}": hashlib.sha256(
            chunk.encode("utf-8")
        ).hexdigest()
        for document_index, chunk_index, chunk in chunks
    }

    # Only prune stale vectors on explicit refresh or first indexing
    if refresh_index:
        active_vector_store.retain_valid_vectors(root_key, valid_vector_hashes)

    stored_vectors: dict[str, dict[str, Any]] = {}
    if use_semantic and query_vector:
        missing_chunks: list[tuple[str, str, str, str]] = []  # doc_id, chunk_hash, source_key, chunk
        for document_index, chunk_index, chunk in chunks:
            source_key = source_keys[document_index]
            document_id = f"{source_key}:{chunk_index}"
            chunk_hash = valid_vector_hashes[document_id]
            cached_entry = active_vector_store.get(document_id)
            if cached_entry:
                stored_vectors[document_id] = cached_entry
            else:
                missing_chunks.append((document_id, chunk_hash, source_key, chunk))

        # Batch embed missing chunks to eliminate sequential HTTP roundtrips
        if missing_chunks:
            batch_size = 50
            for i in range(0, len(missing_chunks), batch_size):
                batch = missing_chunks[i : i + batch_size]
                batch_texts = [item[3] for item in batch]
                batch_vectors = embed_texts(batch_texts, api_key=api_key)
                if not batch_vectors:
                    batch_vectors = [embed_text(txt, api_key=api_key) for txt in batch_texts]
                if batch_vectors:
                    upsert_items = []
                    for (doc_id, c_hash, s_key, _), vec in zip(batch, batch_vectors):
                        if vec and len(vec) == active_vector_store.dimensions:
                            metadata = {
                                "text_hash": c_hash,
                                "root_key": root_key,
                                "source_key": s_key,
                            }
                            upsert_items.append((doc_id, vec, metadata))
                            stored_vectors[doc_id] = {"vector": vec, "metadata": metadata}
                    if upsert_items:
                        active_vector_store.upsert_batch(upsert_items)

    best_by_document: dict[int, dict[str, Any]] = {}
    for chunk_position, (document_index, chunk_index, chunk) in enumerate(chunks):
        document = documents[document_index]
        candidate = document["candidate"]
        if not isinstance(candidate, Candidate):
            continue
        source_key = source_keys[document_index]
        document_id = f"{source_key}:{chunk_index}"
        lexical_score = lexical_scores[chunk_position]
        stored_vector = stored_vectors.get(document_id) if use_semantic and query_vector else None

        vector_score = 0.0
        if query_vector and stored_vector:
            vector_score = max(0.0, cosine_similarity(query_vector, stored_vector["vector"]))
        if use_semantic and query_vector and stored_vector:
            score = bm25_weight * lexical_score + vector_weight * vector_score
            retrieval_mode = "hybrid"
        else:
            score = lexical_score
            retrieval_mode = "BM25"
        if score <= 0:
            continue
        result = {
            "name": str(document["name"]),
            "path": str(document["path"]),
            "text": str(document["text"]),
            "candidate": candidate,
            "score": score,
            "bm25_score": lexical_score,
            "vector_score": vector_score,
            "retrieval_mode": retrieval_mode,
            "chunk_index": chunk_index,
            "matched_chunk": chunk,
            "snippet": " ".join(chunk.split())[:320],
        }
        previous = best_by_document.get(document_index)
        if previous is None or result["score"] > previous["score"]:
            best_by_document[document_index] = result
    return [dict(item) for item in rerank(list(best_by_document.values()))[:limit]]
