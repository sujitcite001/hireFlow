"""Offline retrieval and simple online RAG quality metrics."""

from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

from utils import tokenize


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(item in relevant_ids for item in ranked_ids[:k]) / k


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids or k <= 0:
        return 0.0
    return sum(item in relevant_ids for item in ranked_ids[:k]) / len(relevant_ids)


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: set[str]) -> float:
    return next((1.0 / rank for rank, item in enumerate(ranked_ids, 1) if item in relevant_ids), 0.0)


def evaluate_search_labels(
    labels_file: str | Path,
    directory: str | Path,
    k: int = 10,
    use_semantic: bool = False,
) -> dict[str, float | int]:
    """Evaluate configured search ranking against JSONL query/relevance labels."""
    if k <= 0:
        raise ValueError("k must be positive")
    from core.hybrid_indexer import search_resumes

    label_path = Path(labels_file)
    metric_totals = {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0}
    query_count = 0
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            query = item["query"]
            relevant_ids = item["relevant_candidate_ids"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"Invalid label record on line {line_number}: {error}") from error
        if not isinstance(query, str) or not isinstance(relevant_ids, list):
            raise ValueError(
                f"Invalid label record on line {line_number}: expected query text and an ID list"
            )

        results = search_resumes(
            query,
            directory=directory,
            limit=k,
            use_semantic=use_semantic,
        )
        ranked_ids = [item["candidate"].candidate_id for item in results]
        relevant = {str(candidate_id) for candidate_id in relevant_ids}
        metric_totals["precision_at_k"] += precision_at_k(ranked_ids, relevant, k)
        metric_totals["recall_at_k"] += recall_at_k(ranked_ids, relevant, k)
        metric_totals["mrr"] += reciprocal_rank(ranked_ids, relevant)
        query_count += 1

    if not query_count:
        raise ValueError("The labels file contains no evaluation queries")
    return {
        "queries": query_count,
        **{name: total / query_count for name, total in metric_totals.items()},
    }


def context_recall(retrieved_context: str, expected_context: str) -> float:
    """Measure how much of the expected context appears in retrieved text."""
    expected = set(tokenize(expected_context))
    if not expected:
        return 0.0
    return len(expected & set(tokenize(retrieved_context))) / len(expected)


def context_precision(retrieved_chunks: Sequence[str], relevant_chunks: Sequence[str]) -> float:
    """Measure the share of retrieved chunks that overlap relevant context."""
    if not retrieved_chunks:
        return 0.0
    relevant_tokens = set(tokenize(" ".join(relevant_chunks)))
    if not relevant_tokens:
        return 0.0
    relevant_count = sum(
        bool(set(tokenize(chunk)) & relevant_tokens) for chunk in retrieved_chunks
    )
    return relevant_count / len(retrieved_chunks)


def answer_correctness(answer: str, expected_answer: str) -> float:
    """Use token F1 as a small, dependency-free offline correctness proxy."""
    answer_tokens = set(tokenize(answer))
    expected_tokens = set(tokenize(expected_answer))
    if not answer_tokens or not expected_tokens:
        return 0.0
    shared = len(answer_tokens & expected_tokens)
    precision = shared / len(answer_tokens)
    recall = shared / len(expected_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def faithfulness(answer: str, evidence: Sequence[str]) -> float:
    """Estimate whether answer words are supported by the supplied evidence."""
    answer_tokens = set(tokenize(answer))
    evidence_tokens = set(tokenize(" ".join(evidence)))
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & evidence_tokens) / len(answer_tokens)


def answer_relevancy(answer: str, query: str) -> float:
    """Estimate whether an answer uses the recruiter's requested concepts."""
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    return len(query_tokens & set(tokenize(answer))) / len(query_tokens)


def evaluate_rag(
    retrieved_chunks: Sequence[str],
    expected_context: str,
    answer: str,
    expected_answer: str,
    query: str,
) -> dict[str, float]:
    """Return offline and online evaluation estimates for one example."""
    return {
        "context_recall": context_recall(" ".join(retrieved_chunks), expected_context),
        "context_precision": context_precision(retrieved_chunks, [expected_context]),
        "answer_correctness": answer_correctness(answer, expected_answer),
        "faithfulness": faithfulness(answer, retrieved_chunks),
        "answer_relevancy": answer_relevancy(answer, query),
    }
