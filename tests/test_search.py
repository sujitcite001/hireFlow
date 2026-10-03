"""Focused tests for local search and evaluation helpers."""

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from core.evaluator import evaluate_search_labels, precision_at_k, recall_at_k, reciprocal_rank
from core.hybrid_indexer import search_resumes
from core.local_index import LocalResumeIndex
from core.parsing import split_into_chunks
from core.vector_store import VectorStore


class SearchTests(unittest.TestCase):
    def test_search_ranks_matching_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "engineer.txt").write_text(
                "Python engineer with data engineering and SQL experience.", encoding="utf-8"
            )
            (root / "designer.txt").write_text(
                "Product designer focused on research and visual systems.", encoding="utf-8"
            )

            results = search_resumes("Python data engineer", root)

        self.assertEqual([result["name"] for result in results], ["engineer.txt"])
        self.assertGreater(results[0]["score"], 0)

    def test_chunk_search_returns_the_matching_passage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            resume_text = "Unrelated background. " * 100
            resume_text += "Built production systems with Kubernetes and Helm."
            (root / "platform.txt").write_text(resume_text, encoding="utf-8")

            results = search_resumes("Kubernetes Helm", root)

        self.assertEqual(len(results), 1)
        self.assertIn("Kubernetes", results[0]["matched_chunk"])
        self.assertGreater(results[0]["chunk_index"], 0)

    def test_semantic_vectors_are_cached_per_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            resume_text = "General project work and team delivery.\n" * 100
            resume_text += "Specialist Kubernetes Helm operations.\n"
            (root / "platform.txt").write_text(resume_text, encoding="utf-8")
            index = LocalResumeIndex(root / "resume_cache.sqlite3")
            vectors = VectorStore(root / "vectors.sqlite3", dimensions=3)

            with patch("core.hybrid_indexer.embed_text", return_value=[1.0, 0.0, 0.0]) as embed:
                results = search_resumes(
                    "Kubernetes Helm",
                    root,
                    vector_store=vectors,
                    query_vector=[1.0, 0.0, 0.0],
                    use_semantic=True,
                    resume_index=index,
                )

            expected_chunks = len(split_into_chunks(resume_text))
            self.assertEqual(embed.call_count, expected_chunks)
            self.assertEqual(len(results), 1)
            self.assertIn("Kubernetes", results[0]["matched_chunk"])

    def test_ranking_metrics(self) -> None:
        ranked = ["resume-a", "resume-b", "resume-c"]
        relevant = {"resume-b", "resume-c"}

        self.assertEqual(precision_at_k(ranked, relevant, 2), 0.5)
        self.assertEqual(recall_at_k(ranked, relevant, 2), 0.5)
        self.assertEqual(reciprocal_rank(ranked, relevant), 0.5)

    def test_labeled_search_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "engineer.txt").write_text("Python engineer with SQL experience", encoding="utf-8")
            (root / "designer.txt").write_text("Product designer and researcher", encoding="utf-8")
            labels = root / "labels.jsonl"
            labels.write_text(
                json.dumps(
                    {"query": "Python engineer", "relevant_candidate_ids": ["engineer.txt"]}
                )
                + "\n",
                encoding="utf-8",
            )

            metrics = evaluate_search_labels(labels, root, k=1)

        self.assertEqual(metrics["queries"], 1)
        self.assertEqual(metrics["precision_at_k"], 1.0)
        self.assertEqual(metrics["recall_at_k"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)


if __name__ == "__main__":
    unittest.main()
