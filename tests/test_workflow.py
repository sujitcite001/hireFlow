"""Tests for the recruiter workflow using only temporary local files."""

from pathlib import Path
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from config import HISTORY_RETENTION_DAYS
from core.evaluator import answer_correctness, context_precision, context_recall, faithfulness
from core.filters import filter_candidates
from core.ingestion import extract_text, ingest_resumes
from core.local_index import LocalResumeIndex
from core.memory_rag import record_interaction
from core.models import Candidate
from core.parsing import candidate_has_skill, parse_candidate_text, parse_job_description, split_into_chunks
from core.recruitment import RecruitmentService
from core.vector_store import VectorStore


class ParsingTests(unittest.TestCase):
    def test_job_and_candidate_fields_are_extracted(self) -> None:
        job = parse_job_description(
            "Data Scientist\nRequired: Python, SQL. Docker is nice to have.\n"
            "Minimum 4 years of experience.\nLocation: Toronto"
        )
        candidate = parse_candidate_text(
            "Jamie Candidate\njamie@example.com\nLocation: Toronto\n"
            "5 years of experience with Python and SQL.",
            "jamie",
            "jamie",
        )

        self.assertEqual(job.title, "Data Scientist")
        self.assertEqual(job.min_experience, 4)
        self.assertEqual(job.location, "Toronto")
        self.assertIn("python", job.required_skills)
        self.assertIn("docker", job.optional_skills)
        self.assertEqual(candidate.name, "Jamie Candidate")
        self.assertEqual(candidate.email, "jamie@example.com")
        self.assertEqual(candidate.experience_years, 5)
        self.assertIn("sql", candidate.skills)

    def test_chunking_keeps_overlap(self) -> None:
        chunks = split_into_chunks("abcdefghij", chunk_size=6, overlap=2)
        self.assertEqual(chunks, ["abcdef", "efghij"])

    def test_custom_skills_from_labeled_sections_are_searchable(self) -> None:
        job = parse_job_description(
            "Platform Engineer\nRequired skills: Kotlin, Pulumi\n"
            "Preferred skills: Temporal"
        )
        candidate = parse_candidate_text(
            "Alex Example\nSkills: Kotlin, Pulumi\nBuilt services with Temporal.",
            "alex",
            "alex",
        )

        self.assertIn("kotlin", job.required_skills)
        self.assertIn("pulumi", job.required_skills)
        self.assertIn("temporal", job.optional_skills)
        self.assertTrue(candidate_has_skill(candidate, "Temporal"))
        self.assertEqual(filter_candidates([candidate], required_skills=["Pulumi"]), [candidate])


class RecruitmentTests(unittest.TestCase):
    def test_service_uses_ordered_langgraph_steps(self) -> None:
        workflow = RecruitmentService().workflow
        node_names = set(workflow.get_graph().nodes)

        self.assertTrue(
            {
                "parse_job",
                "find_resumes",
                "apply_filters",
                "evaluate_candidates",
                "prepare_response",
                "save_search",
            }.issubset(node_names)
        )

    def test_job_search_filters_and_returns_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "jamie.txt").write_text(
                "Jamie Candidate\nLocation: Toronto\n5 years of experience.\n"
                "Python and SQL developer with data science experience.",
                encoding="utf-8",
            )
            (root / "alex.txt").write_text(
                "Alex Example\nLocation: Vancouver\n2 years of experience.\n"
                "Python and SQL developer.",
                encoding="utf-8",
            )
            service = RecruitmentService(
                resume_directory=root,
                interaction_file=root / "history.jsonl",
            )
            result = service.search(
                "Data Scientist\nPython and SQL\nMinimum 4 years of experience\nLocation: Toronto",
                required_skills=["python"],
                location="Toronto",
                minimum_experience=4,
            )

            self.assertEqual(result["state"], "matches")
            self.assertEqual([item["candidate"].name for item in result["results"]], ["Jamie Candidate"])
            evaluation = result["results"][0]["evaluation"]
            self.assertGreater(evaluation.fit_score, 0)
            self.assertTrue(evaluation.evidence)
            self.assertTrue((root / "history.jsonl").exists())

            without_hard_filters = service.search(
                "Data Scientist\nPython and SQL\nMinimum 4 years of experience\nLocation: Toronto"
            )
            self.assertEqual(len(without_hard_filters["results"]), 2)

    def test_unreadable_pdf_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "broken.pdf").write_text("not a real PDF", encoding="utf-8")
            result = RecruitmentService(resume_directory=root).search("Python developer")

        self.assertEqual(result["state"], "no_retrieval_matches")
        self.assertTrue(any("broken.pdf" in error for error in result["ingestion_errors"]))


class LocalIndexTests(unittest.TestCase):
    def test_cache_reuses_unchanged_parse_and_invalidates_edits_and_deletions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "resumes"
            root.mkdir()
            resume = root / "candidate.txt"
            resume.write_text("Candidate One\nSkills: Rust, Go", encoding="utf-8")
            index = LocalResumeIndex(Path(temporary_directory) / "index.sqlite3")

            with patch("core.ingestion.extract_text", wraps=extract_text) as extractor:
                first = ingest_resumes(root, index=index)
                cached = ingest_resumes(root, index=index)
                self.assertEqual(extractor.call_count, 1)
                self.assertEqual(cached[0]["tokenized_chunks"], first[0]["tokenized_chunks"])
                ingest_resumes(root, index=index, force_refresh=True)
                self.assertEqual(extractor.call_count, 2)

                resume.write_text("Candidate One\nSkills: Rust, Kotlin", encoding="utf-8")
                changed = ingest_resumes(root, index=index)
                self.assertEqual(extractor.call_count, 3)
                self.assertIn("Kotlin", changed[0]["text"])
                resume.unlink()
                self.assertEqual(ingest_resumes(root, index=index), [])


class InteractionHistoryTests(unittest.TestCase):
    def test_history_minimizes_queries_and_prunes_expired_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            history = Path(temporary_directory) / "interactions.jsonl"
            expired = {
                "timestamp": (
                    datetime.now(timezone.utc) - timedelta(days=HISTORY_RETENTION_DAYS + 1)
                ).isoformat(),
                "query": "expired job description",
            }
            legacy = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query": "Legacy role contact ada@example.com",
                "selected_resume": "ada-resume.pdf",
            }
            history.write_text(
                json.dumps(expired) + "\n" + json.dumps(legacy) + "\n", encoding="utf-8"
            )

            with patch("core.memory_rag.STORE_QUERY_TERMS", False):
                record_interaction(
                    "Python role contact ada@example.com +1 416 555 0123",
                    selected_resume="ada-resume.pdf",
                    path=history,
                )

            records = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 2)
            self.assertTrue(all("query" not in record for record in records))
            self.assertNotIn("ada-resume.pdf", json.dumps(records))
            self.assertNotIn("ada@example.com", json.dumps(records))

    def test_opt_in_query_terms_are_pii_filtered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            history = Path(temporary_directory) / "interactions.jsonl"
            with patch("core.memory_rag.STORE_QUERY_TERMS", True):
                record_interaction(
                    "Python role contact ada@example.com +1 416 555 0123", path=history
                )

            record = json.loads(history.read_text(encoding="utf-8").strip())
            self.assertIn("python", record["query_terms"])
            self.assertNotIn("ada", record["query_terms"])
            self.assertNotIn("example", record["query_terms"])
            self.assertNotIn("555", record["query_terms"])


class EvaluationTests(unittest.TestCase):
    def test_simple_rag_metrics(self) -> None:
        self.assertEqual(context_recall("Python and SQL", "Python SQL AWS"), 2 / 3)
        self.assertEqual(context_precision(["Python developer", "unrelated text"], ["Python work"]), 0.5)
        self.assertEqual(answer_correctness("Python SQL", "Python SQL"), 1.0)
        self.assertEqual(faithfulness("Python engineer", ["Python engineer resume"]), 1.0)

    def test_candidate_score_is_bounded(self) -> None:
        candidate = Candidate(
            candidate_id="id", name="A", text="Python engineer", skills=["python"]
        )
        job = parse_job_description("Engineer\nPython and SQL required")
        from core.re_ranker import evaluate_candidate

        evaluation = evaluate_candidate(candidate, job)
        self.assertGreaterEqual(evaluation.fit_score, 0)
        self.assertLessEqual(evaluation.fit_score, 10)
        self.assertTrue(evaluation.gaps)


class VectorStoreTests(unittest.TestCase):
    def test_vectors_are_saved_and_dimension_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = VectorStore(Path(temporary_directory) / "vectors.sqlite3", dimensions=3)
            store.upsert("candidate-1", [0.1, 0.2, 0.3], {"source": "test"})

            saved = store.get("candidate-1")
            self.assertEqual(saved["vector"], [0.1, 0.2, 0.3])
            self.assertEqual(saved["metadata"]["source"], "test")
            with self.assertRaises(ValueError):
                store.upsert("candidate-2", [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
