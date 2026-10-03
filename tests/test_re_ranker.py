"""Unit tests for rule-based candidate evaluation and reranking."""

import unittest

from core.models import Candidate, JobDescription
from core.re_ranker import evaluate_candidate, rerank


class TestReRanker(unittest.TestCase):
    def test_evaluate_candidate_matches_skills_and_experience(self):
        candidate = Candidate(
            candidate_id="cand-101",
            name="Diana Prince",
            text="Diana Prince Python, SQL, Docker, AWS engineer with 6 years experience in Chicago, IL.",
            skills=["python", "sql", "docker", "aws"],
            experience_years=6.0,
            location="Chicago, IL",
        )
        job = JobDescription(
            jd_id="jd-01",
            title="Senior Python Backend Engineer",
            text="Looking for a Python and Docker engineer with at least 4 years experience in Chicago.",
            required_skills=["python", "docker"],
            optional_skills=["aws"],
            min_experience=4.0,
            location="Chicago",
        )

        evaluation = evaluate_candidate(candidate, job, use_gemini=False)
        self.assertEqual(evaluation.evaluator, "rules")
        self.assertGreaterEqual(evaluation.fit_score, 8.0)
        self.assertTrue(any("python" in s.lower() for s in evaluation.strengths))
        self.assertTrue(any("docker" in s.lower() for s in evaluation.strengths))
        self.assertEqual(len(evaluation.gaps), 0)

    def test_rerank_orders_by_score_descending(self):
        items = [
            {"name": "b.txt", "score": 0.5},
            {"name": "a.txt", "score": 0.9},
            {"name": "c.txt", "score": 0.9},
        ]
        sorted_items = rerank(items, score_key="score")
        self.assertEqual(sorted_items[0]["name"], "a.txt")
        self.assertEqual(sorted_items[1]["name"], "c.txt")
        self.assertEqual(sorted_items[2]["name"], "b.txt")


if __name__ == "__main__":
    unittest.main()
