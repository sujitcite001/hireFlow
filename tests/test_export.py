"""Unit tests for PDF and CSV export utilities."""

import unittest

from core.export import generate_csv_report, generate_pdf_report
from core.models import Candidate, CandidateEvaluation, JobDescription


class TestExport(unittest.TestCase):
    def setUp(self):
        self.candidate = Candidate(
            candidate_id="c1",
            name="Alice Walker",
            text="Alice Walker Python Engineer",
            skills=["python", "docker"],
            experience_years=5.0,
            location="Denver, CO",
        )
        self.evaluation = CandidateEvaluation(
            candidate_id="c1",
            fit_score=9.0,
            strengths=["Strong Python skills"],
            gaps=[],
            risks=[],
            summary="Alice matches all criteria.",
            evidence=["Alice Walker Python Engineer"],
            evaluator="rules",
        )
        self.job = JobDescription(
            jd_id="j1",
            title="Senior Python Engineer",
            text="Looking for a Python engineer.",
            required_skills=["python"],
        )
        self.results = [{
            "candidate": self.candidate,
            "evaluation": self.evaluation,
            "score": 0.88,
            "retrieval_mode": "hybrid",
        }]

    def test_csv_export_contains_headers_and_data(self):
        csv_str = generate_csv_report(self.results)
        self.assertIn("Candidate Name", csv_str)
        self.assertIn("Alice Walker", csv_str)
        self.assertIn("9.0", csv_str)
        self.assertIn("Denver, CO", csv_str)

    def test_pdf_export_generates_valid_pdf_bytes(self):
        pdf_bytes = generate_pdf_report(self.job, self.results)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 100)


if __name__ == "__main__":
    unittest.main()
