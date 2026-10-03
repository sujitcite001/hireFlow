"""Unit tests for resume and job description parsing."""

import unittest

from core.parsing import (
    _extract_candidate_name,
    _extract_experience_from_dates,
    _extract_location,
    clean_text,
    extract_skills,
    parse_candidate_text,
    parse_job_description,
    split_into_chunks,
)


class TestParsing(unittest.TestCase):
    def test_clean_text_normalizes_whitespace(self):
        raw = "Hello   world!\r\n\r\n\r\nSecond   line.\x00"
        cleaned = clean_text(raw)
        self.assertNotIn("\x00", cleaned)
        self.assertNotIn("\r", cleaned)
        self.assertEqual(cleaned, "Hello world!\n\nSecond line.")

    def test_split_into_chunks(self):
        text = "word " * 400
        chunks = split_into_chunks(text, chunk_size=200, overlap=50)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 200 for c in chunks))

    def test_candidate_name_skips_headers(self):
        lines = [
            "CURRICULUM VITAE",
            "Software Engineer",
            "Sarah Connor",
            "sarah@example.com",
            "(555) 019-2834",
        ]
        name = _extract_candidate_name(lines, fallback_name="fallback.txt")
        self.assertEqual(name, "Sarah Connor")

    def test_extract_skills_known_and_labeled(self):
        text = """
        Experienced in Python, Docker, Kubernetes, and PostgreSQL.
        Technical Skills: Go, Rust, Next.js, GraphQL; Redis
        """
        skills = extract_skills(text)
        self.assertIn("python", skills)
        self.assertIn("docker", skills)
        self.assertIn("kubernetes", skills)
        self.assertIn("postgresql", skills)
        self.assertIn("go", skills)
        self.assertIn("rust", skills)
        self.assertIn("next.js", skills)

    def test_extract_experience_from_dates(self):
        text = """
        Senior Engineer | TechCorp (2018 - 2022)
        Junior Developer | Startup (2015 - 2018)
        """
        exp = _extract_experience_from_dates(text)
        self.assertIsNotNone(exp)
        # 2015 to 2022 is 7 years
        self.assertEqual(exp, 7.0)

    def test_extract_experience_explicit(self):
        text = "Senior developer with 8+ years of experience in distributed systems."
        cand = parse_candidate_text(text, "cand-1", "candidate.txt")
        self.assertEqual(cand.experience_years, 8.0)

    def test_extract_location(self):
        text = "Alex Mercer\nAustin, TX\nalex@example.com"
        loc = _extract_location(text)
        self.assertEqual(loc, "Austin, TX")

    def test_parse_job_description(self):
        jd_text = """
        Senior Python Engineer
        Location: Remote, US
        Requirements:
        At least 5 years of experience.
        Must know Python, SQL, and Docker.
        Nice to have: Kubernetes, FastAPI.
        """
        jd = parse_job_description(jd_text)
        self.assertEqual(jd.title, "Senior Python Engineer")
        self.assertEqual(jd.min_experience, 5.0)
        self.assertIn("python", jd.required_skills)
        self.assertIn("sql", jd.required_skills)
        self.assertIn("kubernetes", jd.optional_skills)


if __name__ == "__main__":
    unittest.main()
