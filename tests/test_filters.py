"""Unit tests for post-search candidate filtering."""

import unittest

from core.filters import filter_candidates
from core.models import Candidate


class TestFilters(unittest.TestCase):
    def setUp(self):
        self.c1 = Candidate(
            candidate_id="c1",
            name="Alice",
            text="Alice Python AWS developer in Austin, TX",
            skills=["python", "aws"],
            experience_years=6.0,
            location="Austin, TX",
        )
        self.c2 = Candidate(
            candidate_id="c2",
            name="Bob",
            text="Bob Java developer in Seattle, WA",
            skills=["java"],
            experience_years=3.0,
            location="Seattle, WA",
        )
        self.c3 = Candidate(
            candidate_id="c3",
            name="Charlie",
            text="Charlie Python developer with no experience stated",
            skills=["python"],
            experience_years=None,
            location=None,
        )

    def test_filter_by_skills(self):
        candidates = [self.c1, self.c2, self.c3]
        filtered = filter_candidates(candidates, required_skills=["python"])
        self.assertEqual({c.candidate_id for c in filtered}, {"c1", "c3"})

        filtered_both = filter_candidates(candidates, required_skills=["python", "aws"])
        self.assertEqual({c.candidate_id for c in filtered_both}, {"c1"})

    def test_filter_by_location(self):
        candidates = [self.c1, self.c2, self.c3]
        filtered = filter_candidates(candidates, location="Austin")
        self.assertEqual({c.candidate_id for c in filtered}, {"c1"})

    def test_filter_by_experience(self):
        candidates = [self.c1, self.c2, self.c3]
        filtered = filter_candidates(candidates, minimum_experience=5.0)
        self.assertEqual({c.candidate_id for c in filtered}, {"c1"})

    def test_combined_filters(self):
        candidates = [self.c1, self.c2, self.c3]
        filtered = filter_candidates(
            candidates,
            required_skills=["python"],
            location="Austin",
            minimum_experience=4.0,
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].candidate_id, "c1")


if __name__ == "__main__":
    unittest.main()
