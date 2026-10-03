"""Post-search filtering helpers."""

from collections.abc import Iterable

from core.models import Candidate
from core.parsing import candidate_has_skill


def filter_candidates(
    candidates: Iterable[Candidate],
    required_skills: list[str] | None = None,
    location: str | None = None,
    minimum_experience: float | None = None,
) -> list[Candidate]:
    """Apply only the filters the recruiter selected."""
    skills = {skill.casefold() for skill in required_skills or []}
    wanted_location = (location or "").casefold().strip()
    filtered = []
    for candidate in candidates:
        if skills and not all(candidate_has_skill(candidate, skill) for skill in skills):
            continue
        if wanted_location and wanted_location not in (candidate.location or "").casefold():
            continue
        if minimum_experience is not None and (
            candidate.experience_years is None or candidate.experience_years < minimum_experience
        ):
            continue
        filtered.append(candidate)
    return filtered
