"""Candidate evaluation and stable ordering."""

from collections.abc import Mapping, Sequence
from typing import Any

from core.gemini_client import evaluate_with_gemini
from core.models import Candidate, CandidateEvaluation, JobDescription
from core.parsing import candidate_has_skill


def _evidence_for(candidate: Candidate, phrases: list[str]) -> list[str]:
    """Choose short exact lines from the resume that support an evaluation."""
    evidence = []
    seen = set()
    for line in candidate.text.splitlines():
        clean_line = " ".join(line.strip().split())
        if clean_line and clean_line.casefold() not in seen and any(phrase.casefold() in clean_line.casefold() for phrase in phrases):
            seen.add(clean_line.casefold())
            evidence.append(clean_line[:300])
        if len(evidence) == 5:
            break
    return evidence


def evaluate_candidate(
    candidate: Candidate,
    job: JobDescription,
    use_gemini: bool = False,
    api_key: str | None = None,
) -> CandidateEvaluation:
    """Compare one resume to one job description and explain the result."""
    if use_gemini:
        generated = evaluate_with_gemini(candidate, job, api_key=api_key)
        if generated:
            return generated

    required = {skill.casefold() for skill in job.required_skills}
    optional = {skill.casefold() for skill in job.optional_skills}
    matched_required = sorted(skill for skill in required if candidate_has_skill(candidate, skill))
    missing_required = sorted(required - set(matched_required))
    matched_optional = sorted(skill for skill in optional if candidate_has_skill(candidate, skill))

    strengths = [f"Has required skill: {skill}" for skill in matched_required]
    gaps = [f"Required skill not found: {skill}" for skill in missing_required]
    risks = []
    score_points = 0.0
    total_points = 0.0
    if required:
        skill_match = len(matched_required) / len(required)
        score_points += skill_match * 6.5
        total_points += 6.5
    if optional:
        strengths.extend(f"Has preferred skill: {skill}" for skill in matched_optional)
        optional_match = len(matched_optional) / len(optional)
        score_points += optional_match
        total_points += 1
    if job.min_experience is not None:
        if candidate.experience_years is None:
            gaps.append("Years of experience could not be identified from the resume")
            risks.append("Experience information is missing or unclear")
            total_points += 2.5
        else:
            meets_experience = candidate.experience_years >= job.min_experience
            total_points += 2.5
            if meets_experience:
                score_points += 2.5
                strengths.append(f"Meets experience requirement ({candidate.experience_years:g} years)")
            else:
                gaps.append(f"Has {candidate.experience_years:g} years; job asks for {job.min_experience:g}")
                risks.append("Stated experience is below the job requirement")
    if job.location:
        matches_location = bool(
            candidate.location and job.location.casefold() in candidate.location.casefold()
        )
        total_points += 1
        if matches_location:
            score_points += 1
            strengths.append(f"Location matches: {candidate.location}")
        elif candidate.location:
            gaps.append(f"Location does not match: {candidate.location}")
        else:
            gaps.append("Location could not be identified from the resume")
            risks.append("Location information is missing or unclear")

    if total_points == 0:
        total_points = 1
    fit_score = 10 * score_points / total_points
    if not required and not optional:
        strengths.append("No standard skills were detected in the job description")
    evidence_phrases = matched_required + matched_optional
    if candidate.experience_years is not None:
        evidence_phrases.append(f"{candidate.experience_years:g}")
    evidence = _evidence_for(candidate, evidence_phrases)
    summary = (
        f"{candidate.name} matches {len(matched_required)} of {len(required)} required skills. "
        f"Rule-based fit score: {fit_score:.1f}/10."
    )
    return CandidateEvaluation(
        candidate_id=candidate.candidate_id,
        fit_score=round(fit_score, 1),
        strengths=strengths,
        gaps=gaps,
        risks=risks,
        summary=summary,
        evidence=evidence,
    )


def evaluate_and_rank(
    results: Sequence[Mapping[str, Any]],
    job: JobDescription,
    use_gemini: bool = False,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Evaluate retrieved candidates and combine fit and retrieval scores."""
    ranked = []
    for result in results:
        candidate = result.get("candidate")
        if not isinstance(candidate, Candidate):
            continue
        evaluation = evaluate_candidate(candidate, job, use_gemini=use_gemini, api_key=api_key)
        retrieval_score = float(result.get("score", 0.0))
        final_score = 0.4 * retrieval_score + 0.6 * evaluation.fit_score / 10
        ranked.append({**result, "evaluation": evaluation, "final_score": final_score})
    return [dict(item) for item in rerank(ranked, score_key="final_score")]


def rerank(
    results: Sequence[Mapping[str, object]], score_key: str = "score"
) -> list[Mapping[str, object]]:
    """Sort by descending score, then filename for deterministic ties."""
    return sorted(
        results,
        key=lambda item: (-float(item.get(score_key, 0.0)), str(item.get("name", "")).casefold()),
    )
