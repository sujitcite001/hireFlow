"""Simple validated data models used by the recruitment workflow."""

from pydantic import BaseModel, Field


class Candidate(BaseModel):
    candidate_id: str
    name: str
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    text: str
    skills: list[str] = Field(default_factory=list)
    experience_years: float | None = None


class JobDescription(BaseModel):
    jd_id: str
    title: str
    text: str
    required_skills: list[str] = Field(default_factory=list)
    optional_skills: list[str] = Field(default_factory=list)
    min_experience: float | None = None
    location: str | None = None


class CandidateEvaluation(BaseModel):
    candidate_id: str
    fit_score: float = Field(ge=0, le=10)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    summary: str
    evidence: list[str] = Field(default_factory=list)
    evaluator: str = "rules"