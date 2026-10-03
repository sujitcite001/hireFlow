"""Run candidate search as a modular LangGraph workflow."""

from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from config import MAX_RESULTS, RESUMES_DIR
from core.filters import filter_candidates
from core.hybrid_indexer import search_resumes
from core.ingestion import list_resume_files
from core.local_index import LocalResumeIndex
from core.memory_rag import record_interaction
from core.models import Candidate, CandidateEvaluation, JobDescription
from core.parsing import candidate_has_skill, parse_job_description
from core.re_ranker import evaluate_and_rank
from core.vector_store import VectorStore


class RecruitmentState(TypedDict, total=False):
    """Values passed from one graph step to the next."""

    job_text: str
    limit: int
    required_skills: list[str] | None
    location: str | None
    minimum_experience: float | None
    use_gemini: bool
    api_key: str | None
    refresh_index: bool
    job: JobDescription
    retrieved: list[dict[str, Any]]
    ingestion_errors: list[str]
    selected: list[dict[str, Any]]
    filter_diagnostics: dict[str, Any]
    ranked: list[dict[str, Any]]
    response: dict[str, Any]


class RecruitmentService:
    """Connect job parsing, search, filtering, and evaluation."""

    def __init__(
        self,
        resume_directory: str | Path = RESUMES_DIR,
        vector_store: VectorStore | None = None,
        interaction_file: str | Path | None = None,
        resume_index: LocalResumeIndex | None = None,
    ) -> None:
        self.resume_directory = Path(resume_directory)
        self.vector_store = vector_store
        self.interaction_file = Path(interaction_file) if interaction_file else None
        self.resume_index = resume_index
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        """Connect the search steps in the order they should run."""
        graph = StateGraph(RecruitmentState)
        graph.add_node("parse_job", self._parse_job)
        graph.add_node("find_resumes", self._find_resumes)
        graph.add_node("apply_filters", self._apply_filters)
        graph.add_node("evaluate_candidates", self._evaluate_candidates)
        graph.add_node("prepare_response", self._prepare_response)
        graph.add_node("save_search", self._save_search)

        graph.add_edge(START, "parse_job")
        graph.add_edge("parse_job", "find_resumes")
        graph.add_edge("find_resumes", "apply_filters")
        graph.add_edge("apply_filters", "evaluate_candidates")
        graph.add_edge("evaluate_candidates", "prepare_response")
        graph.add_edge("prepare_response", "save_search")
        graph.add_edge("save_search", END)
        return graph.compile()

    def _parse_job(self, state: RecruitmentState) -> dict[str, Any]:
        return {"job": parse_job_description(state["job_text"])}

    def _find_resumes(self, state: RecruitmentState) -> dict[str, Any]:
        errors: list[str] = []
        resumes = search_resumes(
            state["job"].text,
            directory=self.resume_directory,
            limit=10000,
            vector_store=self.vector_store,
            use_semantic=state.get("use_gemini", False),
            api_key=state.get("api_key"),
            ingestion_errors=errors,
            resume_index=self.resume_index,
            refresh_index=state.get("refresh_index", False),
        )
        return {"retrieved": resumes, "ingestion_errors": errors}

    def _apply_filters(self, state: RecruitmentState) -> dict[str, Any]:
        candidates = [
            resume["candidate"]
            for resume in state["retrieved"]
            if isinstance(resume.get("candidate"), Candidate)
        ]
        accepted = filter_candidates(
            candidates,
            required_skills=state.get("required_skills"),
            location=state.get("location"),
            minimum_experience=state.get("minimum_experience"),
        )
        accepted_ids = {candidate.candidate_id for candidate in accepted}
        selected = [
            resume
            for resume in state["retrieved"]
            if resume["candidate"].candidate_id in accepted_ids
        ]

        # Calculate diagnostics to explain why any candidates were dropped
        diagnostics = {
            "dropped_by_skills": 0,
            "dropped_by_location": 0,
            "dropped_by_experience": 0,
        }
        req_skills = {s.casefold() for s in (state.get("required_skills") or [])}
        req_loc = (state.get("location") or "").casefold().strip()
        min_exp = state.get("minimum_experience")

        for c in candidates:
            if c.candidate_id not in accepted_ids:
                if req_skills and not all(candidate_has_skill(c, s) for s in req_skills):
                    diagnostics["dropped_by_skills"] += 1
                if req_loc and req_loc not in (c.location or "").casefold():
                    diagnostics["dropped_by_location"] += 1
                if min_exp is not None and (c.experience_years is None or c.experience_years < min_exp):
                    diagnostics["dropped_by_experience"] += 1

        return {"selected": selected, "filter_diagnostics": diagnostics}

    def _evaluate_candidates(self, state: RecruitmentState) -> dict[str, Any]:
        candidates = state["selected"]
        if state.get("use_gemini", False):
            # Avoid excessive API calls: evaluate only the search shortlist
            candidates = candidates[: max(0, state.get("limit", MAX_RESULTS))]
        ranked = evaluate_and_rank(
            candidates,
            state["job"],
            use_gemini=state.get("use_gemini", False),
            api_key=state.get("api_key"),
        )
        return {"ranked": ranked[: max(0, state.get("limit", MAX_RESULTS))]}

    def _prepare_response(self, state: RecruitmentState) -> dict[str, Any]:
        ranked = state["ranked"]
        retrieved = state["retrieved"]
        resume_count = len(list_resume_files(self.resume_directory))

        if ranked:
            search_state = "matches"
        elif resume_count == 0:
            search_state = "no_resumes"
        elif not retrieved:
            search_state = "no_retrieval_matches"
        else:
            search_state = "filters_removed_all"

        wider_results = []
        if not ranked and retrieved:
            wider_results = evaluate_and_rank(
                retrieved[:MAX_RESULTS],
                state["job"],
                use_gemini=False,
            )

        response = {
            "job": state["job"],
            "results": ranked,
            "broader_results": wider_results,
            "retrieved_count": len(retrieved),
            "filtered_count": len(state["selected"]),
            "filter_diagnostics": state.get("filter_diagnostics", {}),
            "state": search_state,
            "retrieval_mode": retrieved[0]["retrieval_mode"] if retrieved else "BM25",
            "resume_file_count": resume_count,
            "ingestion_errors": state["ingestion_errors"],
        }
        return {"response": response}

    def _save_search(self, state: RecruitmentState) -> dict[str, Any]:
        record_interaction(state["job"].text, path=self.interaction_file)
        return {}

    def search(
        self,
        job_text: str,
        limit: int = MAX_RESULTS,
        required_skills: list[str] | None = None,
        location: str | None = None,
        minimum_experience: float | None = None,
        use_gemini: bool = False,
        api_key: str | None = None,
        refresh_index: bool = False,
    ) -> dict[str, Any]:
        """Run the graph and return recruiter-facing results."""
        final_state = self.workflow.invoke(
            {
                "job_text": job_text,
                "limit": limit,
                "required_skills": required_skills,
                "location": location,
                "minimum_experience": minimum_experience,
                "use_gemini": use_gemini,
                "api_key": api_key,
                "refresh_index": refresh_index,
            }
        )
        return final_state["response"]
