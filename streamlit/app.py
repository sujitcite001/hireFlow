"""Streamlit recruiter interface for searching and evaluating local resumes."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    GEMINI_API_KEY,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_MODEL,
    HISTORY_RETENTION_DAYS,
    MAX_RESULTS,
    RESUMES_DIR,
    STORE_QUERY_TERMS,
    ensure_data_directories,
)
from core.export import generate_csv_report, generate_pdf_report
from core.gemini_client import test_gemini_connection
from core.ingestion import ingest_resumes, list_resume_files, save_resume
from core.local_index import LocalResumeIndex
from core.memory_rag import record_interaction
from core.models import Candidate
from core.parsing import parse_job_description
from core.recruitment import RecruitmentService

st.set_page_config(page_title="HireFlow", page_icon="🎯", layout="wide")
ensure_data_directories()
service = RecruitmentService()
resume_index = LocalResumeIndex()

# --- Sidebar: Gemini Settings & System Telemetry ---
st.sidebar.title("⚙️ Settings")
st.sidebar.subheader("Gemini Service")
gemini_api_key_input = st.sidebar.text_input(
    "Gemini API Key",
    value=st.session_state.get("custom_gemini_key", GEMINI_API_KEY),
    type="password",
    help="Optional: Enter a Google Gemini API key to enable semantic embeddings and LLM candidate evaluation.",
)
st.session_state["custom_gemini_key"] = gemini_api_key_input.strip()

effective_key = st.session_state["custom_gemini_key"]
has_gemini = bool(effective_key)

if has_gemini:
    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        if st.button("Test Key", use_container_width=True):
            success, msg = test_gemini_connection(effective_key)
            if success:
                st.sidebar.success(msg)
            else:
                st.sidebar.error(msg)
    with col2:
        st.sidebar.info(f"Model: {GEMINI_MODEL}")
else:
    st.sidebar.caption("Gemini is disabled. Using local BM25 search and rule-based evaluation.")

st.sidebar.divider()
st.sidebar.subheader("Privacy & History")
history_detail = "Redacted terms logged" if STORE_QUERY_TERMS else "Terms not stored"
st.sidebar.caption(f"Retention: {HISTORY_RETENTION_DAYS} days | Mode: {history_detail}")

# --- Header ---
st.title("🎯 HireFlow")
st.caption("Local-first candidate search, evidence-based evaluation, and recruitment pipeline")

# --- Section 1: Resume Library Management ---
with st.expander("📁 Resume Library & Document Management", expanded=False):
    up_col1, up_col2 = st.columns([2, 1])
    with up_col1:
        uploads = st.file_uploader(
            "Upload resume files (PDF, TXT, Markdown)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
    with up_col2:
        st.write("")
        st.write("")
        if uploads and st.button("📥 Save Uploaded Resumes", type="primary"):
            saved_files = []
            for upload in uploads:
                try:
                    saved_files.append(save_resume(upload.name, upload.getvalue(), RESUMES_DIR))
                except OSError as error:
                    st.error(f"Could not save {upload.name}: {error}")
            if saved_files:
                st.success(f"Added {len(saved_files)} file(s).")
                st.rerun()

    files = list_resume_files(RESUMES_DIR)
    st.write(f"**{len(files)} resume file(s) available locally:**")

    if files:
        # Load parsed summaries for viewing
        docs = ingest_resumes(RESUMES_DIR, index=resume_index)
        for doc in docs:
            c = doc.get("candidate")
            file_name = doc.get("name")
            if isinstance(c, Candidate):
                with st.container():
                    r_col1, r_col2, r_col3 = st.columns([3, 2, 1])
                    with r_col1:
                        st.markdown(f"**{c.name}** (`{file_name}`)")
                        st.caption(f"Skills: {', '.join(c.skills[:6]) if c.skills else 'None detected'}")
                    with r_col2:
                        exp = f"{c.experience_years:g} yrs" if c.experience_years is not None else "Not specified"
                        st.caption(f"Exp: {exp} | Loc: {c.location or 'Not specified'}")
                    with r_col3:
                        if st.button("Delete", key=f"del-{file_name}"):
                            file_p = Path(doc["path"])
                            if file_p.exists():
                                file_p.unlink()
                                active_paths = {str(p.resolve()) for p in list_resume_files(RESUMES_DIR)}
                                resume_index.prune_missing(RESUMES_DIR, active_paths)
                                st.success(f"Deleted {file_name}")
                                st.rerun()

# --- Section 2: Job Description Input ---
st.subheader("📋 Job Description")
job_text = st.text_area(
    "Paste the job description",
    height=180,
    placeholder="Paste the role title, skills, experience requirements, and location...",
)
parsed_job = parse_job_description(job_text) if job_text.strip() else None

if parsed_job:
    st.info(
        f"**Role:** {parsed_job.title} | "
        f"**Detected Skills:** {', '.join(parsed_job.required_skills) or 'None'} | "
        f"**Min Experience:** {f'{parsed_job.min_experience:g} yrs' if parsed_job.min_experience is not None else 'None stated'} | "
        f"**Location:** {parsed_job.location or 'Any'}"
    )

# --- Section 3: Search and Filter Constraints ---
with st.expander("🔍 Search and Filter Options", expanded=True):
    f_col1, f_col2, f_col3 = st.columns(3)
    skill_choices = (
        parsed_job.required_skills + parsed_job.optional_skills if parsed_job else []
    )
    with f_col1:
        required_skills = st.multiselect(
            "Hard Skill Filters (must have all)",
            options=skill_choices,
            default=[],
            help="Candidates missing any selected skill are strictly filtered out.",
        )
    with f_col2:
        location_filter = st.text_input(
            "Location Constraint",
            value=parsed_job.location or "" if parsed_job else "",
            placeholder="e.g. Remote, San Francisco, New York",
        )
    with f_col3:
        if parsed_job and parsed_job.min_experience is not None:
            apply_exp = st.checkbox(
                f"Require ≥ {parsed_job.min_experience:g} years experience",
                value=False,
            )
            minimum_experience = parsed_job.min_experience if apply_exp else None
        else:
            minimum_experience = None

    c_col1, c_col2, c_col3 = st.columns(3)
    with c_col1:
        result_limit = st.number_input(
            "Maximum results", min_value=1, max_value=100, value=min(MAX_RESULTS, 20)
        )
    with c_col2:
        refresh_index = st.checkbox(
            "Rebuild local resume index",
            value=False,
            help="Re-parse all resumes and rebuild chunk cache before searching.",
        )
    with c_col3:
        use_gemini = st.checkbox(
            "Enable Gemini Hybrid Search & AI Evaluation",
            value=has_gemini,
            disabled=not has_gemini,
            help="When enabled, embeddings & LLM evaluations are run using Gemini.",
        )

# --- Search Execution ---
if st.button("🚀 Search & Rank Candidates", type="primary", disabled=not bool(job_text.strip())):
    with st.spinner("Analyzing candidate pool and generating evidence-backed evaluations..."):
        try:
            st.session_state["hireflow_result"] = service.search(
                job_text,
                limit=int(result_limit),
                required_skills=required_skills,
                location=location_filter,
                minimum_experience=minimum_experience,
                use_gemini=use_gemini,
                api_key=effective_key if has_gemini else None,
                refresh_index=refresh_index,
            )
        except Exception as error:
            st.error(f"Search failed: {error}")
            st.session_state.pop("hireflow_result", None)

# --- Section 4: Ranked Results & Export ---
result = st.session_state.get("hireflow_result")
if result:
    st.divider()
    st.subheader("📊 Ranked Candidates & Fit Analysis")

    if result["state"] == "matches":
        matches = result["results"]
        st.write(
            f"Found **{result['filtered_count']}** qualifying candidate(s) "
            f"via **{result['retrieval_mode']}** retrieval."
        )

        # Export Buttons
        exp_col1, exp_col2, _ = st.columns([1, 1, 3])
        with exp_col1:
            csv_data = generate_csv_report(matches)
            st.download_button(
                label="📥 Export Shortlist (CSV)",
                data=csv_data,
                file_name="hireflow_shortlist.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with exp_col2:
            pdf_data = generate_pdf_report(result["job"], matches)
            st.download_button(
                label="📄 Export Report (PDF)",
                data=pdf_data,
                file_name="hireflow_evaluation_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        # Side-by-side Candidate Comparison
        if len(matches) > 1:
            with st.expander("⚖️ Side-by-Side Comparison Matrix", expanded=False):
                cand_names = [m["candidate"].name for m in matches]
                selected_compare = st.multiselect(
                    "Select 2 to 4 candidates to compare:",
                    options=cand_names,
                    default=cand_names[:min(3, len(cand_names))],
                )
                if selected_compare:
                    comp_cols = st.columns(len(selected_compare))
                    for col, cand_name in zip(comp_cols, selected_compare):
                        cand_item = next(m for m in matches if m["candidate"].name == cand_name)
                        c = cand_item["candidate"]
                        ev = cand_item["evaluation"]
                        with col:
                            st.markdown(f"### {c.name}")
                            st.metric("Fit Score", f"{ev.fit_score:.1f}/10")
                            exp = f"{c.experience_years:g} yrs" if c.experience_years is not None else "N/A"
                            st.write(f"**Experience:** {exp}")
                            st.write(f"**Location:** {c.location or 'N/A'}")
                            st.markdown("**Top Strengths:**")
                            for s in ev.strengths[:3]:
                                st.write(f"- {s}")
                            st.markdown("**Key Gaps:**")
                            for g in ev.gaps[:2]:
                                st.write(f"- {g}")

        # Individual Candidate Cards
        st.write("---")
        for rank, item in enumerate(matches, 1):
            candidate = item["candidate"]
            evaluation = item["evaluation"]
            score_color = "🟢" if evaluation.fit_score >= 7.0 else ("🟡" if evaluation.fit_score >= 4.0 else "🔴")

            card_title = (
                f"{score_color} #{rank} {candidate.name} | "
                f"Fit Score: {evaluation.fit_score:.1f}/10 | "
                f"Search: {item['score']:.2f} ({item.get('retrieval_mode', 'BM25')})"
            )
            with st.expander(card_title, expanded=(rank == 1)):
                st.markdown(f"**Summary:** {evaluation.summary}")
                st.caption(
                    f"**Identified Skills:** {', '.join(candidate.skills) or 'None'} | "
                    f"**Experience:** {f'{candidate.experience_years:g} years' if candidate.experience_years is not None else 'Not identified'} | "
                    f"**Location:** {candidate.location or 'Not identified'}"
                )

                first, second = st.columns(2)
                with first:
                    st.markdown("**💪 Strengths**")
                    for point in evaluation.strengths:
                        st.write(f"- {point}")
                    st.markdown("**⚠️ Gaps**")
                    for point in evaluation.gaps:
                        st.write(f"- {point}")

                with second:
                    st.markdown("**⚡ Risks & Missing Information**")
                    if evaluation.risks:
                        for point in evaluation.risks:
                            st.write(f"- {point}")
                    else:
                        st.write("No major risks identified.")

                    st.markdown("**📌 Resume Evidence Quotes**")
                    if evaluation.evidence:
                        for quote in evaluation.evidence:
                            st.markdown(f"> *\"{quote}\"*")
                    else:
                        st.caption("No direct evidence quotation identified.")

                st.caption(f"Evaluator: `{evaluation.evaluator}` | Candidate ID: `{candidate.candidate_id}`")
                if st.button("Mark Reviewed", key=f"review-{candidate.candidate_id}"):
                    record_interaction(
                        result["job"].text,
                        selected_resume=candidate.candidate_id,
                    )
                    st.success(f"Recorded interaction for {candidate.name}.")

    elif result["state"] == "filters_removed_all":
        st.warning(
            f"Search initially retrieved {result['retrieved_count']} candidate(s), but strict filters eliminated them all."
        )
        diag = result.get("filter_diagnostics", {})
        st.markdown(
            f"- **Excluded by Skills:** {diag.get('dropped_by_skills', 0)}\n"
            f"- **Excluded by Location:** {diag.get('dropped_by_location', 0)}\n"
            f"- **Excluded by Experience:** {diag.get('dropped_by_experience', 0)}"
        )
        st.info("💡 Try unchecking or relaxing hard filters to view candidates based on overall fit.")

        if result.get("broader_results"):
            st.markdown("**Candidates matching search terms prior to filters:**")
            for item in result["broader_results"]:
                c = item["candidate"]
                ev = item["evaluation"]
                st.write(f"- **{c.name}**: Fit Score {ev.fit_score:.1f}/10 (Skills: {', '.join(c.skills[:4])})")

    elif result["state"] == "no_resumes":
        st.info("No resumes are currently in `data/resumes/`. Upload resume files in the Resume Library section above.")
    else:
        st.info("No resumes matched the search terms. Try broader job keywords or enable Gemini semantic search.")

    if result.get("ingestion_errors"):
        with st.expander("⚠️ File Ingestion Warnings"):
            for err in result["ingestion_errors"]:
                st.write(f"- {err}")
