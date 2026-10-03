# HireFlow Task Breakdown

**Source:** [DESIGN.md](DESIGN.md)  
**Scope:** Current local-first application. Hosted or multi-user work is conditional on a separate deployment decision.

## Completed Foundation

### HF-01: Chunk-Level Retrieval

- **Status:** Complete
- **Work:** Score overlapping resume chunks with BM25, return the best matching passage, and persist optional Gemini embeddings per chunk.
- **Acceptance:** Local search works without Gemini; semantic vectors are keyed and invalidated per chunk; retrieval tests cover passage selection.

### HF-02: Incremental Local Index

- **Status:** Complete
- **Work:** Cache parsed resumes, chunks, and tokenized chunks using source-content and cache-version fingerprints. Expose manual refresh in Streamlit and the CLI.
- **Acceptance:** Unchanged documents reuse cached parsing; file changes, deletions, and explicit refresh update the index.

### HF-03: Flexible Skill Extraction

- **Status:** Complete for labeled lists
- **Work:** Extract arbitrary skill terms from labeled skills, technology, tool, framework, and platform sections; use requested terms in filtering/evaluation.
- **Acceptance:** Tests cover terms outside the built-in vocabulary in both job descriptions and resumes.

### HF-04: Privacy-Minimized Interaction History

- **Status:** Complete
- **Work:** Hash query and reviewed-resume identifiers, make query-token storage opt-in, redact email/phone patterns from retained tokens, enforce configurable expiry, and migrate legacy raw records.
- **Acceptance:** Default records contain no raw job text; expired records are pruned; migration removes raw query and selected-resume fields.

### HF-05: Labeled Retrieval Evaluation

- **Status:** Complete; needs real labels
- **Work:** Add a CLI evaluator for JSONL query/relevance examples reporting macro Precision@k, Recall@k, and MRR.
- **Acceptance:** Synthetic tests pass; the CLI accepts a local labeled file and reports the three metrics.

### HF-06: Design and Configuration Documentation

- **Status:** Complete
- **Work:** Document the HLD, LLD, trust boundaries, failure handling, current limits, refresh controls, history settings, and evaluation command. Remove the credential-looking example value.
- **Acceptance:** Documentation matches current code; `.env.example` contains an empty API-key placeholder.

## Next Tasks

### HF-07: Build a Recruiter-Reviewed Evaluation Set

- **Priority:** P1
- **Dependencies:** Recruiter-provided relevance judgments
- **Work:** Create a local JSONL dataset with representative job queries and relevant resume IDs. Keep the dataset and resumes out of source control unless explicitly anonymized and approved.
- **Acceptance:** Dataset covers varied roles and vocabulary; BM25 and opt-in hybrid modes have recorded Precision@k, Recall@k, and MRR baselines. Do not change default weights until results are reviewed.

### HF-08: Evaluate Local Resume Parsing Options

- **Priority:** P1
- **Dependencies:** A recruiter-reviewed, privacy-approved sample with expected fields
- **Work:** Compare current extraction against an offline-capable parsing approach for names, contact details, locations, experience, and unstructured skill sections. Keep the current parser as fallback until a candidate approach demonstrates better accuracy.
- **Acceptance:** Report field-level precision/recall or equivalent error rates; verify offline operation; add representative regression cases; document model/dependency size and limitations before adoption.

### HF-09: Verify Retention and Key Hygiene

- **Priority:** P1
- **Dependencies:** Confirm whether the removed example credential was ever valid
- **Work:** Revoke and replace the credential if valid. Confirm the replacement is stored only in a local ignored `.env` or environment variable.
- **Acceptance:** No active credential is present in tracked examples or source; local-only configuration remains usable.

## Conditional Deployment Tasks

### HF-10: Design Hosted/Multi-User Data Isolation

- **Priority:** Gate before any hosted deployment
- **Dependencies:** Product decision to support shared or hosted use
- **Work:** Define tenant-scoped candidate identity, authorization, encryption, provider consent, retention/deletion, auditability, and coordinated index lifecycle. Reassess local SQLite and filesystem assumptions.
- **Acceptance:** Architecture and threat model reviewed before implementation; tests demonstrate cross-tenant access is impossible; lifecycle and deletion behavior are verified.

## Cross-Cutting Acceptance Criteria

- Local BM25 search and rule-based evaluation require no network service or credentials.
- Gemini remains opt-in, and any transfer of resume/job content is explicit in the UI.
- Search exposes evidence and missing information; it does not make hiring decisions.
- Parser, index, history, and scoring behavior have focused regression tests.
- Run `python -m unittest discover -s tests -v` and `python -m compileall -q .` after Python changes.