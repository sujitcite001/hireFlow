# HireFlow

HireFlow is a local-first tool for finding and reviewing candidates against a job description. It reads PDF, TXT, and Markdown resumes from `data/resumes/`, searches them, applies optional filters, and shows a simple evidence-based evaluation.

By default, documents stay on this computer. Gemini is optional. The recruiter must turn it on in the app before resume or job text is sent to Google's Gemini service. If Gemini is unavailable, HireFlow uses local BM25 search and rule-based evaluation instead.

## Requirements

- Python 3.12 or newer
- No account or API key for local BM25 search and rule-based evaluation
- Optional `GEMINI_API_KEY` to use Gemini embeddings and evaluation

## Install and Run

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run streamlit/app.py
```

The app creates `data/resumes/`, `data/hybrid_index/`, and `data/memory/` automatically. Resume files and interaction history are local and ignored by Git.

The command-line interface can list and search local resumes:

```powershell
python main.py list
python main.py search "Python data science"
python main.py search "platform engineer" --refresh-index --use-gemini
```

The local resume cache is updated when a file is new or its contents change. Use `--refresh-index` to force re-parsing. Ranking quality can be measured against a local JSONL file; each line has a query and relevant candidate file IDs:

```json
{"query":"Python data engineer","relevant_candidate_ids":["resume-01.txt"]}
```

```powershell
python main.py evaluate data/evaluation_labels.jsonl --k 10
```

The command reports macro Precision@k, Recall@k, and mean reciprocal rank. Evaluation labels should reflect recruiter-reviewed relevance; use representative examples before changing BM25/vector weights.

Run the checks with:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q .
```

## Task 1: Requirements

### Functional requirements

1. Recruiters can add PDF, TXT, and Markdown resumes.
2. The system extracts candidate name, contact details, skills, experience, location, and searchable text when those details are present.
3. Recruiters can paste a job description and search for relevant candidates.
4. Recruiters can apply hard filters for skills, location, and experience.
5. The system ranks candidates and shows a fit score, strengths, gaps, risks, and supporting resume text.

### Non-functional requirements

1. Local search works without network access or credentials.
2. Search should respond quickly for a small local resume collection; a larger collection can move to a dedicated search service.
3. A damaged resume or unavailable Gemini service must not stop the rest of the search.
4. API keys and candidate data must not be committed to source control.
5. Search, parsing, filtering, storage, evaluation, and UI code are separate modules.
6. Each search is recorded locally so basic activity can be reviewed.

## Task 2: High-Level Design

```text
Recruiter
   |
   v
Streamlit UI ---- resume upload and job description
   |
   v
RecruitmentService
   |                \
   |                 +---- local interaction history
   v
Resume ingestion -> PDF/text parsing -> Candidate records
   |
   v
Hybrid search: BM25 + optional Gemini embeddings
   |
   v
Recruiter-selected filters
   |
   v
Candidate evaluator: optional Gemini, otherwise local rules
   |
   v
Ranked results with evidence
```

The UI calls `RecruitmentService`, which coordinates the business logic. The service calls small modules for parsing, search, filtering, ranking, and history. `VectorStore` uses SQLite locally. It can be replaced with FAISS or Pinecone if a deployment needs a larger vector index.

The service runs those steps as a LangGraph workflow: parse job, find resumes, apply filters, evaluate candidates, prepare the response, then save the search. Each step reads named values from the workflow state and returns values for the next step. This keeps the order visible and makes individual steps easier to replace or test. LangChain's Google GenAI adapters handle optional Gemini embeddings and structured candidate evaluations. Gemini remains off unless the recruiter enables it and a key is configured. To limit cost and latency, Gemini evaluates only the requested top search results.

## Task 3: Search Decision

HireFlow uses hybrid search when the recruiter enables Gemini and supplies a valid API key. BM25 finds exact terms such as `Python` or `TensorFlow`; embeddings can find related wording. The combined score is `0.6 * BM25 + 0.4 * vector similarity`. Without vectors, search uses BM25 by itself.

This gives exact skill matches a strong signal and can improve semantic recall when Gemini is available. BM25 is fast and simple for small collections. Hybrid search adds an external dependency, cost, and latency, so it is opt-in. The current SQLite vector table suits local development; use a managed index for a large corpus.

## Task 4: Main Components

### Document processing (`core/ingestion.py`, `core/parsing.py`)

- **Responsibility:** Read resume files and create structured candidate records.
- **Input:** PDF, TXT, or Markdown files.
- **Output:** Clean text, candidate details, and overlapping text chunks.
- **Steps:** Discover files, extract text, clean it, identify common fields, and split long text into 1,000-character chunks with 200-character overlap.
- **Dependencies:** `pypdf` for PDFs, Pydantic for data validation.
- **Failures:** Unsupported formats are ignored; unreadable files are reported while other resumes continue. Image-only/scanned PDFs may have no extractable text and need OCR, which is not included.

### Hybrid indexer (`core/hybrid_indexer.py`, `core/vector_store.py`)

- **Responsibility:** Find likely candidates using exact terms and optional semantic vectors.
- **Input:** Job description and parsed resumes.
- **Output:** Ranked candidates and retrieval scores.
- **Steps:** Tokenize text, calculate BM25 scores, optionally create/query Gemini embeddings, combine available scores, and return the highest-ranked candidates.
- **Dependencies:** Local Python code, SQLite, LangChain, and LangGraph; optional Gemini API.
- **Failures:** If Gemini or vector creation fails, continue with BM25. A malformed resume is skipped and reported.

### Candidate evaluator (`core/re_ranker.py`, `core/gemini_client.py`)

- **Responsibility:** Explain how a candidate compares to a job.
- **Input:** Candidate record and job description.
- **Output:** Score from 0 to 10, strengths, gaps, risks, summary, and evidence.
- **Steps:** Try Gemini only after recruiter opt-in; validate its output and verify that evidence occurs in the resume. If it fails or returns unsupported evidence, use a rule-based skill and experience comparison.
- **Dependencies:** Pydantic and LangChain; optional Gemini API.
- **Failures:** Missing fields are shown as unknown rather than assumed. Gemini failure falls back to the rules.

## Task 5: Data Models

The Pydantic models are in `core/models.py`.

- **Candidate:** ID, name, optional email/phone/location, resume text, skills, and optional years of experience.
- **JobDescription:** ID, title, source text, required and optional skills, optional minimum experience, and location.
- **CandidateEvaluation:** Candidate ID, 0-to-10 fit score, strengths, gaps, risks, summary, evidence, and evaluator name.

The parser only extracts common skills and plainly stated fields. Recruiters should check the source resume; extracted fields are not guaranteed to be complete or correct.

## Task 6: Search Sequence

1. Recruiter adds resumes and pastes a job description.
2. HireFlow identifies common skills, experience, title, and location.
3. Search retrieves candidates with BM25 and, when enabled, Gemini vectors.
4. The recruiter-selected hard filters are applied.
5. The evaluator ranks the remaining candidates and prepares explanations.
6. The UI shows evidence and missing information; the recruiter makes the decision.
7. Searches and review clicks are recorded in `data/memory/`.

## Task 7: Failure Handling

- **Vector or Gemini unavailable:** Fall back to BM25 and rule-based evaluation.
- **Damaged or unreadable resume:** Skip that file, show its error, and process the remaining resumes.
- **No resumes:** Ask the recruiter to add resume files.
- **No search matches:** Suggest broader job wording or opt-in semantic search.
- **Filters remove every result:** Show candidates before filters so the recruiter can relax constraints.
- **Missing candidate details:** Report them as missing; do not guess them.
- **Scanned PDF:** Report extraction failure/empty text. OCR must be added as a separate integration.

## Task 8: Evaluation

`core/evaluator.py` includes dependency-free estimates that can be run on labeled examples:

- **Offline context recall:** How much expected context appears in retrieved text.
- **Offline context precision:** What share of retrieved chunks overlap relevant context.
- **Offline answer correctness:** Token F1 against a known expected answer.
- **Online faithfulness:** How much answer vocabulary appears in its supplied evidence.
- **Online answer relevancy:** How much of the query vocabulary appears in the answer.

These simple metrics are useful for exercises, not a replacement for human review or the full RAGAS framework. If faithfulness is low, first improve retrieval and evidence selection; then tighten the prompt and validate/regenerate the answer. HireFlow already rejects Gemini evidence not found in the resume and falls back to rules.

## Configuration and Privacy

Copy `.env.example` to `.env` to set local options. Never commit `.env` or real candidate documents. Gemini stays disabled until the recruiter checks the opt-in box. The local SQLite embedding cache and interaction log are stored under `data/`.
