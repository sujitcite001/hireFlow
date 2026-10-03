# HireFlow Project Instructions

- Keep resume files and interaction data local under `data/`; never commit user documents.
- Keep ingestion, parsing, indexing, ranking, filtering, and UI logic in their respective modules.
- Do not require network services or credentials for local development.
- Run `python -m compileall -q .` after Python changes and `streamlit run streamlit/app.py` to launch the UI.
