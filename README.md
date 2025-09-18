# Databricks Document Chatbot

A Streamlit app to upload documents, build a FAISS vector store with Databricks embeddings, and chat with an LLM (Databricks model serving) over your documents. Includes utilities to list and delete indexed documents and optional evaluation with Ragas.

## Purpose
- **Document ingestion**: Upload PDF/TXT/Markdown and extract text.
- **Vectorization**: Split text with token-aware chunking and generate embeddings using Databricks.
- **Retrieval QA**: Query documents via FAISS-backed retriever and Databricks-hosted LLM.
- **Management**: View indexed files and delete selected or all.
- **Evaluation (optional)**: Ragas-based evaluation using predefined test questions.

## Requirements
- Python 3.10+
- Access to Databricks Model Serving endpoints for:
  - `ChatDatabricks` (e.g., llama-3.3-instruct)
  - `DatabricksEmbeddings` (e.g., bge-large)
- Databricks authentication configured via environment/CLI (e.g., DATABRICKS_HOST, DATABRICKS_TOKEN)

## Installation
```bash
# From the project root
python -m venv venv
# Windows PowerShell
venv\Scripts\Activate.ps1
# Or cmd
# venv\Scripts\activate.bat

pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration
Edit `env.py` as needed:
- `endpoint` for `ChatDatabricks` and `DatabricksEmbeddings`
- `persist_dir` for FAISS storage (default `./storage/faiss`)
Ensure Databricks auth is available via environment or profile.

## Running locally
```bash
# From the project root
streamlit run main.py
```
Then open the Streamlit URL shown in the terminal.

## App navigation
- **Documents**: Upload files, process to chunks, embed, and persist to FAISS.
- **Chatbot**: Ask questions; answers reference the indexed content.
- **Clear Storage**: Select individual files or "all" to remove from the index.

## Storage
FAISS artifacts are stored under `storage/faiss/` (`index.faiss`, `index.pkl`).

## Evaluation (optional)
`evaluation.py` can run Ragas metrics over predefined `test_questions`:
```bash
python evaluation.py
```
It uses the same LLM and embedding instances defined in `env.py`.

## Troubleshooting
- If FAISS index is missing, upload documents first.
- Ensure Databricks credentials are set and endpoints exist.
- On Windows, if activation script is blocked, set execution policy:
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

## License
MIT (or your preferred license).