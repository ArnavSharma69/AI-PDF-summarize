# AI-PDF-summarize
# PDF Retrieval-Augmented Generation (RAG) — Assignment Solution

This repo implements an AI-powered system that ingests PDF files, indexes them in a vector database, and answers questions using retrieval-augmented generation. It satisfies the assignment's functional/technical requirements, including multi-PDF upload, persistent embeddings, and context-grounded answers.


## Features

- Upload and ingest **multiple PDFs**.
- Extract text with **PyMuPDF**, chunk intelligently, and store **embeddings** in **ChromaDB** (persistent).
- Query UI via **Streamlit** and a **FastAPI** backend endpoint.
- RAG: Convert question to embedding → retrieve top-k chunks → generate answer via **OpenAI** (optional) or context-only fallback.
- Shows **sources** (file name, page, chunk id).


## Tech Stack

- **Backend**: FastAPI
- **Vector DB**: ChromaDB (local, persistent)
- **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`)
- **PDF Parsing**: PyMuPDF
- **UI**: Streamlit
- **Optional LLM**: OpenAI (set `OPENAI_API_KEY` to enable)


## Quickstart (Local)

### 1) Environment
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

### 2) (Optional) Enable OpenAI answers
Edit `.env` and set:
```
OPENAI_API_KEY=sk-...
```

### 3) Start the Streamlit UI
```bash
streamlit run ui/streamlit_app.py
```
Open the URL shown in the terminal. Use the sidebar to upload and ingest PDFs, then ask questions.

### 4) Start the FastAPI server (alternative or in parallel)
```bash
uvicorn app.server:app --reload --port 8000
```
- Ingest via HTTP:
```
curl -F "files=@data/samples/AI-assignment.pdf" http://localhost:8000/ingest
```
- Query via HTTP:
```
curl -X POST -F "q=What is the objective?" -F "top_k=5" http://localhost:8000/query
```

## Project Structure
```
app/
  config.py         # env/config
  ingest.py         # PDF parsing, chunking, embedding, and persistence
  rag.py            # retrieval + LLM (optional) answer generation
  server.py         # FastAPI endpoints
ui/
  streamlit_app.py  # simple UI
data/
  samples/          # sample PDFs (includes assignment)
requirements.txt
.env.example
README.md
```

## Notes & Trade-offs
- ChromaDB is chosen for simple local persistence; can be swapped for Pinecone/FAISS with minor changes.
- Without an `OPENAI_API_KEY`, answers return the best-matching **contexts** (still useful for evaluation).
- Chunking is paragraph-based with sentence fallbacks; tune `max_chars` in `ingest.py` for large PDFs.
- Add tests and Dockerfile as bonus extensions.
```

