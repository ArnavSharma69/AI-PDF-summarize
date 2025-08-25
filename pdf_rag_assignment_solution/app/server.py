from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List
import os
from .ingest import ingest_pdfs
from .rag import answer_query

app = FastAPI(title="PDF RAG Server")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    paths = []
    os.makedirs("uploads", exist_ok=True)
    for f in files:
        dest = os.path.join("uploads", f.filename)
        with open(dest, "wb") as out:
            out.write(await f.read())
        paths.append(dest)
    result = ingest_pdfs(paths)
    return JSONResponse(result)

@app.post("/query")
async def query(q: str = Form(...), top_k: int = Form(5)):
    result = answer_query(q, n_results=top_k)
    return JSONResponse(result)
