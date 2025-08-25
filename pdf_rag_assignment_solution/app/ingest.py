from typing import List, Tuple, Dict, Any
import os
import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
from .config import CHROMA_PATH, EMBEDDING_MODEL

def extract_text_from_pdf(path: str) -> List[Tuple[int, str]]:
    """Return list of (page_number, text)."""
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text and text.strip():
            pages.append((i+1, text))
    doc.close()
    return pages

def chunk_text(text: str, max_chars: int = 1200) -> List[str]:
    """Simple paragraph-ish chunking with length cap."""
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    for p in parts:
        if len(p) <= max_chars:
            chunks.append(p)
        else:
            # further split on sentences
            sentences = [s.strip() for s in p.replace("\n", " ").split(". ") if s.strip()]
            buf = ""
            for s in sentences:
                s2 = (s + ".").strip()
                if len(buf) + 1 + len(s2) <= max_chars:
                    buf = (buf + " " + s2).strip()
                else:
                    if buf:
                        chunks.append(buf)
                    buf = s2
            if buf:
                chunks.append(buf)
    return chunks

def get_collection(collection_name: str = "docs"):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return client.get_or_create_collection(name=collection_name, embedding_function=emb_fn)

def ingest_pdfs(paths: List[str], collection_name: str = "docs") -> Dict[str, Any]:
    os.makedirs(CHROMA_PATH, exist_ok=True)
    col = get_collection(collection_name)
    total_chunks = 0
    added_ids = []

    for pdf_path in paths:
        pages = extract_text_from_pdf(pdf_path)
        for page_num, text in pages:
            chunks = chunk_text(text)
            for idx, chunk in enumerate(chunks):
                uid = f"{os.path.basename(pdf_path)}::p{page_num}::c{idx}"
                metadata = {
                    "source": os.path.basename(pdf_path),
                    "page": page_num,
                    "chunk_index": idx
                }
                col.add(documents=[chunk], metadatas=[metadata], ids=[uid])
                total_chunks += 1
                added_ids.append(uid)
    return {"chunks_added": total_chunks, "ids": added_ids}
