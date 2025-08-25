import os
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
from app.ingest import ingest_pdfs
from app.rag import answer_query
from app.config import CHROMA_PATH

st.set_page_config(page_title="PDF RAG", layout="wide")
st.title("📄🔎 PDF Q&A (RAG)")

with st.sidebar:
    st.header("Ingestion")
    uploaded = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Ingest to Vector DB"):
        if not uploaded:
            st.warning("Please upload at least one PDF.")
        else:
            os.makedirs("uploads", exist_ok=True)
            paths = []
            for f in uploaded:
                dest = os.path.join("uploads", f.name)
                with open(dest, "wb") as out:
                    out.write(f.getbuffer())
                paths.append(dest)
            res = ingest_pdfs(paths)
            st.success(f"Ingested {res['chunks_added']} chunks into {CHROMA_PATH}")

st.header("Ask a question")
question = st.text_input("Your question about the uploaded PDFs")
top_k = st.slider("Top K", 1, 10, 5)

if st.button("Ask") and question:
    res = answer_query(question, n_results=top_k)
    st.subheader("Answer")
    st.write(res["answer"])
    st.caption(f"Model: {res['model']}")

    st.subheader("Sources")
    for s in res["sources"]:
        st.write(f"- {s['source']} (page {s['page']}), chunk {s['chunk_index']} — id: {s['id']}")
