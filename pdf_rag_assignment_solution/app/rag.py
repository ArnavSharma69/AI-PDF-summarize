from typing import Dict, Any, List
import os
import chromadb
from chromadb.utils import embedding_functions
from .config import CHROMA_PATH, EMBEDDING_MODEL, OPENAI_API_KEY

def get_collection(collection_name: str = "docs"):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return client.get_or_create_collection(name=collection_name, embedding_function=emb_fn)

SYSTEM_PROMPT = (
    "You are a helpful assistant answering ONLY from the provided context. "
    "If the answer is not in the context, say you don't know."
)

def build_prompt(question: str, contexts: List[str]) -> str:
    header = SYSTEM_PROMPT + "\n\nContext:\n"
    ctx = "\n---\n".join(contexts)
    return f"{header}{ctx}\n\nQuestion: {question}\nAnswer:"

def answer_query(question: str, n_results: int = 5, collection_name: str = "docs") -> Dict[str, Any]:
    col = get_collection(collection_name)
    res = col.query(query_texts=[question], n_results=n_results)
    docs = res.get("documents", [[]])[0]
    metadatas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]

    prompt = build_prompt(question, docs)

    answer_text = None
    used_model = None
    if OPENAI_API_KEY:
        try:
            # OpenAI SDK (>=1.30) with Responses API
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )
            answer_text = completion.choices[0].message.content
            used_model = "openai:gpt-4o-mini"
        except Exception as e:
            answer_text = None
            used_model = f"openai_error:{e}"

    if not answer_text:
        # Fallback: return concatenated top contexts as the 'answer'
        answer_text = "Context-only answer (no LLM configured):\n\n" + "\n\n---\n\n".join(docs)
        used_model = used_model or "context-only"

    sources = []
    for m, i in zip(metadatas, ids):
        sources.append({
            "id": i,
            "source": m.get("source"),
            "page": m.get("page"),
            "chunk_index": m.get("chunk_index")
        })
    return {"answer": answer_text, "sources": sources, "model": used_model}
