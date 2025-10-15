# fastapi_app.py
# Minimal API that wraps your existing rag.py functions.

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse   # <-- added
from pydantic import BaseModel
import rag  # imports your functions; rag.py won't run its CLI because __name__ != "__main__"

app = FastAPI(title="Agentic RAG API", version="1.0")

# Open CORS for dev; tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class AskPayload(BaseModel):
    q: str
    source: str | None = None   # optional filename/URL substring to lock retrieval
    k: int = 50                 # how many chunks when source-locked; generic path uses 15

# -------- Redirect "/" -> "/docs" so home isn't 404 ----------
@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    try:
        rag.qdrant.get_collection(rag.COL)
        return {"ok": True, "collection": rag.COL}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/ingest/docs")
def ingest_docs():
    rag.ensure_collection()
    rag.ingest_docs()
    return {"ok": True}

@app.post("/ask")
def ask(body: AskPayload):
    # 1) ensure ready
    rag.ensure_collection()

    # 2) agentic rewrite
    q_rewritten = rag.rewrite_query(body.q)

    # 3) source lock (explicit beats heuristic)
    src_path = None
    if body.source:
        src_path = rag.guess_source_path(body.source)
        if not src_path:
            raise HTTPException(status_code=404, detail=f"Source not found: {body.source}")
    else:
        src_path = rag.guess_source_path(body.q)

    # 4) retrieve contexts
    if src_path:
        ctxs = rag.retrieve_from_source(q_rewritten, src_path, k=body.k)
    else:
        ctxs = rag.retrieve(q_rewritten, k=15)

    # 5) optional keyword bump (present in your rag.py)
    try:
        ctxs = sorted(ctxs, key=lambda c: rag.keyword_boost(c["text"]), reverse=True)[:body.k]
    except Exception:
        pass

    # 6) answer using your summary-aware prompt
    answer_text = rag.answer(body.q, ctxs)

    return {
        "question": body.q,
        "question_rewritten": q_rewritten,
        "source_locked": src_path,
        "k_used": body.k if src_path else 15,
        "answer": answer_text,
        "contexts": ctxs,   # your UI can show citations from here
    }
