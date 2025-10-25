# fastapi_app.py
# Minimal API that wraps your rag.py functions (learned rerank + optional web).

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import rag  # imports your functions; rag.py won't run its CLI because __name__ != "__main__"

app = FastAPI(title="Agentic RAG API", version="1.2")

# Open CORS for dev; tighten in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class AskPayload(BaseModel):
    q: str
    source: str | None = None   # optional filename/substring to lock retrieval
    k: int = 30                 # final number after rerank
    overfetch: int = 50         # how many to fetch BEFORE rerank
    use_web: bool | None = None # if you add Tavily in rag.py: force web search when True

# Redirect "/" -> Swagger UI so home isn't 404
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

# (optional) quick config peek
@app.get("/config")
def config():
    return {
        "model": getattr(rag, "MODEL_NAME", None),
        "rerank_model": getattr(rag, "RERANK_MODEL", None),
        "tavily_enabled": getattr(rag, "tavily", None) is not None,
        "collection": rag.COL,
        "qdrant_url": rag.QURL,
    }

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
    if body.source:
        src_path = rag.guess_source_path(body.source)
        if not src_path:
            raise HTTPException(status_code=404, detail=f"Source not found: {body.source}")
    else:
        src_path = rag.guess_source_path(body.q)

    # 4) retrieve (overfetch for better reranking)
    overfetch = max(body.k, body.overfetch)
    if src_path:
        ctxs = rag.retrieve_from_source(q_rewritten, src_path, k=overfetch)
    else:
        ctxs = rag.retrieve(q_rewritten, k=overfetch)

    # 4.5) optional web fallback/force if you later add Tavily in rag.py
    used_web = False
    tavily = getattr(rag, "tavily", None)
    if tavily is not None:
        wants_web = (body.use_web is True) or (body.use_web is None and not ctxs)
        if wants_web:
            web_ctxs_fn = getattr(rag, "web_search_snippets", None)
            if callable(web_ctxs_fn):
                web_ctxs = web_ctxs_fn(q_rewritten, k=5)
                if web_ctxs:
                    ctxs = (ctxs or []) + web_ctxs
                    used_web = True

    # 5) learned rerank (Cross-Encoder) -> trim to k
    try:
        ctxs = rag.rerank(q_rewritten, ctxs, top_n=body.k)
    except Exception:
        # fallback: keyword bump then trim
        ctxs = sorted(ctxs, key=lambda c: rag.keyword_boost(c["text"]), reverse=True)[:body.k]

    # 6) light keyword bump as tiebreaker (keep top k)
    try:
        ctxs = sorted(ctxs, key=lambda c: rag.keyword_boost(c["text"]), reverse=True)[:body.k]
    except Exception:
        pass

    # 7) answer with your summary-aware prompt
    answer_text = rag.answer(body.q, ctxs)

    return {
        "question": body.q,
        "question_rewritten": q_rewritten,
        "source_locked": src_path,
        "k_used": body.k,
        "overfetch": overfetch,
        "used_web": used_web,
        "model": getattr(rag, "MODEL_NAME", None),
        "rerank_model": getattr(rag, "RERANK_MODEL", None),
        "answer": answer_text,
        "contexts": ctxs,
    }
