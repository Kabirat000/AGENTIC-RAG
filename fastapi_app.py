# fastapi_app.py
# Minimal API that wraps rag.py (agentic rewrite + source-locked retrieval +
# learned rerank + optional Tavily web blend + DOC/WEB aware answer +
# optional reasoning model escalation)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

import rag  # imports your pipeline; rag.py won't run CLI because __name__ != "__main__"

app = FastAPI(title="Agentic RAG API", version="1.5")

# Open CORS for dev; tighten later for prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskPayload(BaseModel):
    q: str
    source: str | None = None        # optional filename/substring to lock retrieval
    k: int = 30                      # final number after rerank
    overfetch: int = 50              # how many to pull BEFORE rerank
    use_web: bool | None = None      # True = force web, False = block web, None = fallback if docs empty
    reasoner: str | None = None      # None or "gemini" -> Gemini; anything else -> try Groq/deep model

# Redirect "/" -> Swagger UI so home isn't 404
@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    try:
        rag.ensure_collection()
        rag.qdrant.get_collection(rag.COL)
        return {"ok": True, "collection": rag.COL}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# quick config peek
@app.get("/config")
def config():
    return {
        "model_default": getattr(rag, "MODEL_NAME", None),
        "rerank_model": getattr(rag, "RERANK_MODEL", None),
        "tavily_enabled": getattr(rag, "tavily", None) is not None,
        "collection": rag.COL,
        "qdrant_url": rag.QURL,
        # reasoning info so you can show Groq mode is live
        "reasoning_model": getattr(rag, "GROQ_MODEL", None),
        "reasoning_enabled": rag.groq_client is not None,
    }

@app.post("/ingest/docs")
def ingest_docs():
    # Make sure collection exists and then ingest local docs
    rag.ensure_collection()
    rag.ingest_docs()
    return {"ok": True}

@app.post("/ask")
def ask(body: AskPayload):
    # 1) ensure vector DB ready
    rag.ensure_collection()

    # 2) agentic rewrite (query expansion / normalization)
    q_user = body.q
    q_rewritten = rag.rewrite_query(q_user)

    # 3) optional: lock retrieval to a specific source file
    # explicit source param beats heuristic from full question text
    if body.source:
        src_path = rag.guess_source_path(body.source)
        if not src_path:
            raise HTTPException(
                status_code=404,
                detail=f"Source not found: {body.source}"
            )
    else:
        src_path = rag.guess_source_path(q_user)

    # 4) vector retrieval with overfetch so reranker has material
    overfetch = max(body.k, body.overfetch)
    if src_path:
        ctxs = rag.retrieve_from_source(q_rewritten, src_path, k=overfetch)
    else:
        ctxs = rag.retrieve(q_rewritten, k=overfetch)

    # 5) optionally blend in Tavily web snippets
    #    returns (merged_ctxs, used_web_flag)
    ctxs, used_web_flag = rag.maybe_add_web_contexts(
        user_q=q_user,
        rewritten_q=q_rewritten,
        ctxs=ctxs,
        force_web=body.use_web,  # True / False / None
    )

    # 6) learned rerank (Cross-Encoder) -> trim to k
    try:
        ctxs = rag.rerank(q_rewritten, ctxs, top_n=body.k)
    except Exception:
        # fallback ranking if reranker fails (still graceful)
        ctxs = sorted(
            ctxs,
            key=lambda c: rag.keyword_boost(c["text"]),
            reverse=True
        )[:body.k]

    # 7) final keyword bump as a tiebreak, keep top k
    try:
        ctxs = sorted(
            ctxs,
            key=lambda c: rag.keyword_boost(c["text"]),
            reverse=True
        )[:body.k]
    except Exception:
        pass

    # 8) final answer
    # rag.answer returns (final_text, citation_map, model_used)
    final_text, citation_map, model_used = rag.answer(
        q=q_user,
        ctxs=ctxs,
        reasoner=body.reasoner,  # "gemini", "groq", None
    )

    # 9) full structured response for UI / mentor / audit
    return {
        "question": q_user,
        "question_rewritten": q_rewritten,
        "source_locked": src_path,
        "k_used": body.k,
        "overfetch": overfetch,
        "used_web": used_web_flag,
        "model_used_for_answer": model_used,   # which LLM actually produced final_text
        "rerank_model": getattr(rag, "RERANK_MODEL", None),
        "answer": final_text,
        "contexts": ctxs,
        "citations": citation_map,             # maps "1" -> {source, chunk_id}
    }
