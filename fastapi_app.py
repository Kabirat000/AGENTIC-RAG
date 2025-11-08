# fastapi_app.py

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, field_validator
import os

from rag import (
    RagPipeline,
    tavily_client,
    groq_client,
    GEN_MODEL_DEFAULT,
    RERANK_MODEL_DEFAULT,
    GROQ_MODEL_DEFAULT,
    keyword_boost,  # imported so fallback ranking works
)

app = FastAPI(title="Agentic RAG API", version="2.1")

# Open CORS for dev; tighten later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# single global pipeline instance
pipeline = RagPipeline(
    gen_model_name=GEN_MODEL_DEFAULT,
    rerank_model_name=RERANK_MODEL_DEFAULT,
    groq_model_name=GROQ_MODEL_DEFAULT,
    tavily=tavily_client,
    groq_client_obj=groq_client,
)

# -------------------------------------------------
# Request / Response models (validation + feedback)
# -------------------------------------------------

class AskPayload(BaseModel):
    q: str = Field(..., min_length=3, description="User question in natural language.")
    source: str | None = None
    k: int = Field(30, ge=1, le=50, description="How many chunks after rerank.")
    overfetch: int = Field(50, ge=1, le=200, description="Initial retrieval size before rerank.")
    use_web: bool | None = None
    reasoner: str | None = None  # None or "gemini" => Gemini, anything else => Groq if available

    @field_validator("overfetch")
    @classmethod
    def ensure_overfetch_not_less_than_k(cls, overfetch_val, info):
        """
        Pydantic v2: info is ValidationInfo, not a dict.
        Ensure overfetch >= k. If user gives overfetch < k, bump it up.
        """
        k_val = 30
        if hasattr(info, "data") and info.data and "k" in info.data:
            k_val = info.data["k"]

        if overfetch_val < k_val:
            return k_val
        return overfetch_val


class AskResponse(BaseModel):
    question: str
    question_rewritten: str
    source_locked: str | None
    k_used: int
    overfetch: int
    used_web: bool
    model_used_for_answer: str
    rerank_model: str | None
    answer: str
    contexts: list[dict]
    citations: dict


# -------------------------
# Routes
# -------------------------

@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    try:
        pipeline.ensure_collection()
        pipeline.qdrant.get_collection(pipeline.COL)
        return {"ok": True, "collection": pipeline.COL}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/config")
def config():
    return {
        "model_default": pipeline.MODEL_NAME,
        "rerank_model": pipeline.RERANK_MODEL,
        "tavily_enabled": pipeline.tavily is not None,
        "collection": pipeline.COL,
        "qdrant_url": "internal",  # don't expose internal URL in prod
        "reasoning_model": pipeline.GROQ_MODEL,
        "reasoning_enabled": pipeline.groq_client is not None,
    }


# ---------------
# /upload (streaming + size cap)
# ---------------

MAX_MB = 10  # max upload size per file
CHUNK_SIZE = 1024 * 1024  # 1 MB chunk write

@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    """
    Upload a single document (pdf, txt, md, docx, html) and save it into ./docs.
    Then ingest just that file into Qdrant so it's immediately searchable.

    Upgrades:
    - We enforce a max file size (MAX_MB).
    - We stream the file to disk in chunks instead of loading into RAM.
    """
    # 1. validate extension
    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()

    allowed_exts = [".pdf", ".txt", ".md", ".docx", ".html", ".htm"]
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"File type {ext} not supported. Allowed: {', '.join(allowed_exts)}"
        )

    # 2. ensure docs dir exists
    os.makedirs("docs", exist_ok=True)

    # 3. build save path
    save_path = os.path.join("docs", filename)

    # 4. stream the uploaded file to disk with size check
    bytes_written = 0
    try:
        with open(save_path, "wb") as buffer:
            while True:
                chunk = file.file.read(CHUNK_SIZE)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_MB * 1024 * 1024:
                    buffer.close()
                    try:
                        os.remove(save_path)
                    except Exception:
                        pass
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (> {MAX_MB} MB)"
                    )
                buffer.write(chunk)
    except HTTPException:
        # Re-raise our friendly 4xx if we hit size cap
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save file: {e}"
        )

    # 5. ingest that single file into the vector DB
    pipeline.ensure_collection()
    pipeline.ingest_single(save_path, batch_size=64)

    return {
        "ok": True,
        "saved_as": save_path,
        "ingested": True,
        "size_mb": round(bytes_written / (1024 * 1024), 2),
    }


# ---------------
# /ingest/docs (bulk re-index of ./docs)
# ---------------

@app.post("/ingest/docs")
def ingest_docs():
    pipeline.ensure_collection()
    pipeline.ingest_docs(batch_size=64)
    return {"ok": True}


# ---------------
# /ask (now with: web debug + no-context guard)
# ---------------

@app.post("/ask", response_model=AskResponse)
def ask(body: AskPayload):
    pipeline.ensure_collection()

    # 1. rewrite query
    q_user = body.q.strip()
    q_rewritten = pipeline.rewrite_query(q_user)

    # 2. optional source lock
    if body.source:
        src_path = pipeline.guess_source_path(body.source)
        if not src_path:
            raise HTTPException(
                status_code=404,
                detail=f"Source not found: {body.source}"
            )
    else:
        src_path = pipeline.guess_source_path(q_user)

    # 3. initial retrieval (+ overfetch for rerank)
    overfetch = max(body.k, body.overfetch)
    try:
        if src_path:
            ctxs = pipeline.retrieve_from_source(q_rewritten, src_path, k=overfetch)
        else:
            ctxs = pipeline.retrieve(q_rewritten, k=overfetch)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    # 4. maybe blend web
    ctxs, used_web_flag = pipeline.maybe_add_web_contexts(
        user_q=q_user,
        rewritten_q=q_rewritten,
        ctxs=ctxs,
        force_web=body.use_web,
    )

    # optional observability: web requested but not added
    if body.use_web is True and used_web_flag is False:
        print("[ASK] Web was explicitly requested but no web snippets were added.")

    # 5. rerank
    try:
        ctxs = pipeline.rerank(q_rewritten, ctxs, top_n=body.k)
    except Exception:
        # fallback order if reranker dies
        ctxs = sorted(
            ctxs,
            key=lambda c: pipeline is not None and keyword_boost(c["text"]),
            reverse=True
        )[:body.k]

    # 6. keyword tiebreak + final trim
    ctxs = sorted(
        ctxs,
        key=lambda c: keyword_boost(c["text"]),
        reverse=True
    )[:body.k]

    # 7. guard: if we still have nothing useful, return 404 instead of hallucinating
    if not ctxs:
        raise HTTPException(
            status_code=404,
            detail="No relevant context found in the knowledge base. You can try use_web=true or upload a document."
        )

    # 8. final answer
    final_text, citation_map, model_used = pipeline.answer(
        q=q_user,
        ctxs=ctxs,
        reasoner=body.reasoner,
    )

    return AskResponse(
        question=q_user,
        question_rewritten=q_rewritten,
        source_locked=src_path,
        k_used=body.k,
        overfetch=overfetch,
        used_web=used_web_flag,
        model_used_for_answer=model_used,
        rerank_model=pipeline.RERANK_MODEL,
        answer=final_text,
        contexts=ctxs,
        citations=citation_map,
    )
