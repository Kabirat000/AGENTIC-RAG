# --- Minimal Agentic RAG (Gemini 2.x + Qdrant) --------------------------------
# Supports .txt AND .pdf in ./docs
# Agentic query rewrite + source-locked retrieval + keyword rerank + summary-aware prompt

import os, glob, hashlib, sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
import google.generativeai as genai
from pypdf import PdfReader  # PDF support

# --- relevance bump for cash-out topics (and general off-ramp terms)
CASHOUT_KEYS = [
    "cash out","cash-out","cashout","withdraw","withdrawal","sell","fiat","off-ramp","off ramp",
    "bank","p2p","peer-to-peer","escrow","atm","bitcoin atm","debit card","card",
    "limit","limits","fee","fees","commission","kyc","aml","verification","identity","compliance"
]
def keyword_boost(text: str) -> int:
    t = text.lower()
    return sum(k in t for k in CASHOUT_KEYS)

# -------------------- 0) Load settings & init clients -------------------------
load_dotenv()

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    print("ERROR: GOOGLE_API_KEY is missing in .env"); sys.exit(1)
genai.configure(api_key=API_KEY)

COL = os.environ.get("QDRANT_COLLECTION", "docs")
QURL = os.environ.get("QDRANT_URL", "http://localhost:6333")
qdrant = QdrantClient(url=QURL)

# -------------------- 1) Ensure collection exists -----------------------------
def ensure_collection():
    try:
        qdrant.get_collection(COL)
    except Exception:
        qdrant.recreate_collection(
            collection_name=COL,
            vectors_config=qm.VectorParams(size=768, distance=qm.Distance.COSINE)
        )
        print(f"Created collection '{COL}'")

# -------------------- 2) Embedding helper ------------------------------------
def embed(text: str):
    return genai.embed_content(model="text-embedding-004", content=text)["embedding"]

# -------------------- 2.5) Agentic step: rewrite vague queries ----------------
def rewrite_query(q: str, model_name: str = "gemini-2.0-flash"):
    prompt = (
        "You are a query rewriter for document search. "
        "Rewrite the user query to be concise, specific, and self-contained. "
        "Remove pronouns, add obvious keywords, keep it one short line.\n"
        f"USER QUERY: {q}"
    )
    try:
        return genai.GenerativeModel(model_name).generate_content(prompt).text.strip()
    except Exception:
        return q  # graceful fallback

# -------------------- 2.7) PDF text extractor --------------------------------
def extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()

# -------------------- 3) Ingest docs as CHUNKS (.txt + .pdf) ------------------
def ingest_docs():
    # fixed-size chunker with overlap
    def chunk_text(t: str, size: int = 300, overlap: int = 60):
        chunks, i = [], 0
        step = max(1, size - overlap)
        while i < len(t):
            chunk = t[i:i+size]
            if chunk.strip():
                chunks.append(chunk)
            i += step
        return chunks

    any_files = False
    paths = [*glob.glob("docs/*.txt"), *glob.glob("docs/*.pdf")]
    for path in paths:
        any_files = True

        if path.lower().endswith(".pdf"):
            txt = extract_pdf_text(path)
        else:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read()

        if not txt.strip():
            print(f"Skipped (no extractable text): {path}")
            continue

        chunks = chunk_text(txt, size=300, overlap=60)
        base_id = int(hashlib.md5(path.encode()).hexdigest()[:8], 16)

        points = []
        for idx, ch in enumerate(chunks):
            pid = base_id + idx  # stable unique id per (file, chunk)
            points.append(
                qm.PointStruct(
                    id=pid,
                    vector=embed(ch),
                    payload={
                        "text": ch,
                        "source": path,
                        "chunk_id": idx,
                        "total_chunks": len(chunks),
                        "ext": os.path.splitext(path)[1].lower(),
                    },
                )
            )

        if points:
            qdrant.upsert(collection_name=COL, points=points)
            print(f"Ingested {len(points)} chunks from: {path}")

    if not any_files:
        print("NOTE: No .txt or .pdf files found in ./docs — add some and re-run.")

# -------------------- 4) Retrieval (generic) ----------------------------------
def retrieve(query: str, k: int = 5):
    res = qdrant.query_points(
        collection_name=COL,
        query=embed(query),
        limit=k,
        with_payload=True
    )
    hits = res.points or []
    return [
        {
            "text": h.payload["text"],
            "source": h.payload.get("source"),
            "chunk_id": h.payload.get("chunk_id"),
        }
        for h in hits
    ]

# -------- 4b) Utilities: list sources & lock retrieval to a chosen source -----
def list_all_sources(max_points: int = 10000):
    """Return a set of unique 'source' payload values in the collection."""
    sources = set()
    offset = None
    while True:
        res = qdrant.scroll(
            collection_name=COL,
            with_payload=True,
            limit=min(2048, max_points),
            offset=offset
        )
        points, next_page_offset = res
        for p in points or []:
            src = (p.payload or {}).get("source")
            if src:
                sources.add(src)
        if not next_page_offset:
            break
        offset = next_page_offset
    return sources

def guess_source_path(title_like: str):
    """Try to match a user-provided title to a known source path (substring match)."""
    if not title_like:
        return None
    tl = title_like.lower().strip().replace('"', '')
    best = None
    for src in list_all_sources():
        if tl in src.lower():
            best = src
            break
    return best

def retrieve_from_source(query: str, source_path: str, k: int = 30):
    """Retrieve top-k chunks but only from a given source path."""
    res = qdrant.query_points(
        collection_name=COL,
        query=embed(query),
        limit=k * 3,  # over-fetch
        with_payload=True
    )
    hits = res.points or []
    filt = [h for h in hits if (h.payload or {}).get("source") == source_path]

    # Deduplicate near-identical texts (simple hash)
    seen, picked = set(), []
    for h in filt:
        txt = (h.payload or {}).get("text", "")
        key = hashlib.md5(txt.strip().encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        picked.append({
            "text": txt,
            "source": source_path,
            "chunk_id": (h.payload or {}).get("chunk_id"),
        })
        if len(picked) >= k:
            break
    return picked

# -------------------- 5) Pick a supported Gemini model (2.x) ------------------
PREFS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash-thinking-exp",
    "gemini-2.0-flash-lite",
    "gemini-flash-latest",
    "gemini-pro-latest",
]
def pick_model():
    try:
        models = [m for m in genai.list_models()
                  if "generateContent" in getattr(m, "supported_generation_methods", [])]
        names = set(m.name.replace("models/", "") for m in models)
        for p in PREFS:
            if p in names:
                return p
        return next(iter(names)) if names else None
    except Exception as e:
        print(f"Could not list models: {e}")
        return None

# -------------------- 6) Answer with Gemini ----------------------------------
def answer(q: str, ctxs: list):
    if not ctxs:
        return "I couldn’t find anything in the docs related to your question."

    ctx = "\n\n".join(
        f"[{i+1}] (chunk {c.get('chunk_id')}) {c['text']} (source: {c['source']})"
        for i, c in enumerate(ctxs)
    )

    q_lower = q.lower()
    wants_summary = any(w in q_lower for w in ["summary", "summarize", "overview", "bullet"])

    if wants_summary:
        instruct = (
            "From the CONTEXT, produce exactly 5 concise bullets that summarize HOW TO CASH OUT CRYPTOCURRENCY. "
            "Focus on: (1) cash-out methods (CEX, P2P, Bitcoin ATMs, crypto cards), "
            "(2) step-by-step flow, (3) fees/limits, (4) risks/scams & safety, "
            "(5) KYC/AML or legal considerations. Do not invent facts. Cite chunks like [1], [2]."
        )
    else:
        instruct = (
            "Use only the CONTEXT to answer. Be precise and brief. "
            "If the answer is not in CONTEXT, say you don't know. Cite chunks like [1], [2]."
        )

    prompt = f"{instruct}\n\nQ: {q}\n\nCONTEXT:\n{ctx}"

    model_name = pick_model()
    if not model_name:
        return "No generation models available for your API key. Check Google AI Studio access."

    try:
        return genai.GenerativeModel(model_name).generate_content(prompt).text
    except Exception as e:
        return f"Model error with {model_name}: {e}"

# -------------------- 7) CLI flow --------------------------------------------
if __name__ == "__main__":
    ensure_collection()
    ingest_docs()

    q = input("\nAsk a question about your docs: ")

    # agentic rewrite before retrieval
    q_rewritten = rewrite_query(q)
    if q_rewritten != q:
        print(f"\nRewritten query → {q_rewritten}")

    # try to lock retrieval to a specific source (title/file mentioned)
    source_hint = q
    src_path = guess_source_path(source_hint)

    if src_path:
        print(f"Locking retrieval to: {src_path}")
        ctxs = retrieve_from_source(q_rewritten, src_path, k=50)  # larger k for summaries
    else:
        ctxs = retrieve(q_rewritten, k=15)

    # keyword-based rerank to prioritize cash-out related chunks
    ctxs = sorted(ctxs, key=lambda c: keyword_boost(c["text"]), reverse=True)[:50]

    # Debug: show what came back
    print("\nTop sources:")
    for i, c in enumerate(ctxs, 1):
        prev = c["text"].replace("\n", " ")[:120]
        print(f"{i}. {c['source']} (chunk {c.get('chunk_id')}) -> {prev}...")

    print("\nAnswer:\n" + answer(q, ctxs))
