# --- Minimal Agentic RAG (Gemini 2.x + Qdrant) --------------------------------
# Supports .txt AND .pdf in ./docs
# Agentic query rewrite + source-locked retrieval + learned rerank + summary-aware prompt
# + Optional Tavily web-search fallback (keeps same ctx schema) and DOC/WEB tagging
# + Optional reasoning model (e.g. Groq) for final synthesis

import os, glob, hashlib, sys
from typing import List, Dict, Tuple, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
import google.generativeai as genai
from pypdf import PdfReader  # PDF support
from sentence_transformers import CrossEncoder

# -------------------- Settings / Clients --------------------
load_dotenv()

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    print("ERROR: GOOGLE_API_KEY is missing in .env")
    sys.exit(1)

genai.configure(api_key=API_KEY)

# Main LLM for rewrite + fast answering
MODEL_NAME = os.environ.get("GEN_MODEL", "gemini-2.0-flash")

COL = os.environ.get("QDRANT_COLLECTION", "docs")
QURL = os.environ.get("QDRANT_URL", "http://localhost:6333")
qdrant = QdrantClient(url=QURL)

# Optional Tavily web search
TAVILY_KEY = os.environ.get("TAVILY_API_KEY")
tavily = None
if TAVILY_KEY:
    try:
        from tavily import TavilyClient
        tavily = TavilyClient(api_key=TAVILY_KEY)
    except Exception:
        # Keep running even if Tavily isn't installed or import fails
        tavily = None

# Optional deep reasoning model (Groq or similar)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# IMPORTANT: default this to a model you actually tested access for.
# You confirmed access to "llama-3.3-70b-versatile" and "llama-3.1-8b-instant".
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
groq_client = None
if GROQ_API_KEY:
    try:
        # requires `pip install groq`
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception:
        groq_client = None

# Learned reranker (Cross-Encoder)
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
_cross = None
def get_cross():
    """Lazy-load the cross-encoder reranker so startup isn't heavy."""
    global _cross
    if _cross is None:
        _cross = CrossEncoder(RERANK_MODEL)
    return _cross

# --- relevance bump for cash-out / compliance / off-ramp-ish terms -----------
CASHOUT_KEYS = [
    "cash out","cash-out","cashout","withdraw","withdrawal","sell","fiat",
    "off-ramp","off ramp","bank","p2p","peer-to-peer","escrow","atm",
    "bitcoin atm","debit card","card","limit","limits","fee","fees",
    "commission","kyc","aml","verification","identity","compliance"
]

def keyword_boost(text: str) -> int:
    """
    Quick heuristic bump for domain-critical words.
    Returns how many of those keywords appear. Higher -> more boost.
    """
    t = text.lower()
    return sum(k in t for k in CASHOUT_KEYS)

# -------------------- Qdrant helpers -----------------------------------------
def ensure_collection():
    """
    Make sure the target collection exists with the correct vector params.
    Safe to call multiple times.
    """
    try:
        qdrant.get_collection(COL)
    except Exception:
        qdrant.recreate_collection(
            collection_name=COL,
            vectors_config=qm.VectorParams(
                size=768,
                distance=qm.Distance.COSINE
            ),
        )
        print(f"Created collection '{COL}'")

# -------------------- Embedding helper ---------------------------------------
def embed(text: str):
    """
    Embed text using Gemini embeddings, returns list[float] of size 768.
    """
    return genai.embed_content(
        model="text-embedding-004",
        content=text
    )["embedding"]

# -------------------- Agentic query rewrite ----------------------------------
def rewrite_query(q: str, model_name: Optional[str] = None) -> str:
    """
    Ask Gemini to rewrite the query to be self-contained, specific, keyword-rich.
    Fallback: return q unchanged if model call fails.
    """
    model_name = model_name or MODEL_NAME
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

# -------------------- PDF text extractor -------------------------------------
def extract_pdf_text(path: str) -> str:
    """
    Extract raw text from all pages of a PDF.
    """
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()

# -------------------- Chunking + Ingestion -----------------------------------
def _chunk_text(t: str, size: int = 300, overlap: int = 60) -> List[str]:
    """
    Fixed-size sliding window chunker with overlap.
    Returns list[str] chunks.
    """
    chunks, i = [], 0
    step = max(1, size - overlap)
    while i < len(t):
        chunk = t[i:i+size]
        if chunk.strip():
            chunks.append(chunk)
        i += step
    return chunks

def ingest_docs():
    """
    Walk ./docs, read .txt/.pdf, chunk, embed, and upsert into Qdrant.
    Creates the collection if missing.
    """
    ensure_collection()

    paths = [*glob.glob("docs/*.txt"), *glob.glob("docs/*.pdf")]
    if not paths:
        print("NOTE: No .txt or .pdf files found in ./docs — add some and re-run.")
        return

    for path in paths:
        if path.lower().endswith(".pdf"):
            txt = extract_pdf_text(path)
        else:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read()

        if not txt.strip():
            print(f"Skipped (no extractable text): {path}")
            continue

        chunks = _chunk_text(txt, size=300, overlap=60)

        # deterministic base id per file for stable point IDs
        base_id = int(hashlib.md5(path.encode()).hexdigest()[:8], 16)

        points = []
        for idx, ch in enumerate(chunks):
            pid = base_id + idx  # unique-ish id derived from file hash + chunk index
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

# -------------------- Retrieval utilities ------------------------------------
def list_all_sources(max_points: int = 10000):
    """
    Return a set of unique 'source' payload values in the collection.
    This lets us later 'guess' which file the user meant.
    """
    ensure_collection()
    sources = set()
    offset = None
    while True:
        res = qdrant.scroll(
            collection_name=COL,
            with_payload=True,
            limit=min(2048, max_points),
            offset=offset,
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
    """
    Try to match a user-provided title fragment to a known source path via substring.
    Returns the first match or None.
    """
    if not title_like:
        return None
    tl = title_like.lower().strip().replace('"', '')
    best = None
    for src in list_all_sources():
        if tl in src.lower():
            best = src
            break
    return best

def retrieve(query: str, k: int = 5):
    """
    Vector search across the entire collection.
    Returns a list of dicts: {text, source, chunk_id}
    """
    ensure_collection()
    res = qdrant.query_points(
        collection_name=COL,
        query=embed(query),
        limit=k,
        with_payload=True,
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

def retrieve_from_source(query: str, source_path: str, k: int = 30):
    """
    Retrieve top-k chunks but only from a given source path (file).
    We over-fetch, then filter to that specific source, then dedupe similar chunks.
    """
    ensure_collection()
    res = qdrant.query_points(
        collection_name=COL,
        query=embed(query),
        limit=k * 3,  # over-fetch
        with_payload=True,
    )
    hits = res.points or []
    filt = [h for h in hits if (h.payload or {}).get("source") == source_path]

    # Deduplicate near-identical texts using md5
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

# -------------------- Web search retrieval -----------------------------------
def web_search_snippets(q: str, k: int = 5):
    """
    Return web snippets in the exact same schema as doc chunks.
    Only works if Tavily is configured. Otherwise returns [].
    """
    if not tavily:
        return []
    try:
        res = tavily.search(q, max_results=k)
    except Exception:
        return []
    out = []
    for item in res.get("results", []):
        out.append({
            "text": (item.get("content") or item.get("snippet") or "")[:1000],
            "source": item.get("url") or "web",
            "chunk_id": None,
        })
    return out

def maybe_add_web_contexts(
    user_q: str,
    rewritten_q: str,
    ctxs: list,
    force_web: Optional[bool],
) -> Tuple[list, bool]:
    """
    Decide whether to blend in Tavily web snippets.
    - If force_web is True, always fetch web and merge.
    - If force_web is False, never fetch web.
    - If force_web is None, fetch web only if ctxs is empty.
    Returns (merged_ctxs, used_web_flag).
    """
    used_web = False
    # No Tavily client? nothing to add
    if not tavily:
        return ctxs, used_web

    wants_web = False
    if force_web is True:
        wants_web = True
    elif force_web is None and not ctxs:
        wants_web = True

    if wants_web:
        web_ctxs = web_search_snippets(rewritten_q, k=5)
        if web_ctxs:
            ctxs = (ctxs or []) + web_ctxs
            used_web = True

    return ctxs, used_web

# -------------------- Learned reranking --------------------------------------
def rerank(query: str, ctxs: list, top_n: int = 30):
    """
    Learned rerank using CrossEncoder; returns top_n most relevant chunks.
    """
    if not ctxs:
        return []
    model = get_cross()
    pairs = [(query, c["text"]) for c in ctxs]
    scores = model.predict(pairs)  # higher is better
    ranked = sorted(zip(ctxs, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_n]]

# -------------------- Prompt building + Answer generation --------------------
def _tag_contexts_for_prompt(ctxs: List[Dict]) -> Tuple[str, Dict[str, Dict]]:
    """
    Build:
      1. tagged context block that we feed to the LLM
      2. citation_map so we know which [n] -> which source
    """
    tagged_lines = []
    citation_map: Dict[str, Dict] = {}

    for i, c in enumerate(ctxs):
        cite_id = i + 1  # [1], [2], ...
        src = str(c.get("source", ""))
        label = "DOC" if src.lower().startswith("docs") else "WEB"

        tagged_lines.append(
            f"[{cite_id}][{label}] (chunk {c.get('chunk_id')}) {c['text']} (source: {src})"
        )

        citation_map[str(cite_id)] = {
            "source": src,
            "chunk_id": c.get("chunk_id"),
        }

    full_block = "\n\n".join(tagged_lines)
    return full_block, citation_map

def _build_answer_prompt(
    user_question: str,
    ctxs: List[Dict],
) -> Tuple[Optional[str], Dict[str, Dict], Optional[bool]]:
    """
    Make the final prompt string for the LLM, and return a citation map.
    We keep the DOC > WEB safety rule here.
    """
    if not ctxs:
        return None, {}, None

    context_block, citation_map = _tag_contexts_for_prompt(ctxs)

    q_lower = user_question.lower()
    wants_summary = any(
        w in q_lower
        for w in ["summary", "summarize", "summarise", "overview", "bullet", "bullets"]
    )

    if wants_summary:
        instruct = (
            "Use only the CONTEXT. Prefer DOC over WEB if there is any conflict. "
            "Produce exactly 5 concise bullet points that directly answer the QUESTION. "
            "Cover: main process / steps, key requirements or constraints, "
            "risks / limits / fees if relevant, and any compliance or legal notes mentioned. "
            "Do not invent facts. Always cite sources as [n]."
        )
    else:
        instruct = (
            "Use only the CONTEXT. Prefer DOC over WEB if there is any conflict. "
            "Answer the QUESTION precisely and briefly. "
            "If something is not stated in CONTEXT, say you don't know. "
            "Always cite sources as [n]."
        )

    prompt = (
        f"{instruct}\n\n"
        f"QUESTION:\n{user_question}\n\n"
        f"CONTEXT:\n{context_block}"
    )

    return prompt, citation_map, wants_summary

def _generate_with_gemini(prompt: str) -> str:
    """
    Call Gemini (MODEL_NAME) to generate final answer text.
    """
    try:
        return genai.GenerativeModel(MODEL_NAME).generate_content(prompt).text
    except Exception as e:
        return f"Model error with {MODEL_NAME}: {e}"

def _generate_with_reasoner(prompt: str) -> str:
    """
    Try to use the Groq reasoning model for final synthesis.
    If Groq is unavailable or errors out, silently fall back to Gemini
    without exposing debug text to the end user.
    """
    if groq_client is None:
        # no groq, just do gemini
        return _generate_with_gemini(prompt)

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=800,
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        # silent fallback to Gemini (no "[fallback to Gemini]" prefix)
        return _generate_with_gemini(prompt)

def answer(
    q: str,
    ctxs: List[Dict],
    reasoner: Optional[str] = None
) -> Tuple[str, Dict[str, Dict], str]:
    """
    Top-level answer function.
    reasoner:
      None or "gemini"  -> use Gemini for synthesis
      anything else     -> try GROQ_MODEL for synthesis
    Returns (final_text, citation_map, model_used)
    """
    if not ctxs:
        return (
            "I couldn’t find anything in the provided context.",
            {},
            MODEL_NAME,
        )

    prompt, citation_map, _ = _build_answer_prompt(q, ctxs)

    if prompt is None:
        return (
            "I couldn’t find anything in the provided context.",
            {},
            MODEL_NAME,
        )

    # choose which brain writes final answer
    if reasoner is None or (isinstance(reasoner, str) and reasoner.lower() == "gemini"):
        final_text = _generate_with_gemini(prompt)
        model_used = MODEL_NAME
    else:
        final_text = _generate_with_reasoner(prompt)
        model_used = GROQ_MODEL if groq_client is not None else MODEL_NAME

    return final_text, citation_map, model_used

# -------------------- CLI flow (manual test) ---------------------------------
if __name__ == "__main__":
    # 1. Make sure collection exists and ingest docs locally
    ensure_collection()
    ingest_docs()

    # 2. Ask interactively
    q = input("\nAsk a question about your docs (or ask something that may need web): ")

    # 3. Agentic rewrite before retrieval
    q_rewritten = rewrite_query(q)
    if q_rewritten != q:
        print(f"\nRewritten query → {q_rewritten}")

    # 4. Try to lock retrieval to a specific source (if user mentioned filename-ish text)
    src_path = guess_source_path(q)

    # 5. Vector retrieval with overfetch-style behaviour
    if src_path:
        print(f"Locking retrieval to: {src_path}")
        ctxs = retrieve_from_source(q_rewritten, src_path, k=50)
    else:
        ctxs = retrieve(q_rewritten, k=50)

    # 6. Blend web if needed (CLI default: auto fallback if no docs)
    ctxs, used_web_flag = maybe_add_web_contexts(
        user_q=q,
        rewritten_q=q_rewritten,
        ctxs=ctxs,
        force_web=None  # None = only use web if local ctxs is empty
    )
    if used_web_flag:
        print("Added web snippets from Tavily.")

    # 7. Learned rerank for relevance
    ctxs = rerank(q_rewritten, ctxs, top_n=30)

    # 8. Light keyword bump as final tiebreaker, keep top 30
    ctxs = sorted(
        ctxs,
        key=lambda c: keyword_boost(c["text"]),
        reverse=True
    )[:30]

    # Debug: show what came back
    print("\nTop ranked sources:")
    for i, c in enumerate(ctxs, 1):
        prev = c["text"].replace("\n", " ")[:120]
        print(f"{i}. {c['source']} (chunk {c.get('chunk_id')}) -> {prev}...")

    # 9. Final answer (CLI default reasoner=None -> Gemini)
    final_text, citation_map, used_model = answer(q, ctxs, reasoner=None)

    print("\nModel used:", used_model)
    print("Citations:", citation_map)
    print("\nAnswer:\n" + final_text)
