# rag.py
# Agentic Hybrid RAG Pipeline (class-based)
# - batched ingestion with logging + error handling
# - multi-format loaders (.txt, .pdf, .md, .docx, .html)
# - agentic rewrite, per-source retrieval, rerank, keyword boost
# - optional Tavily web fallback, optional Groq reasoning

import os, glob, hashlib, sys, logging
from typing import List, Dict, Tuple, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
import google.generativeai as genai
from pypdf import PdfReader
from sentence_transformers import CrossEncoder

# optional deps
try:
    import docx  # python-docx
except Exception:
    docx = None

try:
    from bs4 import BeautifulSoup  # beautifulsoup4
except Exception:
    BeautifulSoup = None

# -------------------------------------------------------------------
# logging setup (mentor request: error logging instead of silent fail)
# -------------------------------------------------------------------
logger = logging.getLogger("rag")
logger.setLevel(logging.INFO)

_stream = logging.StreamHandler()
_stream.setLevel(logging.INFO)
_fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
_stream.setFormatter(_fmt)

if not logger.handlers:
    logger.addHandler(_stream)

# -------------------------------------------------------------------
# env / global service clients
# -------------------------------------------------------------------
load_dotenv()

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    logger.error("GOOGLE_API_KEY is missing in .env")
    sys.exit(1)

genai.configure(api_key=API_KEY)

GEN_MODEL_DEFAULT = os.environ.get("GEN_MODEL", "gemini-2.0-flash")

QDRANT_COLLECTION_DEFAULT = os.environ.get("QDRANT_COLLECTION", "docs")
QDRANT_URL_DEFAULT = os.environ.get("QDRANT_URL", "http://localhost:6333")

# Optional Tavily web search
TAVILY_KEY = os.environ.get("TAVILY_API_KEY")
tavily_client = None
if TAVILY_KEY:
    try:
        from tavily import TavilyClient
        tavily_client = TavilyClient(api_key=TAVILY_KEY)
    except Exception as e:
        logger.warning(f"Could not init TavilyClient: {e}")
        tavily_client = None

# Optional Groq reasoning model
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_DEFAULT = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
groq_client = None
if GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        logger.warning(f"Could not init Groq client: {e}")
        groq_client = None

# reranker model name
RERANK_MODEL_DEFAULT = os.environ.get(
    "RERANK_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# keywords for boosting compliance / withdrawal style questions
CASHOUT_KEYS = [
    "cash out","cash-out","cashout","withdraw","withdrawal","sell","fiat",
    "off-ramp","off ramp","bank","p2p","peer-to-peer","escrow","atm",
    "bitcoin atm","debit card","card","limit","limits","fee","fees",
    "commission","kyc","aml","verification","identity","compliance"
]


def keyword_boost(text: str) -> int:
    t = text.lower()
    return sum(k in t for k in CASHOUT_KEYS)


# -------------------------------------------------------------------
# Helper functions that don't need pipeline state
# -------------------------------------------------------------------

def _chunk_text(t: str, size: int = 300, overlap: int = 60) -> List[str]:
    """
    Sliding window chunker with overlap.
    ~300 chars with 60 char overlap to preserve context continuity.
    """
    chunks, i = [], 0
    step = max(1, size - overlap)  # default step = 240
    while i < len(t):
        chunk = t[i:i+size]
        if chunk.strip():
            chunks.append(chunk)
        i += step
    return chunks


def _yield_batches(items: list, batch_size: int = 64):
    """
    Yield (sublist, start_index, end_index)
    for batched embedding + upsert.
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size], i, min(i + batch_size, len(items))


# -------------------------------------------------------------------
# RagPipeline class (mentor request: modularize into a class)
# -------------------------------------------------------------------

class RagPipeline:
    """
    End-to-end Agentic RAG:
    - ingest (batch, logged)
    - rewrite
    - retrieve (+ source lock)
    - web fallback
    - rerank
    - answer (Gemini or Groq)
    """

    SUPPORTED_EXTS = [".txt", ".pdf", ".md", ".docx", ".html", ".htm"]

    def __init__(
        self,
        qdrant_url: str = QDRANT_URL_DEFAULT,
        collection_name: str = QDRANT_COLLECTION_DEFAULT,
        gen_model_name: str = GEN_MODEL_DEFAULT,
        rerank_model_name: str = RERANK_MODEL_DEFAULT,
        groq_model_name: str = GROQ_MODEL_DEFAULT,
        tavily=None,
        groq_client_obj=None,
    ):
        self.COL = collection_name
        self.MODEL_NAME = gen_model_name
        self.RERANK_MODEL = rerank_model_name
        self.GROQ_MODEL = groq_model_name

        self.qdrant = QdrantClient(url=qdrant_url)
        self.tavily = tavily
        self.groq_client = groq_client_obj

        self._cross = None  # lazy CrossEncoder

    # ---------------- Qdrant collection mgmt -----------------
    def ensure_collection(self):
        try:
            self.qdrant.get_collection(self.COL)
        except Exception:
            self.qdrant.recreate_collection(
                collection_name=self.COL,
                vectors_config=qm.VectorParams(size=768, distance=qm.Distance.COSINE),
            )
            logger.info(f"Created collection '{self.COL}'")

    # ---------------- Embedding ------------------------------
    def embed(self, text: str) -> List[float]:
        """
        Uses Gemini embeddings (text-embedding-004). Returns 768-dim vector.
        Fail safe: return [] if something explodes.
        """
        try:
            return genai.embed_content(
                model="text-embedding-004",
                content=text
            )["embedding"]
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return []

    # ---------------- Query rewrite --------------------------
    def rewrite_query(self, q: str) -> str:
        prompt = (
            "You are a query rewriter for document search. "
            "Rewrite the user query to be concise, specific, and self-contained. "
            "Remove pronouns, add obvious keywords, keep it one short line.\n"
            f"USER QUERY: {q}"
        )
        try:
            return genai.GenerativeModel(self.MODEL_NAME) \
                        .generate_content(prompt).text.strip()
        except Exception as e:
            logger.warning(f"rewrite_query fallback ({e})")
            return q

    # ---------------- Text extraction ------------------------
    def extract_pdf_text(self, path: str) -> str:
        out = []
        reader = PdfReader(path)
        for page in reader.pages:
            out.append(page.extract_text() or "")
        return "\n".join(out).strip()

    def extract_docx_text(self, path: str) -> str:
        if docx is None:
            raise RuntimeError("python-docx not installed")
        d = docx.Document(path)
        return "\n".join(p.text for p in d.paragraphs)

    def extract_html_text(self, path: str) -> str:
        if BeautifulSoup is None:
            raise RuntimeError("beautifulsoup4 not installed")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        soup = BeautifulSoup(raw, "html.parser")
        return soup.get_text(separator="\n")

    def extract_plain_text(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def extract_text_generic(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            return self.extract_pdf_text(path)
        if ext in [".txt", ".md"]:
            return self.extract_plain_text(path)
        if ext in [".html", ".htm"]:
            return self.extract_html_text(path)
        if ext == ".docx":
            return self.extract_docx_text(path)
        raise ValueError(f"Unsupported extension: {ext}")

    # ---------------- Discover docs --------------------------
    def discover_docs(self) -> List[str]:
        paths = []
        for ext in self.SUPPORTED_EXTS:
            paths.extend(glob.glob(f"docs/*{ext}"))
        return paths

    # ---------------- Batched embedding ----------------------
    def _embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        vecs = []
        for t in texts:
            v = self.embed(t)
            if not v:
                v = None
            vecs.append(v)
        return vecs

    # ---------------- Ingestion (batched + logging) ----------
    def ingest_docs(self, batch_size: int = 64):
        """
        Walk ./docs, extract text from each supported file,
        chunk it, embed in batches, and upsert into Qdrant.
        Logs errors instead of crashing.
        """
        self.ensure_collection()

        paths = self.discover_docs()
        if not paths:
            logger.warning("No supported docs found in ./docs")
            return

        for path in paths:
            self.ingest_single(path, batch_size=batch_size)

    # ---------------- Ingest a single file (for /upload) -----
    def ingest_single(self, path: str, batch_size: int = 64):
        """
        Ingest ONE file into Qdrant.
        Same logic as ingest_docs(), but scoped to a single path.
        Useful for /upload so we don't have to re-index everything.
        """
        self.ensure_collection()

        ext = os.path.splitext(path)[1].lower()
        if ext not in self.SUPPORTED_EXTS:
            logger.warning(f"[INGEST_SINGLE] Skipping unsupported type: {path}")
            return

        # read text
        try:
            txt = self.extract_text_generic(path)
        except Exception as e:
            logger.error(f"[INGEST_SINGLE] Failed to read {path}: {e}")
            return

        if not txt.strip():
            logger.info(f"[INGEST_SINGLE] Skipped empty text in {path}")
            return

        # chunk
        chunks = _chunk_text(txt, size=300, overlap=60)
        if not chunks:
            logger.info(f"[INGEST_SINGLE] No chunks after splitting {path}")
            return

        base_id = int(hashlib.md5(path.encode()).hexdigest()[:8], 16)

        # batch embed + upsert
        for chunk_batch, start_i, end_i in _yield_batches(chunks, batch_size):
            vecs = self._embed_batch(chunk_batch)

            points = []
            for local_idx, (ch, vec) in enumerate(
                zip(chunk_batch, vecs),
                start=start_i
            ):
                if vec is None:
                    logger.error(
                        f"[INGEST_SINGLE] embed failed for {path} chunk {local_idx}"
                    )
                    continue

                pid = base_id + local_idx
                points.append(
                    qm.PointStruct(
                        id=pid,
                        vector=vec,
                        payload={
                            "text": ch,
                            "source": path,
                            "chunk_id": local_idx,
                            "total_chunks": len(chunks),
                            "ext": os.path.splitext(path)[1].lower(),
                        },
                    )
                )

            if not points:
                logger.warning(
                    f"[INGEST_SINGLE] No valid points for {path} batch {start_i}:{end_i}"
                )
                continue

            try:
                self.qdrant.upsert(collection_name=self.COL, points=points)
                logger.info(
                    f"[INGEST_SINGLE] Upserted {len(points)} chunks "
                    f"({start_i}:{end_i}) from {path}"
                )
            except Exception as e:
                logger.error(
                    f"[INGEST_SINGLE] Qdrant upsert failed for {path} "
                    f"batch {start_i}:{end_i}: {e}"
                )

    # ---------------- Source discovery / locking -------------
    def list_all_sources(self, max_points: int = 10000):
        """
        Return set of unique 'source' payload values.
        """
        self.ensure_collection()
        sources = set()
        offset = None
        while True:
            res = self.qdrant.scroll(
                collection_name=self.COL,
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

    def guess_source_path(self, title_like: str | None):
        """
        Try substring match against known sources to lock retrieval to one file.
        """
        if not title_like:
            return None
        tl = title_like.lower().strip().replace('"', '')
        for src in self.list_all_sources():
            if tl in src.lower():
                return src
        return None

    # ---------------- Retrieval ------------------------------
    def retrieve(self, query: str, k: int = 5):
        """
        Vector search across entire collection.
        Returns [{text, source, chunk_id}, ...]
        """
        self.ensure_collection()
        vec = self.embed(query)
        res = self.qdrant.query_points(
            collection_name=self.COL,
            query=vec,
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

    def retrieve_from_source(self, query: str, source_path: str, k: int = 30):
        """
        Fetch from only one file: overfetch 3x, then filter to matching source.
        Deduplicate near-identical chunks via md5.
        """
        self.ensure_collection()
        vec = self.embed(query)
        res = self.qdrant.query_points(
            collection_name=self.COL,
            query=vec,
            limit=k * 3,
            with_payload=True,
        )
        hits = res.points or []
        filt = [h for h in hits if (h.payload or {}).get("source") == source_path]

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

    # ---------------- Web fallback ---------------------------
    def web_search_snippets(self, q: str, k: int = 5):
        """
        Web retrieval (Tavily). Return same schema as doc chunks.
        """
        if not self.tavily:
            return []
        try:
            res = self.tavily.search(q, max_results=k)
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
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
        self,
        user_q: str,
        rewritten_q: str,
        ctxs: list,
        force_web: Optional[bool],
    ) -> Tuple[list, bool]:
        """
        Decide if we add Tavily snippets.
        True  -> always
        False -> never
        None  -> only if we have zero local ctx
        """
        used_web = False
        if not self.tavily:
            return ctxs, used_web

        wants_web = False
        if force_web is True:
            wants_web = True
        elif force_web is None and not ctxs:
            wants_web = True

        if wants_web:
            web_ctxs = self.web_search_snippets(rewritten_q, k=5)
            if web_ctxs:
                ctxs = (ctxs or []) + web_ctxs
                used_web = True

        return ctxs, used_web

    # ---------------- Rerank -------------------------------
    def _get_cross(self):
        if self._cross is None:
            self._cross = CrossEncoder(self.RERANK_MODEL)
        return self._cross

    def rerank(self, query: str, ctxs: list, top_n: int = 30):
        if not ctxs:
            return []
        model = self._get_cross()
        pairs = [(query, c["text"]) for c in ctxs]
        scores = model.predict(pairs)
        ranked = sorted(zip(ctxs, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:top_n]]

    # ---------------- Prompt building / Answer -------------
    def _tag_contexts_for_prompt(self, ctxs: List[Dict]) -> Tuple[str, Dict[str, Dict]]:
        tagged_lines = []
        citation_map: Dict[str, Dict] = {}

        for i, c in enumerate(ctxs):
            cite_id = i + 1  # [1], [2], ...
            src = str(c.get("source", ""))
            label = "DOC" if src.lower().startswith("docs") else "WEB"

            tagged_lines.append(
                f"[{cite_id}][{label}] (chunk {c.get('chunk_id')}) "
                f"{c['text']} (source: {src})"
            )

            citation_map[str(cite_id)] = {
                "source": src,
                "chunk_id": c.get("chunk_id"),
            }

        full_block = "\n\n".join(tagged_lines)
        return full_block, citation_map

    def _build_answer_prompt(
        self,
        user_question: str,
        ctxs: List[Dict],
    ) -> Tuple[Optional[str], Dict[str, Dict], Optional[bool]]:
        if not ctxs:
            return None, {}, None

        context_block, citation_map = self._tag_contexts_for_prompt(ctxs)

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

    def _generate_with_gemini(self, prompt: str) -> str:
        try:
            return genai.GenerativeModel(self.MODEL_NAME).generate_content(prompt).text
        except Exception as e:
            return f"Model error with {self.MODEL_NAME}: {e}"

    def _generate_with_reasoner(self, prompt: str) -> str:
        if self.groq_client is None:
            return self._generate_with_gemini(prompt)

        try:
            completion = self.groq_client.chat.completions.create(
                model=self.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800,
            )
            return completion.choices[0].message.content.strip()
        except Exception:
            # silent fallback to Gemini
            return self._generate_with_gemini(prompt)

    def answer(
        self,
        q: str,
        ctxs: List[Dict],
        reasoner: Optional[str] = None
    ) -> Tuple[str, Dict[str, Dict], str]:
        """
        reasoner:
          None or "gemini"  -> Gemini
          anything else     -> Groq reasoning model (if available)
        """
        if not ctxs:
            return (
                "I could not retrieve anything from your documents. "
                "Try enabling web mode or ingesting more files.",
                {},
                self.MODEL_NAME,
            )

        prompt, citation_map, _ = self._build_answer_prompt(q, ctxs)
        if prompt is None:
            return (
                "I could not retrieve anything from your documents. "
                "Try enabling web mode or ingesting more files.",
                {},
                self.MODEL_NAME,
            )

        # model selection: Gemini by default, Groq if requested
        if reasoner is None or (isinstance(reasoner, str) and reasoner.lower() == "gemini"):
            final_text = self._generate_with_gemini(prompt)
            model_used = self.MODEL_NAME
        else:
            final_text = self._generate_with_reasoner(prompt)
            model_used = self.GROQ_MODEL if self.groq_client is not None else self.MODEL_NAME

        return final_text, citation_map, model_used
