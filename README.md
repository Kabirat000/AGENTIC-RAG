# Agentic RAG Compliance Assistant (Local Prototype)

This is a local Retrieval-Augmented Generation (RAG) system focused on crypto off-ramp / KYC / AML / compliance questions.

It can:
- Ingest your own PDFs / text docs.
- Retrieve the most relevant chunks from those docs.
- Optionally blend in fresh web research (via Tavily).
- Rerank results using a learned cross-encoder.
- Generate a final answer with citations.
- Choose between a fast model (Gemini) and a deeper reasoning model (Groq).

Everything runs locally right now (FastAPI + Qdrant in Docker).  
No public deployment required.

---

## 🔍 What this solves

Typical RAG demos just do “retrieve top 3 chunks and stuff them into an LLM.”  
This project goes further in ways that actually matter for compliance/off-ramp use cases:

1. **Agentic Query Rewrite**  
   - User asks something vague like:  
     “what should we warn users about cashing out?”  
   - System rewrites it into something retrieval-friendly:  
     “crypto off-ramp withdrawal fraud and AML obligations when sending large amounts to bank”
   - This makes vector search way more accurate.

2. **Blended Context: Internal Docs + Web**  
   - You can ingest internal policy / training docs in `./docs/`.
   - If needed, the system also fetches recent web info (via Tavily).
   - Both are merged into a single context with DOC / WEB tags.
   - The LLM is told: **use DOC when it conflicts with WEB.**
   - This is important for compliance: your policy should override the internet.

3. **Learned Reranking + Domain Boost**  
   - After retrieval, we rerank using a cross-encoder (`sentence-transformers`) instead of relying only on vector similarity.
   - Then we boost chunks that mention risky terms like `withdrawal`, `KYC`, `AML`, `bank`, `limits`, `fraud`, etc.
   - Result: compliance-relevant content surfaces first.

4. **Two “Brains”: Fast vs Deep**  
   - `reasoner = "gemini"` → fast answer using Gemini.
   - `reasoner = "groq"` → deeper reasoning answer using a Groq-hosted LLaMA model.
   - If Groq is not configured or fails, it silently falls back to Gemini.
   - You can see which model answered in the API response.

5. **Citations + Traceability**  
   - Final answers come with `[1]`, `[2]`, … style citations.
   - The API returns a `citations` map so you can inspect which source chunk each reference came from (either your doc or a web URL).
   - You also get the ranked `contexts` that were actually fed to the LLM.
   - This makes it auditable (“why did the AI tell me this?”).

---

## 🧠 Architecture

### High-level flow for `/ask`:

1. Rewrite the user question using Gemini → `question_rewritten`
2. Embed that rewritten query using `text-embedding-004`
3. Retrieve relevant chunks from Qdrant
4. Optionally pull web snippets from Tavily and merge them
5. Rerank all candidate chunks with a cross-encoder
6. Apply domain keyword boosting (KYC / AML / cash out / off-ramp)
7. Build a prompt that:
   - tags each chunk as DOC or WEB
   - enforces “prefer DOC over WEB”
   - enforces “cite sources as [n]”
8. Send that prompt to the chosen reasoning model:
   - Gemini (fast mode) or
   - Groq model (deeper mode)
9. Return:
   - final answer text
   - which model wrote it
   - contexts used
   - citations map for UI / auditing

---

## 🗂 Project Layout

```text
.
├─ fastapi_app.py        # FastAPI API server
├─ rag.py                # Core pipeline: ingest, retrieve, rerank, answer
├─ docs/                 # Local knowledge base (.pdf, .txt)
├─ requirements.txt      # Python dependencies
├─ .env                  # Local secrets (NOT committed)
└─ README.md             # This file
