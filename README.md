# Agentic RAG Assistant (Local Prototype)

This is a local Retrieval-Augmented Generation (RAG) system that can answer questions using your own documents, optionally enrich those answers with relevant web context, and generate cited, auditable responses.

It’s designed for domains where accuracy and traceability matter (policies, procedures, compliance, onboarding, risk analysis, internal knowledge bases, etc.), but it can adapt to any topic. Whatever you put in `./docs` becomes its knowledge.

It can:
- Ingest your own PDFs / text docs.
- Retrieve the most relevant chunks from those docs.
- Optionally blend in fresh web snippets (via Tavily).
- Rerank results using a learned cross-encoder (not just vector similarity).
- Generate a final answer with citations.
- Choose between a fast model (Gemini) and a deeper reasoning model (Groq).

Everything runs locally right now (FastAPI + Qdrant in Docker).  
No public cloud deployment is required.

---

## 🔍 Why this is not “just another RAG demo”

Most basic RAG = “embed doc → semantic search → dump chunks into an LLM.”

This system adds multiple production-style steps:

### 1. Agentic Query Rewrite
Before we even retrieve, we rewrite the user’s question using Gemini to:
- make it self-contained,
- add obvious keywords,
- remove vague pronouns like “this” / “that.”

Example:
> User asks: "what are the rules before payout?"  
> Rewrite becomes: "payout approval rules and required checks before releasing funds to user bank account"

That rewritten query gets embedded and used for retrieval.  
This gives you better recall from Qdrant.

---

### 2. Blended Context: Local Docs + Web
The assistant can use:
- **Your documents** in `./docs` (PDF or TXT)
- **Web context** via Tavily (optional)

Both are merged into one context list so the model sees everything at once.

Each chunk is tagged internally as `DOC` or `WEB`, and the prompt tells the model:

> Prefer DOC over WEB if there's any conflict.

That means your internal policy/knowledge wins over the internet.  
This is important for accuracy and trust.

---

### 3. Learned Reranking and Domain Boost
After getting candidate chunks, we:
1. Rerank them using a cross-encoder model from `sentence-transformers` instead of relying only on vector distance.
2. Apply a light keyword boost for domain-important terms.  
   (For example, terms around review, approval, identity, fraud, limits, etc. You can customize or remove this.)

Result: you get answers that surface the “important operational steps / rules / risks,” not just text that happens to be semantically similar.

---

### 4. Two Answering Modes (“fast vs deep”)
When generating the final answer, you can choose the reasoning model:

- `reasoner = "gemini"`  
  Uses Gemini for a fast, cheap, responsive answer.

- `reasoner = "groq"`  
  Uses a Groq-hosted LLaMA model for more structured / analytical answers.

If Groq isn't configured or fails, we silently fall back to Gemini (no ugly debug text exposed to users).

The API will tell you which one was actually used.

---

### 5. Auditable Answers with Citations
Every final answer comes with:
- Inline citations like `[1]`, `[2]`, `[3]`
- A `citations` map in the API response that tells you:
  - which source each `[n]` came from,
  - whether it was a local document chunk or web.

The API also returns the exact ranked `contexts` that were fed to the model to produce the answer.

That means you can explain why the model said what it said.

This is extremely useful for:
- internal policy assistants
- onboarding/training assistants
- risk/compliance review assistants
- product support knowledge assistants
- audit trails

---

## 🧠 Architecture: end-to-end flow

When you call `POST /ask`, here’s what happens:

1. We rewrite the user question using Gemini  
   → `question_rewritten`

2. We embed the rewritten query using `text-embedding-004` and do semantic retrieval from Qdrant.

3. If requested (or if the local docs are empty), we also fetch web snippets using Tavily and merge them into the same context format.

4. We rerank all candidate chunks using a cross-encoder (`sentence-transformers`).

5. We apply a final priority boost for certain important terms in your domain (these keywords are customizable / removable).

6. We build a prompt for the answering model that:
   - lists each chunk as `[1][DOC]` or `[2][WEB]`
   - includes the source path/URL
   - instructs the model to:
     - prefer DOC over WEB,
     - avoid inventing facts,
     - answer concisely,
     - always cite sources `[n]`.

7. We generate a final answer using either:
   - Gemini (fast mode), or
   - Groq LLaMA (deeper reasoning mode).

8. We return JSON with:
   - the answer,
   - which model was used,
   - all supporting chunks,
   - the citations map.

So you end up with not just “an answer,” but something you can defend and trace.

---

## 📂 Project Layout

```text
.
├─ fastapi_app.py        # FastAPI API server
├─ rag.py                # Core RAG pipeline (ingest, retrieve, rerank, answer)
├─ docs/                 # Your local knowledge base (.pdf, .txt)
├─ requirements.txt      # Python dependencies
├─ .env                  # Your secrets (NOT committed)
└─ README.md             # This file
