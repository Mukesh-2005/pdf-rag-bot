# PDF RAG Chatbot — Project Documentation

---

## 1. Goal

Build a simple RAG (Retrieval-Augmented Generation) system as a first,
small project before tackling something bigger. The idea: upload one
PDF, ask questions about it, get answers grounded only in that document
— not the model's general knowledge.

---

## 2. Architecture

```
PDF  →  text extraction  →  chunking  →  embeddings  →  FAISS index
                                                              │
question  →  embed question  →  search index  →  top-k chunks
                                                              │
                                      chunks + question → Gemini → answer
```

Three scripts, split by responsibility:

- **`ingest.py`** — one-time processing of the PDF into a searchable index.
- **`query.py`** — takes a question, retrieves relevant chunks, calls the
  LLM to generate an answer.
- **`app.py`** — Gradio UI wiring the two together (upload + chat).

---

## 3. Key decisions and why

### Interface: Gradio, not Streamlit
Started with Streamlit as the default choice, but switched to Gradio
because it has a built-in `ChatInterface` component purpose-built for
chat — much less boilerplate than manually looping over
`st.chat_message` in Streamlit. Still just one Python file, still free
to deploy (e.g. Hugging Face Spaces).

### Vector store: FAISS, not Postgres/pgvector
FAISS is in-memory and needs no server — just a pip install. Good fit
for a "simple, single-PDF" project. Already know Postgres/pgvector from
other work, so that's a natural upgrade path for a bigger/multi-document
version later, not needed here.

### Embeddings: sentence-transformers (local), not an API
Used `all-MiniLM-L6-v2` — runs on CPU, no GPU needed, no per-call cost or
API key required just to embed text. Keeps the "retrieval" half of RAG
completely free and offline; only the "generation" half needs an API.

### LLM for generation: switched from Claude API to Gemini API
Originally built with the Anthropic API since that's the model I use
day-to-day. Switched to Google's Gemini API (`gemini-2.0-flash`) by
request — free tier, and decouples this project from needing an
Anthropic key specifically. The swap only touched `query.py` (the
retrieval/embedding logic didn't change at all) — a good sign the
retrieval and generation layers were properly separated from the start.

### Chunking: simple character-based sliding window
`CHUNK_SIZE = 800` characters, `CHUNK_OVERLAP = 100`. Deliberately basic
(not sentence/paragraph-aware) to keep the first version simple. Known
limitation: can cut a chunk mid-sentence or split a bulleted list across
two chunks — this came back to matter later (see Section 4).

---

## 4. What `TOP_K` is, and a real retrieval failure

`TOP_K` (in `query.py`) controls how many chunks get pulled from the
FAISS index and handed to the LLM as context for a given question.
Higher = more context (but more noise / risk of diluting relevant
content); lower = tighter context (but higher risk of missing the chunk
that actually has the answer).

**Real example of this mattering:** ran 5 test questions against the
TNPSC counselling PDF (see `eval_questions.md`). Four answers were
accurate. The fifth — "What are the two stated advantages of the
counselling system?" — was wrong in an interesting way: the model
*noticed* its own retrieval was incomplete (it said the advantages
section "is cut off") but then, instead of saying "I don't know,"
substituted a real detail from a different part of the document to fill
the gap. The answer sounded confident and reasonable, but it wasn't
actually what the source said.

**Why:** the "Advantages" section is a short bulleted list near the very
end of a 5-page document. With `CHUNK_SIZE=800` and `TOP_K=4`, there's a
real risk of the chunk containing the full list not being retrieved, or
being split across a chunk boundary.

**Current status:** with `TOP_K=4`, the system works fine in practice —
no code or config changes were needed. This case is documented as a
reminder of the failure mode to watch for, not as an open bug.

**Lesson — the big one:** a RAG system can produce an answer that is
fluent, confident, and *partially* grounded in the real document, while
still being factually wrong. It's not always obvious hallucination
(inventing facts from nowhere) — it can be *misattributed retrieval*
(true facts from the wrong place in the document, used to answer the
wrong question). This is exactly why having ground-truth Q&A pairs to
test against matters — this failure would have been easy to miss just
skimming the answer, since it read as plausible.

---

## 5. Evaluation approach

Built `eval_questions.md` — 5 hand-picked questions from the source PDF,
each with a ground-truth answer written by hand from the actual
document text (not generated). Deliberately included two "precise fact"
questions (a specific ratio, a specific rule) and one "spans a bulleted
list" question, since those are the categories most likely to expose
chunking/retrieval problems rather than pure LLM reasoning problems.

**Lesson:** for RAG specifically, evaluation questions are more useful
if chosen to stress *retrieval*, not just the LLM's ability to summarize
text it's already been handed. A model can be excellent at generation
and still give wrong answers if the retrieval step feeds it the wrong
chunks.

---

## 6. Version control

Set up git locally, added a `.gitignore` to keep `venv/`, the generated
`index/` folder, and any API keys out of the repository — all of those
are either machine-specific, regenerable, or secret, and none belong in
version control.


