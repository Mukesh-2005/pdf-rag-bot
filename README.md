# PDF RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot that answers questions
about a single PDF. Upload a PDF, then ask questions about its content in a
chat interface — answers come only from what's in the document.

## How it works

1. `ingest.py` — extracts text from the PDF, splits it into overlapping
   chunks, embeds each chunk locally (via `sentence-transformers`), and
   stores them in a FAISS index.
2. `query.py` — embeds your question, retrieves the most relevant chunks
   from the index, and asks Google's Gemini to answer using only that
   retrieved context.
3. `app.py` — a Gradio web interface that ties both together: upload a PDF
   at the top, chat below.

## Setup

### 1. Install Python dependencies
```
pip install -r requirements.txt
```

### 2. Get a free Google Gemini API key
Go to https://aistudio.google.com/apikey, sign in, and click "Create API key".

### 3. Set the API key as an environment variable

**Windows (PowerShell):**
```
$env:GOOGLE_API_KEY="your-key-here"
```
This only lasts for the current terminal session — you'll need to set it
again each time you open a new PowerShell window, unless you add it to your
system environment variables permanently.

**macOS / Linux:**
```
export GOOGLE_API_KEY="your-key-here"
```

## Running it

```
python app.py
```

This starts a local Gradio server and prints a URL (usually
`http://127.0.0.1:7860`). Open that in your browser.

1. Upload a PDF using the file box at the top — wait for the "Indexed!"
   status message.
2. Ask questions about it in the chat box below.

## Notes

- **First upload will feel slow.** The embedding model
  (`all-MiniLM-L6-v2`) and its underlying libraries (like `torch`) take
  10–30 seconds to load the first time you use them, plus a one-time
  ~80MB model download if you haven't run it before on this machine.
  This is normal and only happens once per session.
- **Scanned/image-only PDFs won't work** — this uses text extraction, not
  OCR. If a PDF has no selectable text, you'll see a warning.
- **This is built for a single PDF at a time.** Uploading a new PDF
  overwrites the previous index.

## Project structure

```
pdf-rag-bot/
  ├── ingest.py         # PDF -> text -> chunks -> embeddings -> FAISS index
  ├── query.py           # question -> retrieval -> Gemini answer
  ├── app.py              # Gradio chat interface
  └── requirements.txt
```
