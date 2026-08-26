import os
import pickle

import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

INDEX_DIR = "index"
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.pkl")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 4
GEMINI_MODEL = "gemini-3.6-flash"   # fast + free-tier friendly; swap for gemini-2.5-pro if you need more accuracy

_embed_model = None
_index = None
_chunks = None
_gemini_model = None


def _lazy_load():
    """Load the index, chunks, embedding model, and Gemini client once."""
    global _embed_model, _index, _chunks, _gemini_model

    if _index is None:
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(
                "No index found. Run 'python ingest.py your_file.pdf' first."
            )
        _index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            _chunks = pickle.load(f)

    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    if _gemini_model is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY not set. Get a key at https://aistudio.google.com/apikey "
                'and set it with: $env:GOOGLE_API_KEY="your-key-here"'
            )
        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel(GEMINI_MODEL)


def retrieve(question: str, top_k: int = TOP_K) -> list[str]:
    """Return the top_k chunks most similar to the question."""
    _lazy_load()
    q_embedding = _embed_model.encode([question], convert_to_numpy=True).astype("float32")
    distances, indices = _index.search(q_embedding, top_k)
    return [_chunks[i] for i in indices[0] if i != -1]


def answer(question: str, top_k: int = TOP_K) -> str:
    """Retrieve context, then ask Gemini to answer using only that context."""
    _lazy_load()
    retrieved_chunks = retrieve(question, top_k)
    context = "\n\n---\n\n".join(retrieved_chunks)

    prompt = f"""You are answering a question using ONLY the context below, taken from a PDF document.
If the answer isn't in the context, say you don't know based on the document — don't make something up.

Context:
{context}

Question: {question}

Answer:"""

    response = _gemini_model.generate_content(prompt)
    return response.text


if __name__ == "__main__":
    # quick CLI test loop
    _lazy_load()
    print("Ask questions about your PDF (Ctrl+C to quit).")
    while True:
        try:
            q = input("\nQ: ")
        except KeyboardInterrupt:
            break
        if not q.strip():
            continue
        print("\nA:", answer(q))