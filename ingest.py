import sys
import os
import pickle

import faiss
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer

# ---- config ----
CHUNK_SIZE = 800       # characters per chunk (rough proxy for tokens)
CHUNK_OVERLAP = 100    # characters of overlap between consecutive chunks
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast, runs fine on CPU
INDEX_DIR = "index"
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.pkl")


def extract_text(pdf_path: str) -> str:
    """Pull all text out of the PDF, page by page."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
            else:
                print(f"  [warn] page {page_num} had no extractable text (scanned image?)")
    return "\n".join(text_parts)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple sliding-window character chunker with overlap."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_index(chunks: list[str], model: SentenceTransformer) -> faiss.Index:
    """Embed all chunks and build a flat L2 FAISS index."""
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def main():
    if len(sys.argv) != 2:
        print("Usage: python ingest.py path/to/document.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    os.makedirs(INDEX_DIR, exist_ok=True)

    print(f"[1/4] Extracting text from {pdf_path} ...")
    raw_text = extract_text(pdf_path)
    if not raw_text.strip():
        print("No text could be extracted. Is this a scanned/image-only PDF? "
              "You'd need OCR (e.g. pytesseract) first.")
        sys.exit(1)
    print(f"  extracted {len(raw_text):,} characters")

    print("[2/4] Chunking text ...")
    chunks = chunk_text(raw_text)
    print(f"  created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    print(f"[3/4] Loading embedding model '{EMBED_MODEL_NAME}' ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("[4/4] Embedding chunks and building FAISS index ...")
    index = build_index(chunks, model)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"\nDone. Index saved to '{INDEX_PATH}', chunks saved to '{CHUNKS_PATH}'.")
    print(f"You can now run: python app.py")


if __name__ == "__main__":
    main()
