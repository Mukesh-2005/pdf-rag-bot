import os
import tempfile

import gradio as gr

import ingest
import query


def handle_upload(pdf_file):
    """Called when the user uploads a PDF. Runs the full ingest pipeline."""
    if pdf_file is None:
        return "No file uploaded."

    pdf_path = pdf_file.name if hasattr(pdf_file, "name") else pdf_file

    os.makedirs(ingest.INDEX_DIR, exist_ok=True)

    raw_text = ingest.extract_text(pdf_path)
    if not raw_text.strip():
        return "Couldn't extract any text from that PDF (might be scanned images)."

    chunks = ingest.chunk_text(raw_text)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(ingest.EMBED_MODEL_NAME)
    index = ingest.build_index(chunks, model)

    import faiss
    import pickle
    faiss.write_index(index, ingest.INDEX_PATH)
    with open(ingest.CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    # force query.py to reload the fresh index on the next question
    query._index = None
    query._chunks = None

    return f"Indexed! {len(chunks)} chunks ready. Ask away in the chat below."


def handle_chat(message, history):
    """Called on every chat turn."""
    if not os.path.exists(query.INDEX_PATH):
        return "Upload a PDF first (top of the page), then ask your question."
    try:
        return query.answer(message)
    except Exception as e:
        return f"Error: {e}"


with gr.Blocks(title="PDF RAG Chatbot") as demo:
    gr.Markdown("## Chat with your PDF\nUpload a PDF, then ask questions about it.")

    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        status_box = gr.Textbox(label="Status", interactive=False)

    pdf_input.change(fn=handle_upload, inputs=pdf_input, outputs=status_box)

    gr.ChatInterface(fn=handle_chat)

if __name__ == "__main__":
    demo.launch()
