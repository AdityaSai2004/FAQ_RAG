import os
import faiss
import numpy as np
import pypdf
from sentence_transformers import SentenceTransformer

# -------------------
# CONFIGURATION
# -------------------
PDF_PATH = "FAQ_MOMA.pdf"
INDEX_PATH = "data/faiss_index.bin"
TEXT_STORE_PATH = "data/text_chunks.npy"

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file and returns it as a single string."""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=300):
    """Splits text into smaller chunks for embedding storage."""
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]

def build_vector_store():
    """Extracts text, chunks it, embeds it, and stores in FAISS."""
    print("🔄 Extracting text from PDF...")
    text = extract_text_from_pdf(PDF_PATH)
    chunks = chunk_text(text)

    print(f"📌 Creating embeddings for {len(chunks)} chunks...")
    embeddings = np.array([embedding_model.encode(chunk) for chunk in chunks])

    print("✅ Storing embeddings in FAISS...")
    vector_dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(vector_dim)
    faiss_index.add(embeddings)

    # Save FAISS index & text chunks
    faiss.write_index(faiss_index, INDEX_PATH)
    np.save(TEXT_STORE_PATH, np.array(chunks))

    print("🎉 FAISS index & text chunks saved successfully!")

if __name__ == "__main__":
    build_vector_store()
