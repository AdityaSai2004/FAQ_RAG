import os
import faiss
import numpy as np
from groq import Groq
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# -------------------
# CONFIGURATION
# -------------------
load_dotenv()
INDEX_PATH = "data/faiss_index.bin"
TEXT_STORE_PATH = "data/text_chunks.npy"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index & stored text chunks
print("🔄 Loading FAISS index & text chunks...")
faiss_index = faiss.read_index(INDEX_PATH)
faq_texts = np.load(TEXT_STORE_PATH, allow_pickle=True)

# Groq API client
client = Groq(api_key = GROQ_API_KEY)

def retrieve_relevant_chunk(query):
    """Finds the most relevant chunk using FAISS similarity search."""
    query_embedding = np.array([embedding_model.encode(query)])
    _, indices = faiss_index.search(query_embedding, 1)  # Top-1 result
    return faq_texts[indices[0][0]]

def ask_groq(question, context):
    """Generates an answer using Groq LLM based on retrieved context."""
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": """You are an AI assistant answering questions **only** based on the provided official document. If the document does not contain relevant information, say: "The document does not contain this information." Do not make assumptions."""},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content.strip()

# -------------------
# FLASK API
# -------------------

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def answer_question():
    """API endpoint for question answering."""
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Please provide a question"}), 400

    # Retrieve relevant chunk
    context = retrieve_relevant_chunk(question)

    # Generate answer using Groq LLM
    answer = ask_groq(question, context)
    # print(context)

    return jsonify({"question": question, "answer": answer, "context": context})

# -------------------
# START SERVER
# -------------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
