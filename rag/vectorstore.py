# rag/vectorstore.py

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rag.chunker import Chunk  # your current Chunk class

# Paths
CHUNKS_FILE = "data/chunks.pkl"
FAISS_INDEX_FILE = "data/faiss_index.index"
BATCH_SIZE = 32
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load chunks
with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

print(f"Total chunks loaded: {len(chunks)}")

# Embed chunks
print("Embedding chunks...")
model = SentenceTransformer(EMBEDDING_MODEL)

texts = [c.text for c in chunks]
embeddings = model.encode(texts, show_progress_bar=True, batch_size=BATCH_SIZE)
embeddings = np.array(embeddings).astype("float32")

print(f"Embeddings shape: {embeddings.shape}")

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance
index.add(embeddings)
print(f"FAISS index built with {index.ntotal} vectors.")

# Save FAISS index
os.makedirs("data", exist_ok=True)
faiss.write_index(index, FAISS_INDEX_FILE)
print(f"FAISS index saved → {FAISS_INDEX_FILE}")

# Save chunks metadata (so we can retrieve text later)
with open("data/faiss_chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)
print(f"Chunks metadata saved → data/faiss_chunks.pkl")


# --------------------
# Query function
# --------------------
def query_vector_store(query_text, top_k=5):
    query_vec = model.encode([query_text]).astype("float32")
    distances, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        chunk = chunks[idx]
        results.append({
            "source": chunk.source,
            "section": chunk.section,
            "text": chunk.text,
            "score": float(score)
        })

    return results


# --------------------
# Interactive querying
# --------------------
if __name__ == "__main__":
    while True:
        query = input("\nEnter your query (or 'exit'): ")
        if query.lower() == "exit":
            break

        top_results = query_vector_store(query)
        for i, r in enumerate(top_results):
            print(f"\nResult #{i+1}")
            print(f"Source: {r['source']}, Section: {r['section']}, Score: {r['score']:.4f}")
            print(f"Text preview: {r['text'][:300]}...")
