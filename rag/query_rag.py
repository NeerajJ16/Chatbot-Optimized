import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
#from rag.llm_phi2 import Phi2LLM  # your local phi-2 wrapper
from rag.llm_phi2 import get_llm
from rag.chunker import Chunk  # use your Chunk dataclass
import re

EMBEDDINGS_PATH = "data/chunks.pkl"  # contains list of Chunk objects
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5  # number of chunks to retrieve


def extract_date_key(text):
    """
    Returns a sortable date key (year, month) if present.
    Falls back to (0, 0) if no date is found.
    """
    year_match = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    month_match = re.findall(
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
        text.lower()
    )

    year = int(year_match[-1]) if year_match else 0
    month = month_match[-1] if month_match else ""

    month_map = {
        "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
        "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12
    }

    return (year, month_map.get(month, 0))


def reorder_chunks_for_query(query, chunks):
    q = query.lower()

    if any(word in q for word in ["recent", "latest", "current", "most recent"]):
        return sorted(
            chunks,
            key=lambda c: extract_date_key(c.text),
            reverse=True
        )

    return chunks


# Load chunks and embeddings from FAISS or pickle
def load_chunks_and_embeddings(file_path=EMBEDDINGS_PATH):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    # Assuming each Chunk has a precomputed embedding attribute, else compute here
    chunks = data  # list of Chunk objects
    return chunks


def embed_chunks(chunks, model_name=EMBEDDING_MODEL):
    """Compute embeddings for chunks if not already stored."""
    model = SentenceTransformer(model_name)
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def retrieve_chunks(query, chunks, embeddings, top_k=TOP_K):
    """Retrieve top-k relevant chunks using cosine similarity."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_emb = model.encode([query])
    scores = cosine_similarity(query_emb, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices], [scores[i] for i in top_indices]


def retrieve_chunks_by_section(query, chunks, embeddings, top_k=TOP_K):
    """Filter chunks by section if query matches section name."""
    section_names = {c.section.lower() for c in chunks}
    query_lower = query.lower()
    matched_section = None
    for sec in section_names:
        if sec in query_lower:
            matched_section = sec
            break

    if matched_section:
        filtered_chunks = [c for c in chunks if c.section.lower() == matched_section]
        filtered_embeddings = np.array([embeddings[i] for i, c in enumerate(chunks) if c.section.lower() == matched_section])
    else:
        filtered_chunks = chunks
        filtered_embeddings = embeddings

    return retrieve_chunks(query, filtered_chunks, filtered_embeddings, top_k)


def build_context(chunks):
    """Join retrieved chunks into a context string for LLM."""
    return "\n\n".join([f"{c.section.upper()}: {c.text}" for c in chunks])


def answer_question(query, chunks, embeddings):
    """Full RAG pipeline: retrieval + LLM answer generation."""
    top_chunks, scores = retrieve_chunks_by_section(query, chunks, embeddings)
    top_chunks = reorder_chunks_for_query(query, top_chunks)
    context = build_context(top_chunks)

    print("\nTop retrieved chunks:")
    for i, (c, score) in enumerate(zip(top_chunks, scores), 1):
        print(f"{i}. Source: {c.source}, Section: {c.section}, Score: {score:.4f}")
        print(f"Text preview: {c.text[:300]}...\n")

    context = build_context(top_chunks)

    # Generate answer using Phi-2 (CPU-friendly small model)
    llm = get_llm()
    answer = llm.generate_answer(context, query)
    return answer


if __name__ == "__main__":
    print("Loading chunks and embeddings...")
    chunks = load_chunks_and_embeddings()
    embeddings = embed_chunks(chunks)
    print(f"Loaded {len(chunks)} chunks, embeddings shape: {embeddings.shape}")

    while True:
        query = input("\nEnter your question (or 'exit'): ").strip()
        if query.lower() == "exit":
            break

        answer = answer_question(query, chunks, embeddings)
        print("\nANSWER:\n", answer)
