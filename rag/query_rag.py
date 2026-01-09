# rag/query_rag_langchain.py

import re
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from rag.llm_phi2 import get_llm

# --------------------
# Configuration
# --------------------
VECTORSTORE_PATH = "data/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


# --------------------
# Utility: date extraction (your logic)
# --------------------
def extract_date_key(text):
    """
    Returns a sortable (year, month) tuple if a date is found.
    Falls back to (0, 0) if no date exists.
    """
    year_match = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    month_match = re.findall(
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
        text.lower()
    )

    year = int(year_match[-1]) if year_match else 0

    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4,
        "may": 5, "jun": 6, "jul": 7, "aug": 8,
        "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }

    month = month_map.get(month_match[-1], 0) if month_match else 0
    return (year, month)


# --------------------
# Utility: reorder for "latest" queries
# --------------------
def reorder_chunks_for_query(query, docs):
    q = query.lower()

    if any(word in q for word in ["recent", "latest", "current", "most recent"]):
        return sorted(
            docs,
            key=lambda d: extract_date_key(d.page_content),
            reverse=True
        )

    return docs


# --------------------
# Load vector store
# --------------------
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


# --------------------
# Retrieve documents
# --------------------
def retrieve_documents(query, vectorstore):
    """
    Uses FAISS similarity search.
    Returns list of (Document, score).
    """
    return vectorstore.similarity_search_with_score(
        query,
        k=TOP_K
    )


# --------------------
# Build context for LLM
# --------------------
def build_context(docs):
    """
    Formats retrieved documents into a single context string.
    """
    return "\n\n".join(
        f"{doc.metadata.get('section', '').upper()}: {doc.page_content}"
        for doc in docs
    )


# --------------------
# Full RAG pipeline
# --------------------
def answer_question(query, vectorstore):
    results = retrieve_documents(query, vectorstore)

    docs = [doc for doc, _ in results]
    scores = [score for _, score in results]

    # Apply your custom ranking logic
    docs = reorder_chunks_for_query(query, docs)

    # Debug output (same spirit as your original code)
    print("\nTop retrieved chunks:")
    for i, (doc, score) in enumerate(zip(docs, scores), 1):
        print(
            f"{i}. Source: {doc.metadata.get('source')}, "
            f"Section: {doc.metadata.get('section')}, "
            f"Score: {score:.4f}"
        )
        print(f"Text preview: {doc.page_content[:300]}...\n")

    context = build_context(docs)

    llm = get_llm()
    return llm.generate_answer(context, query)


# --------------------
# Interactive loop
# --------------------
if __name__ == "__main__":
    print("Loading vector store...")
    vectorstore = load_vectorstore()

    while True:
        query = input("\nEnter your question (or 'exit'): ").strip()
        if query.lower() == "exit":
            break

        answer = answer_question(query, vectorstore)
        print("\nANSWER:\n", answer)
