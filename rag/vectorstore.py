# rag/vectorstore_langchain.py

import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from rag.chunker import load_documents, chunk_documents

VECTORSTORE_PATH = "data/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_vectorstore():
    print("Loading documents...")
    documents = load_documents()

    print("Chunking documents...")
    chunks = chunk_documents(documents)

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)

    print(f"Vector store saved → {VECTORSTORE_PATH}")


if __name__ == "__main__":
    build_vectorstore()
