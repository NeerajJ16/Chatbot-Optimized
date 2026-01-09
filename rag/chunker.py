import os
import re
import pickle
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

FILES_DIR = "Files"
OUTPUT_PATH = "data/chunks.pkl"

MIN_CHARS = 150
MAX_CHARS = 800


def clean_text(text: str) -> str:
    text = re.sub(r"\[conversation_history\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_documents():
    documents = []

    for filename in os.listdir(FILES_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(FILES_DIR, filename)

            loader = TextLoader(path, encoding="utf-8")
            docs = loader.load()

            for doc in docs:
                doc.page_content = clean_text(doc.page_content)
                doc.metadata = {
                    "source": filename,
                    "section": filename.replace(".txt", "")
                }
                documents.append(doc)

    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHARS,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    # Optional: enforce minimum size
    chunks = [
        chunk for chunk in chunks
        if len(chunk.page_content) >= MIN_CHARS
    ]

    return chunks


def save_chunks(chunks):
    os.makedirs("data", exist_ok=True)

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved {len(chunks)} chunks → {OUTPUT_PATH}")


if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    save_chunks(chunks)
