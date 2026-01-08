# rag/chunker.py
import os
import pickle
import re
from rag.schema import Chunk

FILES_DIR = "Files"
OUTPUT_PATH = "data/chunks.pkl"

MIN_CHARS = 150
MAX_CHARS = 800


def load_files():
    documents = []

    for filename in os.listdir(FILES_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(FILES_DIR, filename)

            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            documents.append({
                "section": filename.replace(".txt", ""),
                "source": filename,
                "text": text
            })

    return documents


def clean_text(text: str) -> str:
    text = re.sub(r"\[conversation_history\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def semantic_chunk(text: str):
    lines = [clean_text(l) for l in text.split("\n") if clean_text(l)]
    chunks = []

    buffer = ""

    for line in lines:
        # Merge until chunk is meaningful
        if len(buffer) < MIN_CHARS:
            buffer += " " + line
        else:
            chunks.append(buffer.strip())
            buffer = line

        # Hard cap
        if len(buffer) >= MAX_CHARS:
            chunks.append(buffer.strip())
            buffer = ""

    if buffer.strip():
        chunks.append(buffer.strip())

    return chunks


def create_chunks():
    documents = load_files()
    chunks = []

    for doc in documents:
        semantic_chunks = semantic_chunk(doc["text"])

        for chunk_text in semantic_chunks:
            chunks.append(
                Chunk(
                    source=doc["source"],
                    section=doc["section"],
                    text=chunk_text
                )
            )

    return chunks


def save_chunks(chunks):
    os.makedirs("data", exist_ok=True)

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved {len(chunks)} chunks → {OUTPUT_PATH}")


if __name__ == "__main__":
    chunks = create_chunks()
    save_chunks(chunks)
