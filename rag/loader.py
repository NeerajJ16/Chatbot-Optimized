from pathlib import Path
from rag.chunker import paragraph_chunk_text

FILES_DIR = Path("Files")

def load_and_chunk_files():
    all_chunks = []

    for file_path in FILES_DIR.glob("*.txt"):
        section = file_path.stem  # filename without .txt

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = paragraph_chunk_text(
            text=text,
            section=section,
            source=file_path.name
        )

        all_chunks.extend(chunks)

    return all_chunks

