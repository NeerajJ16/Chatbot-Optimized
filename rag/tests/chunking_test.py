from rag.loader import load_and_chunk_files

chunks = load_and_chunk_files()

print(f"Total chunks: {len(chunks)}\n")

for c in chunks:
    print("SECTION:", c.section)
    print("SOURCE:", c.source)
    print("TEXT:", c.text)
    print("-" * 50)
