import argparse
import hashlib
import os

from app.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, PINECONE_NAMESPACE
from app.chunking import split_into_chunks
from app.embeddings import embed_texts
from app.pinecone_client import ensure_index


def load_documents(data_dir: str) -> list[tuple[str, str]]:
    docs = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith((".md", ".txt")):
            continue
        path = os.path.join(data_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            docs.append((fname, f.read()))
    return docs


def make_chunk_id(source_file: str, chunk_index: int) -> str:
    raw = f"{source_file}::chunk-{chunk_index}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def run_ingest(reset: bool = False, batch_size: int = 64) -> None:
    index = ensure_index()

    if reset:
        print(f"Resetting namespace '{PINECONE_NAMESPACE}'...")
        try:
            index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
        except Exception as e:
            print(f"(nothing to delete, or delete skipped: {e})")

    docs = load_documents(DATA_DIR)
    if not docs:
        print(f"No .md/.txt files found in '{DATA_DIR}'. Nothing to ingest.")
        return

    vectors = []
    total_chunks = 0

    for source_file, text in docs:
        chunks = split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            continue
        chunk_embeddings = embed_texts(chunks)
        for i, (chunk_text, vector) in enumerate(zip(chunks, chunk_embeddings)):
            chunk_id = make_chunk_id(source_file, i)
            vectors.append(
                {
                    "id": chunk_id,
                    "values": vector,
                    "metadata": {
                        "source_file": source_file,
                        "chunk_id": chunk_id,
                        "chunk_index": i,
                        "text": chunk_text,
                    },
                }
            )
            total_chunks += 1
        print(f"  {source_file}: {len(chunks)} chunks")

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch, namespace=PINECONE_NAMESPACE)

    print(
        f"\nIngest complete: {total_chunks} chunks from {len(docs)} files "
        f"upserted into namespace '{PINECONE_NAMESPACE}'."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Pinecone.")
    parser.add_argument("--reset", action="store_true", help="Delete existing vectors first.")
    args = parser.parse_args()
    run_ingest(reset=args.reset)