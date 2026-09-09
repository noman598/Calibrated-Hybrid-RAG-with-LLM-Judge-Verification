# src/dump_chunks_for_labeling.py

import json
from pathlib import Path

def dump_chunks_readable(collection, output_path: str = "eval_sets/chunk_dump.txt"):
    """
    Pulls everything from Chroma and writes a human-readable file
    so you can scan through chunk_id + text pairs quickly for labeling.
    """
    data = collection.get(include=["documents", "metadatas"])
    chunk_ids = data["ids"]
    chunk_texts = data["documents"]
    chunk_metadatas = data["metadatas"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for cid, text, meta in zip(chunk_ids, chunk_texts, chunk_metadatas):
            f.write(f"chunk_id: {cid}\n")
            f.write(f"doc_name: {meta.get('doc_name', 'N/A')}\n")
            f.write(f"text:\n{text}\n")
            f.write("-" * 80 + "\n\n")

    print(f"Dumped {len(chunk_ids)} chunks to {output_path}")


def dump_chunks_jsonl(collection, output_path: str = "eval_sets/chunk_dump.jsonl"):
    """
    Same data, but JSONL — useful if you want to filter/search programmatically
    instead of scrolling a text file (e.g. grep for a keyword to find candidate chunk_ids fast).
    """
    data = collection.get(include=["documents", "metadatas"])
    chunk_ids = data["ids"]
    chunk_texts = data["documents"]
    chunk_metadatas = data["metadatas"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for cid, text, meta in zip(chunk_ids, chunk_texts, chunk_metadatas):
            f.write(json.dumps({
                "chunk_id": cid,
                "doc_name": meta.get("doc_name", "N/A"),
                "text": text,
            }) + "\n")

    print(f"Dumped {len(chunk_ids)} chunks to {output_path}")


if __name__ == "__main__":
    from src.semantic_chunking import collection  # your Chroma collection

    dump_chunks_readable(collection)
    dump_chunks_jsonl(collection)