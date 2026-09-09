# src/chunking.py
from dataclasses import dataclass
from pathlib import Path
import re

@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    text: str
    method: str          # "naive" or "semantic" (added in Step 4)
    start_char: int
    end_char: int

def simple_tokenize(text: str) -> list[str]:
    """Whitespace tokenization - good enough for chunk sizing, not for BM25 (that needs its own)."""
    return text.split()

def naive_fixed_chunks(
    text: str,
    doc_name: str,
    chunk_size: int = 200,      # tokens per chunk
    overlap: int = 20,          # tokens of overlap between consecutive chunks
) -> list[Chunk]:
    """Split text into fixed-size overlapping chunks. This is the naive baseline —
    it will cut sentences/ideas in half, that's the point (to demonstrate the weakness)."""
    tokens = simple_tokenize(text)
    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than chunk_size")

    i = 0
    chunk_idx = 0
    while i < len(tokens):
        window = tokens[i : i + chunk_size]
        if not window:
            break
        chunk_text = " ".join(window)

        # approximate char offsets (not exact since we tokenized on whitespace,
        # good enough for traceability, not for precise citation highlighting)
        start_char = len(" ".join(tokens[:i]))
        end_char = start_char + len(chunk_text)

        chunks.append(Chunk(
            chunk_id=f"{doc_name}_naive_{chunk_idx}",
            doc_name=doc_name,
            text=chunk_text,
            method="naive",
            start_char=start_char,
            end_char=end_char,
        ))
        chunk_idx += 1
        i += step

    return chunks

def chunk_all_documents(
    cleaned_text_dir: Path,
    chunk_size: int = 200,
    overlap: int = 20,
) -> list[Chunk]:
    all_chunks = []
    for txt_path in cleaned_text_dir.glob("*.txt"):
        text = txt_path.read_text(encoding="utf-8")
        doc_chunks = naive_fixed_chunks(text, doc_name=txt_path.stem, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(doc_chunks)
    return all_chunks

if __name__ == "__main__":
    chunks = chunk_all_documents(Path("data/cleaned_text"))
    print(f"Generated {len(chunks)} naive chunks")
    print(f"Example chunk:\n{chunks[0]}")

    