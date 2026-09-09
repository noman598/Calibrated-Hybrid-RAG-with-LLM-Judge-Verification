# # tests/test_ingestion.py
# from src.ingestion import clean_page_text, is_near_empty, detect_repeated_lines

# def test_removes_page_numbers():
#     text = "Some real content here\n42"
#     boilerplate = set()
#     cleaned = clean_page_text(text, boilerplate)
#     assert "42" not in cleaned.split("\n")

# def test_joins_broken_sentences():
#     text = "the model learns\nrepresentations from data."
#     cleaned = clean_page_text(text, set())
#     assert "the model learns representations from data." in cleaned

# def test_near_empty_detection():
#     assert is_near_empty("Chapter 3")
#     assert not is_near_empty("This is a much longer sentence with real content in it, spanning many words.")

import sys
sys.path.append("./src")

from src.semantic_chunking import chunk_all_documents, embed_sentences,dense_search
from pathlib import Path
from src.bm25 import BM25

# BM25
chunks = chunk_all_documents(Path("data/cleaned_text"))
chunk_texts = [c.text for c in chunks]
bm25 = BM25(chunks=chunk_texts, k1=1.5, b=0.75)

query = "positional encoding"
bm25_results = bm25.search(query, top_k=3)
for idx, score, text in bm25_results:
    print(f"BM25 | score={score:.3f} | {text[:100]}")

# Dense (Chroma)
query_vec = embed_sentences([query])[0]
dense_results = dense_search(query_vec, top_k=3)
for doc in dense_results["documents"][0]:
    print(f"Dense | {doc[:100]}")