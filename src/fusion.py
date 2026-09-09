import numpy as np


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Combines multiple ranked lists of chunk_ids into a single fused ranking."""
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def get_dense_ranked_ids(query_embedding: np.ndarray, collection, top_k: int = 20) -> list[str]:
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
    )
    return results["ids"][0]


def load_corpus_from_chroma(collection) -> tuple[list[str], list[str], dict[str, str]]:
    """
    Pull everything already indexed in Chroma — no re-chunking needed.
    Returns: chunk_ids (order matches texts), chunk_texts, and a chunk_id -> text lookup.
    """
    data = collection.get(include=["documents", "metadatas"])
    chunk_ids = data["ids"]
    chunk_texts = data["documents"]
    chunk_lookup = {cid: text for cid, text in zip(chunk_ids, chunk_texts)}
    return chunk_ids, chunk_texts, chunk_lookup


def get_bm25_ranked_ids(bm25, chunk_ids: list[str], query: str, top_k: int = 20) -> list[str]:
    """
    bm25 was built over chunk_texts in the SAME order as chunk_ids (see build step below).
    bm25.search returns (idx, score, text) — map idx back to chunk_id via that shared order.
    """
    results = bm25.search(query, top_k=top_k)
    return [chunk_ids[idx] for idx, score, text in results]


def fuse_retrieval(
    query: str,
    bm25,
    chunk_ids: list[str],
    collection,
    embed_fn,
    top_k_per_retriever: int = 20,
    k: int = 60,
) -> list[tuple[str, float]]:
    bm25_ids = get_bm25_ranked_ids(bm25, chunk_ids, query, top_k=top_k_per_retriever)

    query_vec = embed_fn([query])[0]
    dense_ids = get_dense_ranked_ids(query_vec, collection, top_k=top_k_per_retriever)

    return reciprocal_rank_fusion([bm25_ids, dense_ids], k=k)

if __name__ == "__main__":
    # Example wiring — adjust imports based on your actual file/module names
    from pathlib import Path
    from src.semantic_chunking import embed_sentences, collection
    from src.bm25 import BM25

    chunk_ids, chunk_texts, chunk_lookup = load_corpus_from_chroma(collection)
    bm25 = BM25(chunks=chunk_texts, k1=1.5, b=0.75)   # built once from Chroma's stored texts

    query = "multi-head attention"
    fused_results = fuse_retrieval(query, bm25, chunk_ids, collection, embed_sentences, top_k_per_retriever=20)

    print(f"Fused RRF results for: '{query}'\n")
    for chunk_id, score in fused_results[:5]:
        print(f"RRF score={score:.4f} | {chunk_lookup[chunk_id][:100]}")