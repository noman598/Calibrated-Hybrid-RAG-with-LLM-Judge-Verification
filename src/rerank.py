# src/rerank.py

from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, fused_results: list[tuple[str, float]], chunk_lookup: dict[str, str], top_k: int = 5) -> list[dict]:
    """
    chunk_lookup: chunk_id -> text (from load_corpus_from_chroma)
    """
    candidate_ids = [chunk_id for chunk_id, _ in fused_results]
    pairs = [(query, chunk_lookup[cid]) for cid in candidate_ids]

    scores = reranker.predict(pairs)

    reranked = []
    for (chunk_id, rrf_score), rerank_score in zip(fused_results, scores):
        reranked.append({
            "chunk_id": chunk_id,
            "text": chunk_lookup[chunk_id],
            "rrf_score": rrf_score,
            "rerank_score": float(rerank_score),
        })

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


if __name__ == "__main__":
    from src.semantic_chunking import embed_sentences, collection
    from src.bm25 import BM25
    from fusion import load_corpus_from_chroma, fuse_retrieval

    chunk_ids, chunk_texts, chunk_lookup = load_corpus_from_chroma(collection)
    bm25 = BM25(chunks=chunk_texts, k1=1.5, b=0.75)

    query = "multi-head attention"
    fused_results = fuse_retrieval(query, bm25, chunk_ids, collection, embed_sentences, top_k_per_retriever=20)

    print("--- Before reranking (RRF top 5) ---")
    for chunk_id, score in fused_results[:5]:
        print(f"RRF={score:.4f} | {chunk_lookup[chunk_id][:100]}")

    reranked = rerank(query, fused_results, chunk_lookup, top_k=5)

    print("\n--- After reranking (top 5) ---")
    for r in reranked:
        print(f"rerank={r['rerank_score']:.4f} (was RRF={r['rrf_score']:.4f}) | {r['text'][:100]}")