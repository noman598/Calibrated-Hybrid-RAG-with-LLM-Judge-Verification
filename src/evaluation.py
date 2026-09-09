# src/evaluation.py

import json
from pathlib import Path


def load_gold_set(path: str) -> list[dict]:
    gold = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            gold.append(json.loads(line))
    return gold  # list of {"query": ..., "relevant_chunk_ids": [...]}


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval_method(
    gold_set: list[dict],
    retrieve_fn,   # a function: (query) -> list[chunk_id], ranked best-first
    k: int = 5,
) -> dict:
    """
    retrieve_fn must take a query string and return a ranked list of chunk_ids.
    This lets you plug in BM25-only, dense-only, RRF-fused, or reranked — same eval code for all.
    """
    precisions, recalls, rrs = [], [], []

    for example in gold_set:
        query = example["query"]
        relevant_ids = set(example["relevant_chunk_ids"])

        retrieved_ids = retrieve_fn(query)

        precisions.append(precision_at_k(retrieved_ids, relevant_ids, k))
        recalls.append(recall_at_k(retrieved_ids, relevant_ids, k))
        rrs.append(reciprocal_rank(retrieved_ids, relevant_ids))

    return {
        "precision@k": sum(precisions) / len(precisions),
        "recall@k": sum(recalls) / len(recalls),
        "mrr": sum(rrs) / len(rrs),
        "k": k,
        "num_queries": len(gold_set),
    }


if __name__ == "__main__":
    from .semantic_chunking import collection, embed_sentences
    from .bm25 import BM25
    from .fusion import load_corpus_from_chroma, fuse_retrieval, get_bm25_ranked_ids, get_dense_ranked_ids
    from .rerank import rerank

    gold_set = load_gold_set("eval_sets/retrieval_gold.jsonl")
    chunk_ids, chunk_texts, chunk_lookup = load_corpus_from_chroma(collection)
    bm25 = BM25(chunks=chunk_texts, k1=1.5, b=0.75)

    K = 5

    # --- Method 1: BM25 only ---
    def bm25_only(query):
        return get_bm25_ranked_ids(bm25, chunk_ids, query, top_k=K)

    # --- Method 2: Dense only ---
    def dense_only(query):
        query_vec = embed_sentences([query])[0]
        return get_dense_ranked_ids(query_vec, collection, top_k=K)

    # --- Method 3: RRF fused ---
    def rrf_fused(query):
        fused = fuse_retrieval(query, bm25, chunk_ids, collection, embed_sentences, top_k_per_retriever=20)
        return [cid for cid, score in fused[:K]]

    # --- Method 4: RRF + reranked ---
    def rrf_reranked(query):
        fused = fuse_retrieval(query, bm25, chunk_ids, collection, embed_sentences, top_k_per_retriever=20)
        reranked = rerank(query, fused, chunk_lookup, top_k=K)
        return [r["chunk_id"] for r in reranked]

    print("=== Retrieval Evaluation (k=5) ===\n")
    for name, fn in [
        ("BM25 only", bm25_only),
        ("Dense only", dense_only),
        ("RRF fused", rrf_fused),
        ("RRF + Reranked", rrf_reranked),
    ]:
        results = evaluate_retrieval_method(gold_set, fn, k=K)
        print(f"{name:20s} | Precision@{K}: {results['precision@k']:.3f} | "
              f"Recall@{K}: {results['recall@k']:.3f} | MRR: {results['mrr']:.3f}")


