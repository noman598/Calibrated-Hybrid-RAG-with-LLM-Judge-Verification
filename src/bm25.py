import math
import re
from collections import Counter, defaultdict
import chromadb

chroma_client = chromadb.PersistentClient(path="data/chroma_db")
collection = chroma_client.get_or_create_collection(name="course_chunks")

class BM25:

    def __init__(self, chunks: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks = chunks
        self.N = len(chunks)

        # Step 1 & 3: Tokenize, build vocabulary stats, and chunk lengths
        self.tokenized_chunks = []
        self.chunk_lengths = []
        self.tf = []  # List of Counter objects (one per chunk)
        df = defaultdict(int)

        total_tokens = 0

        for chunk in chunks:
            tokens = self._tokenize(chunk)
            self.tokenized_chunks.append(tokens)

            chunk_len = len(tokens)
            self.chunk_lengths.append(chunk_len)
            total_tokens += chunk_len

            # Calculate raw TF for this chunk
            term_counts = Counter(tokens)
            self.tf.append(term_counts)

            # Update Document Frequency (df) using unique terms
            for term in term_counts.keys():
                df[term] += 1

        # Calculate average document length (avgdl)
        self.avgdl = total_tokens / self.N if self.N > 0 else 0

        # Step 2: Precompute IDF for all terms in corpus
        self.idf = self._compute_idf(df)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _compute_idf(self, df: dict[str, int]) -> dict[str, float]:
        """Smoothed Robertson-Spärck Jones IDF formula."""
        idf = {}
        for term, term_df in df.items():
            numerator = self.N - term_df + 0.5
            denominator = term_df + 0.5
            idf[term] = math.log((numerator / denominator) + 1.0)
        return idf

    def score_chunk(self, query_tokens: list[str], chunk_idx: int) -> float:
        """Step 4: Computes the BM25 score for a single chunk against query tokens."""
        score = 0.0
        doc_len = self.chunk_lengths[chunk_idx]
        doc_tf = self.tf[chunk_idx]

        # Calculate length normalization factor
        len_norm = 1.0 - self.b + self.b * (doc_len / self.avgdl)

        for term in query_tokens:
            if term not in doc_tf:
                continue

            term_idf = self.idf.get(term, 0.0)
            term_freq = doc_tf[term]

            # Saturation term frequency numerator and denominator
            num = term_freq * (self.k1 + 1.0)
            denom = term_freq + self.k1 * len_norm

            score += term_idf * (num / denom)

        return score

    def search(self, query: str, top_k: int = 3) -> list[tuple[int, float, str]]:
        """Step 5: Ranks all chunks for a given query and returns top-k matches."""
        query_tokens = self._tokenize(query)
        scores = []

        for idx in range(self.N):
            s = self.score_chunk(query_tokens, idx)
            scores.append((idx, s, self.chunks[idx]))

        # Sort descending by score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# --- Step 6: Sanity Testing ---
if __name__ == "__main__":
    all_data = collection.get(include=["documents", "metadatas"])
    chunk_ids = all_data["ids"]
    chunk_texts = all_data["documents"]
    chunk_metadatas = all_data["metadatas"]

    bm25 = BM25(chunks=chunk_texts, k1=1.5, b=0.75)

    # Sanity Test Query
    query = "scaled dot-product attention"
    results = bm25.search(query, top_k=2)

    print(f"Query: '{query}'\n")
    for rank, (idx, score, text) in enumerate(results, 1):
        print(f"Rank {rank} (Chunk #{idx}) | Score: {score:.4f}")
        print(f"Text: \"{text}\"\n")