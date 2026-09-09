import spacy
from pathlib import Path
from dataclasses import dataclass
import numpy as np
# from google import genai
# from google.genai.types import EmbedContentConfig
from src.utils import Config
import chromadb
from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    text: str
    method: str
    start_char: int
    end_char: int


# --- Clients (kept separate — do NOT reuse the same variable name for both) ---
nlp = spacy.load("en_core_web_sm")
# genai_client = genai.Client(api_key=Config.GEMINI_API_KEY)
chroma_client = chromadb.PersistentClient(path="data/chroma_db")
collection = chroma_client.get_or_create_collection(name="course_chunks")


def split_sentences_spacy(text: str) -> list[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


# def embed_sentences(sentences: list[str], batch_size: int = 100) -> np.ndarray:
#     """Embed a list of texts (sentences OR chunks) via Gemini, batched."""
#     all_vectors = []
#     for i in range(0, len(sentences), batch_size):
#         batch = sentences[i : i + batch_size]
#         result = genai_client.models.embed_content(
#             model="gemini-embedding-001",
#             contents=batch,
#             config=EmbedContentConfig(
#                 task_type="RETRIEVAL_DOCUMENT",
#                 output_dimensionality=1536,
#             ),
#         )
#         vectors = [e.values for e in result.embeddings]
#         all_vectors.extend(vectors)
#     return np.array(all_vectors)


# Load once, reuse everywhere (loading the model is the expensive part)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_sentences(sentences: list[str], batch_size: int = 32) -> np.ndarray:
    """Local embedding via sentence-transformers — no API calls, no rate limits, free."""
    embeddings = embed_model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings



def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def calculate_similarities(embeddings: np.ndarray) -> list[float]:
    return [
        cosine_similarity(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]


def adaptive_semantic_chunking(
    sentences: list[str],
    embeddings: np.ndarray,
    k: float = 1.25,
    window_size: int = 3,
    min_tokens: int = 50,
    max_tokens: int = 300,
) -> list[str]:
    """Chunks sentences dynamically based on adaptive similarity drops and size limits."""
    if not sentences:
        return []
    if len(sentences) == 1:
        return [sentences[0]]

    similarities = calculate_similarities(embeddings)
    num_sims = len(similarities)

    split_decisions = []
    for i in range(num_sims):
        start_idx = max(0, i - window_size)
        end_idx = min(num_sims, i + window_size + 1)
        window = similarities[start_idx:end_idx]
        local_mean = np.mean(window)
        local_std = np.std(window)
        threshold = local_mean - (k * local_std)
        split_decisions.append(similarities[i] < threshold)

    chunks = []
    current_chunk = []
    current_token_count = 0

    for i, sentence in enumerate(sentences):
        sentence_tokens = len(sentence.split())
        force_split = (
            current_token_count + sentence_tokens > max_tokens
            and current_token_count >= min_tokens
        )
        adaptive_split = (
            split_decisions[i - 1] if (i > 0 and i - 1 < len(split_decisions)) else False
        )

        if (adaptive_split and current_token_count >= min_tokens) or force_split:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_token_count = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_token_count += sentence_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def build_chunk_objects(chunk_texts: list[str], doc_name: str, full_text: str) -> list[Chunk]:
    """Wrap raw chunk strings into Chunk objects with metadata + approximate offsets."""
    chunk_objs = []
    search_start = 0
    for idx, text in enumerate(chunk_texts):
        start_char = full_text.find(text[:50], search_start)
        if start_char == -1:
            start_char = search_start
        end_char = start_char + len(text)
        chunk_objs.append(Chunk(
            chunk_id=f"{doc_name}_semantic_{idx}",
            doc_name=doc_name,
            text=text,
            method="semantic",
            start_char=start_char,
            end_char=end_char,
        ))
        search_start = end_char
    return chunk_objs


def index_chunks(chunks: list[Chunk], embeddings: np.ndarray):
    """Upsert chunks into Chroma — upsert (not add) so reruns don't error/duplicate."""
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings.tolist(),
        documents=[c.text for c in chunks],
        metadatas=[{"doc_name": c.doc_name, "method": c.method} for c in chunks],
    )


def dense_search(query_embedding: np.ndarray, top_k: int = 5):
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
    )
    return results


def chunk_all_documents(
    cleaned_text_dir: Path,
    k: float = 1.25,
    window_size: int = 3,
    min_tokens: int = 50,
    max_tokens: int = 300,
) -> list[Chunk]:
    all_chunks = []
    for txt_path in cleaned_text_dir.glob("*.txt"):
        text = txt_path.read_text(encoding="utf-8")

        sentences = split_sentences_spacy(text)
        if not sentences:
            print(f"Skipping {txt_path.name} — no sentences found")
            continue

        sentence_embeddings = embed_sentences(sentences)
        print(f"{txt_path.stem}: {len(sentences)} sentences -> embeddings shape {sentence_embeddings.shape}")

        chunk_texts = adaptive_semantic_chunking(
            sentences=sentences,
            embeddings=sentence_embeddings,
            k=k,
            window_size=window_size,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )

        # Embed the CHUNKS themselves (not sentences) — this is what gets indexed for dense retrieval
        chunk_vectors = embed_sentences(chunk_texts, batch_size=10)

        doc_chunks = build_chunk_objects(chunk_texts, doc_name=txt_path.stem, full_text=text)
        index_chunks(doc_chunks, chunk_vectors)
        all_chunks.extend(doc_chunks)

        print(f"{txt_path.stem}: {len(chunk_texts)} semantic chunks created and indexed")

    return all_chunks


if __name__ == "__main__":
    chunks = chunk_all_documents(Path("data/cleaned_text"))
    print(f"\nGenerated {len(chunks)} total semantic chunks")
    if chunks:
        print(f"\nExample chunk:\n{chunks[0]}")
        print(f"\nChroma collection count: {collection.count()}")

        