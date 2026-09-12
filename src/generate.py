# src/generate.py

from utils import Config
from groq import Groq

client = Groq(api_key=Config.GROQ_API_KEY)  # swap for whichever LLM you're using

GENERATION_PROMPT_TEMPLATE = """You are answering a question using ONLY the provided context chunks.
Do not use any outside knowledge.

Context:
{context_block}

Question: {query}

Instructions:
- Answer using only the information in the context above.
- After each claim, cite the chunk id it came from, like [chunk_7].
- If a claim draws on multiple chunks, cite all of them, like [chunk_7][chunk_10].
- If the context does not contain enough information to answer, say "I cannot answer this from the given context" — do not guess.

Answer:"""


def build_context_block(retrieved_chunks: list[dict]) -> str:
    """
    retrieved_chunks: list of {"chunk_id": ..., "text": ...}
    Formats them as labeled context for the prompt.
    """
    lines = []
    for c in retrieved_chunks:
        short_id = c["chunk_id"].split("_")[-1]  # e.g. "7" from "..._semantic_7"
        lines.append(f"[chunk_{short_id}]: {c['text']}")
    return "\n".join(lines)


def generate_answer(query: str, retrieved_chunks: list[dict], model: str = "qwen/qwen3.6-27b") -> str:
    context_block = build_context_block(retrieved_chunks)
    prompt = GENERATION_PROMPT_TEMPLATE.format(context_block=context_block, query=query)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # deterministic-ish, reduces hallucination variance
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # quick manual test
    test_chunks = [
        {"chunk_id": "doc_semantic_7", "text": "Attention(Q,K,V) = softmax(QK^T / sqrt(dk)) V"},
    ]
    answer = generate_answer("What formula is used for attention?", test_chunks)
    print(answer)