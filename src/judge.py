# src/judge.py

import re
from .utils import Config
from groq import Groq

client = Groq(api_key=Config.GROQ_API_KEY)

JUDGE_PROMPT_TEMPLATE = """You are verifying whether a claim is faithfully supported by its cited source text.

Cited source:
{chunk_text}

Claim: "{claim}"

Question: Is this claim fully supported by the cited source above?
Respond in exactly this format:
VERDICT: <SUPPORTED | PARTIALLY_SUPPORTED | NOT_SUPPORTED>
REASON: <one sentence explaining why>
"""


def extract_claims(answer_text: str) -> list[dict]:
    """
    Splits the generated answer into sentences, extracts any [chunk_X] citations
    attached to each sentence. Returns list of {"claim": ..., "cited_chunk_ids": [...]}.
    """
    sentences = re.split(r'(?<=[.!?])\s+', answer_text.strip())
    claims = []
    for sent in sentences:
        cited = re.findall(r'\[chunk_(\w+)\]', sent)
        if cited:
            claim_text = re.sub(r'\[chunk_\w+\]', '', sent).strip()
            claims.append({
                "claim": claim_text,
                "cited_chunk_ids": cited,  # e.g. ["7", "10"]
            })
    return claims


def judge_claim(claim: str, chunk_text: str, model: str = "qwen/qwen3.6-27b") -> dict:
    prompt = JUDGE_PROMPT_TEMPLATE.format(chunk_text=chunk_text, claim=claim)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100,
        reasoning_effort="none"
        # extra_body={"chat_template_kwargs": {"enable_thinking": False}}  # Disables CoT
    )
    raw = response.choices[0].message.content

    verdict_match = re.search(r'VERDICT:\s*(\w+)', raw)
    reason_match = re.search(r'REASON:\s*(.+)', raw)

    return {
        "verdict": verdict_match.group(1) if verdict_match else "UNKNOWN",
        "reason": reason_match.group(1).strip() if reason_match else raw,
    }


def score_answer_faithfulness(answer_text: str, chunk_lookup: dict[str, str], short_id_to_full_id: dict[str, str]) -> dict:
    """
    chunk_lookup: full_chunk_id -> chunk text
    short_id_to_full_id: e.g. {"7": "doc_semantic_7"} — maps citation shorthand back to real chunk_id
    """
    claims = extract_claims(answer_text)
    results = []

    for c in claims:
        # if multiple chunks cited, check against each and take the weakest verdict
        verdicts = []
        for short_id in c["cited_chunk_ids"]:
            full_id = short_id_to_full_id.get(short_id)
            if not full_id or full_id not in chunk_lookup:
                verdicts.append({"verdict": "NOT_SUPPORTED", "reason": "cited chunk not found"})
                continue
            chunk_text = chunk_lookup[full_id]
            verdicts.append(judge_claim(c["claim"], chunk_text))

        results.append({
            "claim": c["claim"],
            "cited_chunk_ids": c["cited_chunk_ids"],
            "verdicts": verdicts,
        })

    total = len(results)
    supported = sum(1 for r in results if all(v["verdict"] == "SUPPORTED" for v in r["verdicts"]))
    faithfulness_score = supported / total if total > 0 else None

    return {
        "claims": results,
        "faithfulness_score": faithfulness_score,
        "total_claims": total,
        "supported_claims": supported,
    }


if __name__ == "__main__":
    chunk_lookup = {
        "doc_semantic_7": "Attention(Q,K,V) = softmax(QK^T / sqrt(dk)) V",
    }
    short_id_to_full_id = {"7": "doc_semantic_7"}

    test_answer = "The attention formula is softmax(QK^T / sqrt(dk)) V [chunk_7]. The model uses 8 attention heads [chunk_7]."

    result = score_answer_faithfulness(test_answer, chunk_lookup, short_id_to_full_id)
    print(f"Faithfulness score: {result['faithfulness_score']:.2f}")
    for r in result["claims"]:
        print(f"\nClaim: {r['claim']}")
        for v in r["verdicts"]:
            print(f"  {v['verdict']} — {v['reason']}")