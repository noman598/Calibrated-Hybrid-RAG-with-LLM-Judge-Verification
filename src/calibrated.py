# src/calibration.py

import json
from sklearn.metrics import cohen_kappa_score, accuracy_score

def load_labeled_claims(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def calibrate_judge(labeled_claims: list[dict], chunk_lookup: dict[str, str]) -> dict:
    """
    labeled_claims: list of {"claim", "cited_chunk_id", "human_verdict"}
    Runs the judge on each, compares to human_verdict.
    """
    from .judge import judge_claim  # your Part B function

    human_labels = []
    judge_labels = []
    mismatches = []
    import time
    for item in labeled_claims:
        chunk_text = chunk_lookup[item["cited_chunk_id"]]
        time.sleep(2)
        judge_result = judge_claim(item["claim"], chunk_text)

        human_labels.append(item["human_verdict"])
        judge_labels.append(judge_result["verdict"])

        if item["human_verdict"] != judge_result["verdict"]:
            mismatches.append({
                "claim": item["claim"],
                "human": item["human_verdict"],
                "judge": judge_result["verdict"],
                "judge_reason": judge_result["reason"],
            })

    accuracy = accuracy_score(human_labels, judge_labels)
    kappa = cohen_kappa_score(human_labels, judge_labels)

    return {
        "accuracy": accuracy,
        "cohen_kappa": kappa,
        "num_examples": len(labeled_claims),
        "mismatches": mismatches,
    }


if __name__ == "__main__":
    # chunk_lookup: full_chunk_id -> text, reuse from your Chroma load
    from .fusion import load_corpus_from_chroma
    from .semantic_chunking import collection

    _, _, chunk_lookup = load_corpus_from_chroma(collection)

    labeled_claims = load_labeled_claims("eval_sets/faithfulness_gold.jsonl")
    results = calibrate_judge(labeled_claims, chunk_lookup)

    print(f"Judge accuracy vs human labels: {results['accuracy']:.2%}")
    print(f"Cohen's kappa: {results['cohen_kappa']:.3f}")
    print(f"\n{len(results['mismatches'])} mismatches out of {results['num_examples']}:\n")
    for m in results["mismatches"]:
        print(f"Claim: {m['claim']}")
        print(f"  Human: {m['human']} | Judge: {m['judge']} | Judge reason: {m['judge_reason']}\n")