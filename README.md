<div align="center">

# 🚀 TrustRAG

**A high-accuracy RAG pipeline built with Python, ChromaDB, and Sentence-Transformers. It improves retrieval using adaptive chunking and Reciprocal Rank Fusion, and uses a calibrated LLM-judge to verify that all generated claims are actually grounded in the source text.**

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge\&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116-009688?style=for-the-badge\&logo=fastapi)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-316192?style=for-the-badge\&logo=postgresql)
![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-Embeddings-blue?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Database-orange?style=for-the-badge)


</div>

---

# 🧠 What is the Calibrated Hybrid RAG Pipeline?

It is a precision-focused retrieval system designed to pull the most relevant information from documents and absolutely ensure the AI doesn't hallucinate.
this system mathematically fuses different search methods to find the best context, and then puts the final answer on trial using an automated "judge" to verify every single claim against the source text.

* Hybrid RAG + Custom-implemented BM25
* Adaptive semantic chunking
* AI reranking
* Structured source-grounded answers
* Dense vector embeddings
* Reciprocal Rank Fusion
* Cross-encoder reranking
* Claim-level faithfulness verification
  
# 🏗️ High-Level Architecture

```text
                    USER QUERY
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
         BM25 Search        Vector Search
              │                   │
              └─────────┬─────────┘
                        ▼
              Reciprocal Rank Fusion
                        │
                        ▼
              Cross-Encoder Reranking
                        │
                        ▼
                Answer Generation
                        │
                        ▼
             LLM-Judge Verification
                (Grounded Output)
```


---

## 🔍 Hybrid Retrieval Strategy

### Vector Retrieval
Captures semantic similarity using **Sentence-Transformers** and **ChromaDB**.
Useful when the question and document express the same idea using different wording.

### Custom BM25
Captures lexical relevance through a hand-implemented search engine.
Useful for:
- Exact terminology
- Names and identifiers
- Uncommon phrases
- Keyword-heavy questions

Combining these two signals provides:
```text
Semantic Understanding (Vector)
        +
Lexical Precision (BM25)
```

### Reciprocal Rank Fusion (RRF)
Running two different search methods means you get two different lists of results. To determine the absolute best matches, this project mathematically merges them using Reciprocal Rank Fusion (RRF).

Conceptually:

```text
Dense Vector Search      Custom BM25 Search
       │                          │
       └───────────┬────────────┘
                   ▼
         Reciprocal Rank Fusion
                   │
                   ▼
         Unified Candidate Pool
```

### Adaptive Pipeline & LLM-Judge Verification
To guarantee high accuracy and eliminate hallucinations, the pipeline doesn't just blindly feed text to an AI. It filters and verifies information at every step.
-Adaptive Semantic Chunking
-Hybrid Search & Reranking
-Claim-Level LLM Judge

```text
Document Text
      │
      ▼
Adaptive Semantic Chunking
      │
      ▼
Hybrid Retrieval (Vector + BM25)
      │
      ▼
Cross-Encoder Reranking
      │
      ▼
Answer Generation
      │
      ▼
Claim-Level LLM Judge (92% Accuracy)
      │
      ├── FAIL -> Reject/Regenerate
      └── PASS -> Final Grounded Output
```


# 📈 Performance & Calibration

The Calibrated Hybrid RAG pipeline was benchmarked against a hand-labeled evaluation set. Instead of just measuring speed, this system is optimized and tested for **retrieval accuracy** and **hallucination prevention**.

### Retrieval Metrics

Combining custom BM25 with dense embeddings via Reciprocal Rank Fusion (RRF), and applying cross-encoder reranking, yielded significant improvements over single-method baselines:

| Metric | Baseline (Single Strategy) | Calibrated (Fused + Reranked) |
| :--- | :--- | :--- |
| **Mean Reciprocal Rank (MRR)** | 0.58 *(BM25-only)* | **0.75** |
| **Recall@5** | 0.63 *(Single-retriever)* | **0.75** |

### LLM-Judge Verification Metrics

To ensure the generated answers are actually grounded in the retrieved context, the claim-level faithfulness judge was calibrated against human evaluators:

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Verification Accuracy** | **92%** | High reliability in catching ungrounded claims. |
| **Cohen’s Kappa (κ)** | **0.776** | Strong statistical agreement with human judgment. |


---

## 🚀 Future Roadmap

While the current pipeline focuses strictly on high-precision retrieval and hallucination prevention, the next phase of development will evolve this core engine into a fully asynchronous, agentic document intelligence platform. 

Upcoming features and architectural upgrades include:

### 🤖 Agentic Conversational AI
- **Stateful Memory Systems:** Implementing short-term conversational context, current-session, and cross-session semantic memory.
- **Dynamic Tool Selection:** Upgrading the LLM to route queries autonomously to specialized tools (e.g., Web Search, Document QA, Intent Classification).
- **Advanced Query Transformations:** Adding adaptive routing techniques like HyDE, Step-Back Prompting, and Query Decomposition.

### 🏗️ Production Infrastructure (Async & Scale)
- **Asynchronous Processing:** Integrating **FastAPI, Celery, and Redis** for background document parsing and user-isolated indexing.
- **Structure-Aware Parsing:** Utilizing advanced parsing tools (like Docling) to understand document layouts and hierarchies before chunking.
- **Multi-Index RAG & Caching:** Implementing three-tier semantic caching and multi-index strategies backed by **PostgreSQL** for enterprise-scale deployments.

### Future Target Architecture
```text
                    USER INTERFACE
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
      DOCUMENT QA TOOL          CONVERSATIONAL AI
    (The Current Engine)      (Memory + Tool Routing)
              │                       │
              ▼                       ▼
      Hybrid Retrieval           Agentic Actions
              │                       │
              ▼                       ▼
      LLM-Judge Verified        Context-Aware Output
