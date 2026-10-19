# Evaluation

How retrieval quality was measured, what the numbers mean, and what they don't. All figures come from runs recorded in [`STATUS.md`](../STATUS.md) (2026-09-23); none were produced for this document.

## Setup

| | |
|---|---|
| Benchmark | [EnterpriseRAG-Bench](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench) (`onyx-dot-app`, MIT) question/answer pairs, each linked to one gold source document |
| Test split | **47** held-out questions (`data/finetune/test.jsonl`), split by source document so no test document appears in fine-tuning data (`prepare_data.py`, `stable_split`) |
| Corpus indexed | **50,000 of 511,957** documents (≈9.8%), chunked 1,000 chars / 150 overlap → **311,442** chunks, in both Pinecone and the BM25 index |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-d, cosine) |
| Reranker | `BAAI/bge-reranker-base` cross-encoder, 20 candidates → top 3 |
| Generator | `Qwen2.5-0.5B-Instruct` + QLoRA adapter (1 epoch: train loss 3.442, eval loss 3.213) |
| Harness | `python evaluate_model.py`, which runs every question through the same `EnterpriseRetriever.search()` path the API uses, against the live Pinecone index |
| Hardware | The author's local Windows development machine (CUDA GPU for generation); not a controlled benchmark host |

## Metrics

- **Context Recall@3:** per question, 1 if any of the three reranked chunks comes from the gold document, else 0; reported as the mean (`evaluate_model.py`, `context_recall_at_k`). It measures whether the right *document* reached the generator, not whether the answer is correct.
- **Faithfulness (heuristic):** the fraction of the answer's content words (longer than 3 characters) that also appear in the retrieved chunks (`faithfulness_score`). This is a lexical-overlap proxy, **not** an NLI or LLM-judge faithfulness score. Paraphrased but faithful answers score lower, and an "insufficient context" refusal scores 1.0 by design (declining is not a hallucination).
- **Latency:** mean wall-clock per question for retrieval, reranking and generation together.

## Results

| Configuration | Context Recall@3 | 95% CI (Wilson) | Faithfulness (heuristic) | Mean latency |
|---|---|---|---|---|
| Dense only (Pinecone) → rerank | **0.085** (4/47) | 0.034–0.199 | 0.457 | 14.9 s |
| Hybrid (Pinecone + BM25, weighted RRF 0.5/0.5) → rerank | **0.128** (6/47) | 0.060–0.252 | 0.494 | 14.8 s |

An earlier run against a 500-document index scored **0.000**: the gold documents simply weren't indexed.

## How to read these numbers

1. **The absolute recall is low mainly because of coverage.** With about 9.8% of the corpus indexed, most gold documents are not retrievable at all. Recall ≈ coverage is the expected ceiling, and the dense-only result (8.5%) sits right at it. These numbers say little about how the pipeline would do on a fully indexed corpus.
2. **The hybrid gain is two questions out of 47.** The relative change (+50%) sounds large; the absolute change (+4.3 points) is small, and the confidence intervals overlap heavily. On its own this run does **not** establish that hybrid search is better.
3. **The targeted failure case is the stronger evidence.** For "What is the status of build-job #4572?", dense-only retrieval did not return the gold document anywhere in its top 20. Hybrid retrieval returned it at rank 1 (rerank score 0.524 vs 0.0005 for the next candidate). That is the mechanism BM25 was added for, exact identifiers that dense embeddings blur, and the aggregate result is consistent with it.
4. **Latency is dominated by generation and reranking.** Adding BM25 did not change the mean measurably. 15 s per answer reflects a 0.5B model on a development machine, not a tuned serving setup.
5. **Faithfulness moved in the same direction, but it is a word-overlap heuristic.** Treat it as a smoke test, not a quality claim.

## Limitations and next steps

- **Index the ~423 QA-linked documents** (or the full corpus) so recall measures retrieval quality rather than coverage.
- **Enlarge the test set** and keep per-question results, so a paired test (McNemar) can compare configurations. `evaluation_results.csv` is gitignored, so the per-question rows from these runs aren't in the repo.
- **Replace the faithfulness heuristic** with an NLI model or a Gemini/local LLM judge. The project rules exclude OpenAI as a judge.
- **Apply metadata filters to the BM25 leg.** Department/role filters currently constrain only the dense retriever.
- Record hardware and model versions with every run.

## Reproducing

```bash
# Prerequisites: MSSQL running and seeded, Pinecone index populated, BM25 index built (see README)
python evaluate_model.py              # all 47 questions; prints Recall@3, latency, faithfulness
python evaluate_model.py --limit 5    # quick smoke run (--test-path to use another JSONL split)
```

Results depend on how much of the corpus is indexed: check `GET /health` (`document_count`, `vector_count`) before comparing runs.
